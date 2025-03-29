import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
import pandas as pd
from tqdm import tqdm
import logging
import wandb

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
MASK_ID = 126336  # The token id of [MASK] in LLaDA (from GUIDELINES.md line 16)

class SFTDataset(Dataset):
    """
    Dataset for Supervised Fine-Tuning based on Question/Answer pairs.

    Formats data using the tokenizer's chat template, e.g., for Instruct models:
    <|im_start|>user\nQuestion<|im_end|>\n<|im_start|>assistant\nAnswer<|im_end|>

    Calculates the length of the prompt part for SFT loss calculation.
    """
    def __init__(self, csv_path, tokenizer, max_length=4096):
        """
        Args:
            csv_path (str): Path to the CSV file with 'question' and 'answer' columns.
            tokenizer: The tokenizer instance.
            max_length (int): Maximum sequence length for padding/truncation.
        """
        logging.info(f"Loading dataset from: {csv_path}")
        try:
            self.dataframe = pd.read_csv(csv_path)
            # Ensure required columns exist
            if 'question' not in self.dataframe.columns or 'answer' not in self.dataframe.columns:
                raise ValueError("CSV must contain 'question' and 'answer' columns.")
            # Handle potential NaN values
            self.dataframe.dropna(subset=['question', 'answer'], inplace=True)
            self.dataframe = self.dataframe.astype({"question": str, "answer": str}) # Ensure string type
            logging.info(f"Loaded {len(self.dataframe)} QA pairs.")
        except FileNotFoundError:
            logging.error(f"CSV file not found at {csv_path}")
            raise
        except Exception as e:
            logging.error(f"Error loading or processing CSV: {e}")
            raise

        self.tokenizer = tokenizer
        self.max_length = max_length

        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            logging.warning("Tokenizer does not have a pad token. Using eos_token for padding.")
            self.tokenizer.pad_token = self.tokenizer.eos_token
            if self.tokenizer.pad_token is None: # If EOS is also None
                 raise ValueError("Tokenizer needs a pad_token or eos_token for padding.")

        if len(self.dataframe) > 0:
            logging.info(f"Dataset example: Q: {self.dataframe.iloc[0]['question']} A: {self.dataframe.iloc[0]['answer']}")

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        """
        Processes a single QA pair, tokenizes it using the chat template,
        and calculates the prompt length.
        """
        item = self.dataframe.iloc[idx]
        question = item["question"]
        answer = item["answer"]

        messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer}
        ]
        try:
            # Tokenize the full conversation
            tokenized_output = self.tokenizer.apply_chat_template(
                messages,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
                add_generation_prompt=False
            )

            # --- Handle inconsistent return type ---
            # Check if the output is a dictionary (expected) or a tensor (observed)
            if isinstance(tokenized_output, dict):
                # Standard case: extract tensor from dictionary
                if "input_ids" not in tokenized_output:
                     logging.error(f"Tokenizer output dictionary missing 'input_ids' key at index {idx}")
                     raise ValueError(f"Tokenizer returned dict without 'input_ids' for item {idx}")
                input_ids_tensor = tokenized_output["input_ids"]
            elif isinstance(tokenized_output, torch.Tensor):
                # Observed case: use the returned tensor directly
                input_ids_tensor = tokenized_output
                # Optional: Log this occurrence if you want to track it
                # logging.warning(f"Tokenizer returned Tensor directly at index {idx}.")
            else:
                # Unexpected case: raise an error
                logging.error(f"Unexpected output type from tokenizer at index {idx}: {type(tokenized_output)}")
                raise TypeError(f"Tokenizer returned unexpected type {type(tokenized_output)} for item {idx}")
            # --- End handling ---


            # --- Process the input_ids tensor ---
            # Now, ensure the tensor has the expected shape and remove the batch dim
            if input_ids_tensor.dim() == 2 and input_ids_tensor.shape[0] == 1:
                 input_ids = input_ids_tensor.squeeze(0) # Remove batch dimension -> shape (seq_len,)
            # Handle edge case where it might already be 1D (less likely with padding="max_length")
            elif input_ids_tensor.dim() == 1:
                 logging.warning(f"Tokenizer returned 1D tensor at index {idx}. Using directly.")
                 input_ids = input_ids_tensor
            else:
                # Log unexpected shape and raise error
                logging.error(f"Unexpected shape for tokenized input_ids at index {idx}: {input_ids_tensor.shape}")
                raise ValueError(f"Tokenizer returned unexpected shape {input_ids_tensor.shape} for item {idx}")
            # --- End processing ---


            # --- Calculate prompt length (remains the same) ---
            messages_prompt_only = [{"role": "user", "content": question}]
            prompt_text = self.tokenizer.apply_chat_template(
                messages_prompt_only,
                add_generation_prompt=True,
                tokenize=False
            )
            # Tokenize prompt separately to get its length accurately
            prompt_tokenized = self.tokenizer(
                prompt_text,
                truncation=False, # Don't truncate prompt part calculation
                add_special_tokens=False, # Assume template includes necessary special tokens
                return_tensors="pt"
            )
            prompt_length = prompt_tokenized["input_ids"].shape[1]
            prompt_length = min(prompt_length, self.max_length)
            # --- End prompt length calculation ---

        except Exception as e:
            logging.error(f"Error processing item {idx}: Q: '{question}', A: '{answer}'. Error: {e}")
            # Add context about the problematic tokenized object if possible
            if 'tokenized_output' in locals():
                 logging.error(f"Problematic tokenized object structure: {type(tokenized_output)}")
                 # Log tensor details if it was a tensor
                 if isinstance(tokenized_output, torch.Tensor):
                      logging.error(f"Problematic tensor: shape={getattr(tokenized_output, 'shape', 'N/A')}, value={str(tokenized_output)[:200]}...") # Truncate print
            raise

        return {
            "input_ids": input_ids,
            "prompt_length": torch.tensor(prompt_length, dtype=torch.long)
        }

def forward_process(input_ids, eps=1e-3):
    """
    Adds noise to input_ids by masking tokens randomly based on a dynamic probability.
    (Based on GUIDELINES.md lines 25-34)

    Args:
        input_ids: Tensor of token ids (batch_size, seq_len)
        eps: Small constant for mask probability calculation

    Returns:
        noisy_batch: Token ids with some replaced by MASK_ID
        masked_indices: Boolean tensor indicating which tokens were masked initially
        p_mask: Probability mask used for masking each token (batch_size, seq_len)
    """
    b, l = input_ids.shape
    # Generate a random noise level 't' for each sequence in the batch
    t = torch.rand(b, device=input_ids.device)
    # Calculate the mask probability for each sequence (linear schedule)
    p_mask_per_sequence = (1 - eps) * t + eps
    # Expand the probability to cover the full sequence length
    p_mask = p_mask_per_sequence[:, None].repeat(1, l)

    # Determine which tokens to mask based on p_mask
    masked_indices = torch.rand((b, l), device=input_ids.device) < p_mask
    # Replace masked tokens with MASK_ID
    noisy_batch = torch.where(masked_indices, MASK_ID, input_ids)

    return noisy_batch, masked_indices, p_mask

def main():
    # --- Hyperparameters & Config ---
    # You can also move these to argparse if preferred
    config = {
        "model_path": "GSAI-ML/LLaDA-8B-Instruct",
        "dataset_csv": "dataset/output/dataset/training.csv", # <--- *** SET THIS PATH ***
        "output_dir": "./LLaDA-8B-Instruct-finetuned-wandb",
        "epochs": 10,
        "batch_size": 1, # Physical batch size per device
        "learning_rate": 1e-5,
        "max_length": 1024,
        "warmup_ratio": 0.1,
        "save_steps": 500, # Save checkpoint every N global steps
        "gradient_accumulation_steps": 4,
        "wandb_run_name": "llada-reversal", # <-- Set specific run name, or None for auto-generated
        "log_steps": 5, # <-- Log metrics every N global steps
    }
    # --- End Hyperparameters & Config ---


    # --- Initialize W&B ---
    wandb.init(
        name=config["wandb_run_name"],
        config=config # Log all hyperparameters
    )
    # Use wandb.config for accessing hyperparameters after init
    config = wandb.config
    # --- End W&B Init ---


    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    wandb.config.update({"device": str(device)}) # Log device info

    # Load model and tokenizer
    logging.info(f"Loading model and tokenizer from {config.model_path}")
    # Use bf16 for potential speedup and memory saving if supported
    model_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    logging.info(f"Using model dtype: {model_dtype}")
    wandb.config.update({"model_dtype": str(model_dtype)}) # Log dtype

    tokenizer = AutoTokenizer.from_pretrained(config.model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        config.model_path,
        trust_remote_code=True,
        torch_dtype=model_dtype
    ).to(device)

    # Load dataset and dataloader
    try:
        train_dataset = SFTDataset(config.dataset_csv, tokenizer, max_length=config.max_length)
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
        wandb.finish(exit_code=1) # <-- Finish W&B run with error code
        return # Exit if dataset fails

    # Adjust effective batch size
    effective_batch_size = config.batch_size * config.gradient_accumulation_steps
    logging.info(f"Physical Batch Size: {config.batch_size}, Grad Accumulation Steps: {config.gradient_accumulation_steps}, Effective Batch Size: {effective_batch_size}")
    wandb.config.update({"effective_batch_size": effective_batch_size}) # Log effective batch size


    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    num_training_batches_per_epoch = len(train_dataloader)
    total_steps = (num_training_batches_per_epoch // config.gradient_accumulation_steps) * config.epochs
    num_warmup_steps = int(config.warmup_ratio * total_steps)
    logging.info(f"Total optimization steps: {total_steps}, Warmup steps: {num_warmup_steps}")
    wandb.config.update({"total_steps": total_steps, "num_warmup_steps": num_warmup_steps}) # Log steps

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_steps
    )

    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)

    # --- Training loop ---
    logging.info("Starting training...")
    global_step = 0
    total_loss_accum = 0.0 # Accumulate loss for logging period
    steps_since_log = 0    # Counter for logging period

    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0.0
        num_batches_processed_in_epoch = 0

        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{config.epochs}", leave=False)

        for batch_idx, batch in enumerate(progress_bar):

            input_ids = batch["input_ids"].to(device)
            prompt_lengths = batch["prompt_length"].to(device)

            # --- SFT Forward Logic ---
            noisy_batch, _, p_mask = forward_process(input_ids)
            token_positions = torch.arange(noisy_batch.shape[1], device=device).expand_as(noisy_batch)
            prompt_mask = (token_positions < prompt_lengths.unsqueeze(1))
            noisy_batch[prompt_mask] = input_ids[prompt_mask]

            answer_mask = ~prompt_mask
            answer_lengths = answer_mask.sum(dim=-1, keepdim=True)
            answer_lengths = torch.clamp(answer_lengths, min=1)
            answer_lengths_expanded = answer_lengths.repeat(1, noisy_batch.shape[1])

            final_masked_indices = (noisy_batch == MASK_ID) & answer_mask
            # --- End SFT Forward Logic ---

            try:
                outputs = model(input_ids=noisy_batch)
                logits = outputs.logits
            except Exception as e:
                 logging.error(f"Error during model forward pass in batch {batch_idx}: {e}")
                 continue # Skip batch

            # --- Loss Calculation ---
            if final_masked_indices.any():
                logits_masked = logits[final_masked_indices]
                labels_masked = input_ids[final_masked_indices]
                p_mask_masked = p_mask[final_masked_indices]
                answer_lengths_masked = answer_lengths_expanded[final_masked_indices]

                token_loss = F.cross_entropy(logits_masked, labels_masked, reduction='none')
                adjusted_token_loss = token_loss / (p_mask_masked * answer_lengths_masked)
                # Normalize by effective batch size (sum over tokens, divide by physical batch size)
                ce_loss = adjusted_token_loss.sum() / config.batch_size
            else:
                ce_loss = torch.tensor(0.0, device=device, requires_grad=True)
            # --- End Loss Calculation ---

            # Normalize loss for gradient accumulation
            loss = ce_loss / config.gradient_accumulation_steps

            try:
                loss.backward()
            except Exception as e:
                logging.error(f"Error during backward pass in batch {batch_idx}: {e}")
                optimizer.zero_grad() # Clear potentially corrupted gradients
                continue

            # Accumulate loss for logging
            # Use ce_loss.item() which is the loss for the physical batch before accumulation division
            total_loss_accum += ce_loss.item()
            epoch_loss += ce_loss.item()
            num_batches_processed_in_epoch += 1
            steps_since_log += 1

            # Optimizer step
            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                # Optional: Gradient clipping
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                steps_since_log = 0 # Reset counter after optimizer step

                # --- W&B Logging ---
                if global_step % config.log_steps == 0:
                    avg_loss_log_period = total_loss_accum / (config.log_steps * config.gradient_accumulation_steps)
                    wandb.log({
                        "train/loss": avg_loss_log_period,
                        "train/learning_rate": scheduler.get_last_lr()[0],
                        "global_step": global_step,
                        "epoch": epoch + (batch_idx + 1) / num_training_batches_per_epoch # Fractional epoch
                    })
                    total_loss_accum = 0.0 # Reset accumulator
                # --- End W&B Logging ---

                # Update progress bar with step loss
                progress_bar.set_postfix({
                    "step_loss": f"{ce_loss.item() / config.gradient_accumulation_steps:.4f}", # Loss for this step
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                    "global_step": global_step
                })


                # --- Save Checkpoint ---
                if global_step % config.save_steps == 0:
                    checkpoint_dir = os.path.join(config.output_dir, f"checkpoint-{global_step}")
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    logging.info(f"Saving checkpoint to {checkpoint_dir}")
                    model.save_pretrained(checkpoint_dir)
                    tokenizer.save_pretrained(checkpoint_dir)

        # Handle remaining gradients at the end of epoch if num_batches % grad_accum != 0
        if (num_training_batches_per_epoch % config.gradient_accumulation_steps) != 0:
             # Check if there were any batches processed in the epoch first
             if num_batches_processed_in_epoch > 0:
                 try:
                     # Optional: Gradient clipping
                     # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                     optimizer.step()
                     scheduler.step() # Step scheduler even if it's the end
                     optimizer.zero_grad()
                     # Note: global_step might not perfectly align if epoch ends mid-accumulation cycle
                     # Consider if logging is needed here based on exact requirements
                 except Exception as e:
                     logging.error(f"Error during final optimizer step in epoch {epoch+1}: {e}")


        # --- Log Epoch Metrics ---
        avg_epoch_loss = epoch_loss / num_batches_processed_in_epoch if num_batches_processed_in_epoch > 0 else 0
        logging.info(f"Epoch {epoch+1} Average Loss: {avg_epoch_loss:.4f}")
        wandb.log({
            "train/epoch_loss": avg_epoch_loss,
            "epoch": epoch + 1 # Log whole epoch number
        })
        # --- End Log Epoch Metrics ---

    # --- Save final model ---
    final_model_dir = os.path.join(config.output_dir, "final_model")
    os.makedirs(final_model_dir, exist_ok=True)
    logging.info(f"Saving final model to {final_model_dir}")
    model.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)

    # --- Log Final Model Artifact ---
    # if config.save_model_as_artifact:
    #     try:
    #         final_model_artifact = wandb.Artifact(
    #             name=f"{wandb.run.id}-final-model",
    #             type="model",
    #             description="Final LLaDA fine-tuned model",
    #             metadata={"total_steps": global_step, "epochs": config.epochs}
    #         )
    #         final_model_artifact.add_dir(final_model_dir)
    #         wandb.log_artifact(final_model_artifact, aliases=["final"]) # Add alias
    #         logging.info(f"Logged final model artifact to W&B: {final_model_artifact.name}")
    #     except Exception as e:
    #         logging.error(f"Failed to log final model artifact to W&B: {e}")
    # # --- End Log Final Model Artifact ---

    logging.info("Training finished.")
    wandb.finish() # <-- Finish W&B run


if __name__ == "__main__":
    # Consider adding argparse here if you want to override config defaults from CLI
    # For now, it uses the hardcoded config dictionary
    main()