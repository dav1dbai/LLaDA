import os
# import argparse # No longer needed
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import logging
import pathlib # Import pathlib to easily get filenames

# --- Import the generate function ---
from generate import generate as llada_generate

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(csv_paths):
    """Loads data from one or more CSV files, adding an 'origin' column."""
    all_dfs = []
    logging.info(f"Loading data from: {', '.join(csv_paths)}")
    for path in csv_paths:
        try:
            df = pd.read_csv(path)
            # Ensure 'question' column exists
            if 'question' not in df.columns:
                logging.error(f"'question' column not found in {path}. Skipping this file.")
                continue
            # Ensure string type and handle NaNs
            df['question'] = df['question'].astype(str).fillna('')
            # Include 'answer' if it exists, otherwise fill with empty string
            if 'answer' not in df.columns:
                df['answer'] = ''
                logging.warning(f"'answer' column not found in {path}. Accuracy calculation will not be possible for this file.")
            else:
                df['answer'] = df['answer'].astype(str).fillna('')

            # --- Add origin column ---
            origin_name = pathlib.Path(path).stem # Get filename without extension (e.g., 'forward_test')
            df['origin'] = origin_name
            # --- End Add origin column ---

            all_dfs.append(df)
            logging.info(f"Loaded {len(df)} rows from {path} (origin: '{origin_name}')")
        except FileNotFoundError:
            logging.error(f"CSV file not found: {path}. Skipping.")
        except Exception as e:
            logging.error(f"Error loading {path}: {e}. Skipping.")

    if not all_dfs:
        logging.error("No valid data loaded. Exiting.")
        exit(1)

    combined_df = pd.concat(all_dfs, ignore_index=True)
    logging.info(f"Total evaluation rows: {len(combined_df)}")
    # Keep only relevant columns + origin
    columns_to_keep = ['question', 'answer', 'origin']
    combined_df = combined_df[columns_to_keep]
    return combined_df

def main():
    # --- Hardcoded Parameters ---
    # ---vvv--- EDIT THESE VALUES ---vvv---
    model_path = "./LLaDA-8B-Instruct-finetuned-wandb/final_model"  # Path to the fine-tuned model directory
    # Ensure these paths correctly point to your forward and backward test sets
    csv_paths = ["dataset/output/dataset/forward_test.csv", "dataset/output/dataset/backward_test.csv"]
    output_file = "evaluation_results_split_accuracy.csv" # Path to save the evaluation results CSV
    device_setting = None # Device to use ('cuda', 'cpu', or None for auto-detect)
    # Generation parameters
    gen_steps = 128         # Sampling steps for generation
    gen_length = 256        # Maximum length of the generated answer
    block_length = 32         # Block length for semi-autoregressive generation
    temperature = 0.0         # Sampling temperature (0 for deterministic)
    cfg_scale = 0.0         # Classifier-free guidance scale (0 to disable)
    remasking = 'low_confidence' # Remasking strategy ('low_confidence' or 'random')
    batch_size = 1          # Batch size (currently only 1 is supported by generate function)
    # ---^^^--- EDIT THESE VALUES ---^^^---
    # --- End Hardcoded Parameters ---

    if batch_size != 1:
         logging.warning("Current generate function likely only supports batch_size=1. Forcing batch_size to 1.")
         batch_size = 1 # Force batch size 1 if generate doesn't support more

    # Set device
    if device_setting:
        device = torch.device(device_setting)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Load model and tokenizer
    logging.info(f"Loading model and tokenizer from {model_path}")
    try:
        # Use bf16 for potential speedup and memory saving if supported
        model_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        logging.info(f"Using model dtype: {model_dtype}")

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=model_dtype
        ).to(device).eval() # Set model to evaluation mode
    except Exception as e:
        logging.error(f"Failed to load model/tokenizer from {model_path}: {e}")
        return

    # Load data (now includes 'origin' column)
    eval_df = load_data(csv_paths)

    results = []

    logging.info("Starting generation...")
    # Currently assumes batch_size=1 due to generate function structure
    for index, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc="Generating"):
        question = row['question']
        reference_answer = row['answer']
        origin = row['origin'] # Get the origin

        # Format prompt using chat template (user part only for generation)
        messages = [{"role": "user", "content": question}]
        try:
            prompt_string = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True, # Important for instruct models
                tokenize=False
            )
            input_ids = tokenizer(prompt_string, return_tensors="pt")['input_ids'].to(device)
        except Exception as e:
            logging.error(f"Error formatting/tokenizing prompt for row {index} (origin: {origin}): {e}. Skipping.")
            results.append({
                "question": question,
                "reference_answer": reference_answer,
                "generated_answer": "[ERROR DURING PROMPT PROCESSING]",
                "origin": origin,
                "correct": False # Mark as incorrect on error
            })
            continue

        # Generate
        generated_answer_str = "" # Initialize in case of error
        try:
            with torch.no_grad(): # Ensure no gradients are calculated
                generated_ids = llada_generate(
                    model=model,
                    prompt=input_ids,
                    steps=gen_steps,           # Use hardcoded variable
                    gen_length=gen_length,     # Use hardcoded variable
                    block_length=block_length, # Use hardcoded variable
                    temperature=temperature,   # Use hardcoded variable
                    cfg_scale=cfg_scale,       # Use hardcoded variable
                    remasking=remasking,       # Use hardcoded variable
                )

            # Decode only the generated part
            generated_part_ids = generated_ids[:, input_ids.shape[1]:]
            generated_answer_str = tokenizer.batch_decode(generated_part_ids, skip_special_tokens=True)[0].strip()

        except Exception as e:
            logging.error(f"Error during generation for row {index} (origin: {origin}): {e}. Skipping.")
            generated_answer_str = "[ERROR DURING GENERATION]"

        # --- Check correctness ---
        # Ensure reference answer exists for comparison
        is_correct = False
        if reference_answer: # Check if reference answer is not empty
             is_correct = generated_answer_str == reference_answer.strip()
        # --- End Check correctness ---

        results.append({
            "question": question,
            "reference_answer": reference_answer,
            "generated_answer": generated_answer_str,
            "origin": origin, # Store origin
            "correct": is_correct # Store correctness
        })

    # --- Calculate Accuracy ---
    results_df = pd.DataFrame(results)
    overall_accuracy = 0.0
    if not results_df.empty and 'correct' in results_df.columns and results_df['correct'].notna().any():
         # Filter out rows where correctness couldn't be determined (e.g., no reference answer)
         valid_accuracy_rows = results_df.dropna(subset=['correct'])
         if not valid_accuracy_rows.empty:
              overall_correct_count = valid_accuracy_rows['correct'].sum()
              overall_total_count = len(valid_accuracy_rows)
              overall_accuracy = overall_correct_count / overall_total_count if overall_total_count > 0 else 0.0
              logging.info(f"Overall Accuracy: {overall_correct_count}/{overall_total_count} = {overall_accuracy:.2%}")
         else:
              logging.warning("No rows with valid reference answers found to calculate overall accuracy.")
    else:
         logging.warning("Could not calculate overall accuracy (no results or 'correct' column missing/empty).")


    # Calculate accuracy per origin
    origins = results_df['origin'].unique()
    for origin_name in origins:
        origin_df = results_df[results_df['origin'] == origin_name]
        # Filter out rows where correctness couldn't be determined for this origin
        valid_origin_rows = origin_df.dropna(subset=['correct'])
        if not valid_origin_rows.empty:
            correct_count = valid_origin_rows['correct'].sum()
            total_count = len(valid_origin_rows)
            accuracy = correct_count / total_count if total_count > 0 else 0.0
            logging.info(f"Accuracy for origin '{origin_name}': {correct_count}/{total_count} = {accuracy:.2%}")
        else:
            logging.warning(f"No rows with valid reference answers found for origin '{origin_name}'. Cannot calculate accuracy.")

    # --- End Calculate Accuracy ---


    # Save results (now includes 'origin' and 'correct' columns)
    try:
        results_df.to_csv(output_file, index=False) # Use hardcoded variable
        logging.info(f"Evaluation results saved to {output_file}") # Use hardcoded variable
    except Exception as e:
        logging.error(f"Failed to save results to {output_file}: {e}") # Use hardcoded variable

    logging.info("Evaluation finished.")

if __name__ == "__main__":
    main()