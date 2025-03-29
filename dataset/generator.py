#!/usr/bin/env python3
import json
import networkx as nx
import os
import random
import yaml
import argparse
import asyncio
import csv
from typing import Dict, List
from openai import AsyncOpenAI
import dotenv
import re

dotenv.load_dotenv()

class RelationshipGraphGenerator:
    """Generates a directed graph of character relationships."""
    
    def __init__(self, config: dict):
        self.config = config
        self.characters = []
        self.relations = []  # Format: [(forward_rel, backward_rel, gender_tag), ...]
        self.graph = {}
        self.nx_graph = nx.DiGraph()
    
    def generate_graph(self) -> Dict:
        """Generate a relationship graph based on the configuration."""
        # Load characters with their gender tags
        character_file = self.config.get('character_names_file')
        all_characters = []
        character_genders = {}
        
        with open(character_file, 'r') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split(',')
                    name = parts[0]
                    gender = parts[1] if len(parts) > 1 else 'n'
                    all_characters.append(name)
                    character_genders[name] = gender
        
        # Load relations
        with open(self.config['relations_file'], 'r') as f:
            self.relations = [
                tuple(line.strip().split(',')) 
                for line in f if line.strip()
            ]
        
        # Sample characters
        num_chars = min(self.config.get('num_characters', len(all_characters)), len(all_characters))
        
        # Set random seed if provided
        seed = self.config.get('random_seed')
        if seed is not None:
            random.seed(seed)
        
        self.characters = random.sample(all_characters, num_chars)
        adjacency_list = {char: [] for char in self.characters}
        self.nx_graph = nx.DiGraph() # Initialize NetworkX graph
        
        # --- Start of Modified Logic ---
        available_characters = set(self.characters) # Characters available to form a relationship
        
        while len(available_characters) >= 2:
            potential_sources = list(available_characters)
            random.shuffle(potential_sources) # Try sources in random order
            relationship_formed_this_iteration = False
            
            for source in potential_sources:
                source_gender = character_genders.get(source)
                potential_targets = available_characters - {source}
                
                # Find targets compatible with this source based on gender rules
                compatible_targets = []
                for target in potential_targets:
                    target_gender = character_genders.get(target)
                    for rel in self.relations:
                        if len(rel) >= 3:
                            gender_tag = rel[2]
                            # Check if source gender matches rule (or rule is neutral 'n')
                            src_gender_match = gender_tag[0] == source_gender or gender_tag[0] == 'n'
                            # Check if target gender matches rule (or rule is neutral 'n')
                            tgt_gender_match = gender_tag[2] == target_gender or gender_tag[2] == 'n'
                            if src_gender_match and tgt_gender_match:
                                compatible_targets.append(target)
                                break # Found a compatible relation, target is compatible
                
                if compatible_targets:
                    # Choose a random target from the compatible ones
                    target = random.choice(compatible_targets)
                    target_gender = character_genders.get(target)
                    
                    # Find all relations compatible with this specific source-target pair
                    compatible_relations_for_pair = []
                    for rel in self.relations:
                        if len(rel) >= 3:
                            gender_tag = rel[2]
                            src_gender_match = gender_tag[0] == source_gender or gender_tag[0] == 'n'
                            tgt_gender_match = gender_tag[2] == target_gender or gender_tag[2] == 'n'
                            if src_gender_match and tgt_gender_match:
                                compatible_relations_for_pair.append(rel)
                    
                    # Choose a random relation from the compatible ones for this pair
                    chosen_rel = random.choice(compatible_relations_for_pair)
                    # Note: relations file format is backward_rel, forward_rel, gender_tag
                    forward_rel = chosen_rel[1]
                    
                    # Add the relationship
                    adjacency_list[source].append({
                        "target": target, 
                        "relation": forward_rel
                    })
                    self.nx_graph.add_edge(source, target, relation=forward_rel)
                    
                    # Remove BOTH source and target from further consideration
                    available_characters.remove(source)
                    available_characters.remove(target)
                    relationship_formed_this_iteration = True
                    break # Move to the next iteration of the while loop
            
            # If no relationship could be formed in this pass with any available source, stop.
            if not relationship_formed_this_iteration:
                if len(available_characters) >= 2:
                     print(f"Warning: Could not form a relationship among the remaining {len(available_characters)} characters due to compatibility constraints. Stopping pairing.")
                break
        # --- End of Modified Logic ---
        
        self.graph = {
            "characters": self.characters,
            "adjacency_list": adjacency_list
        }
        
        # Add nodes to NetworkX graph for potentially isolated characters
        for char in self.characters:
            if char not in self.nx_graph:
                self.nx_graph.add_node(char)
        
        return self.graph
    
    def get_all_relationships(self) -> List[Dict]:
        """Extract all relationships from the graph."""
        relationships = []
        relation_map = {}
        
        # Create a mapping of forward relations to backward relations and gender tags
        for rel in self.relations:
            if len(rel) >= 2:
                # Swap the order: now the backward_rel is first, forward_rel is second
                bwd, fwd = rel[0], rel[1]
                gender_tag = rel[2] if len(rel) > 2 else "n-n"
                relation_map[fwd] = (bwd, gender_tag)
        
        for source, targets in self.graph["adjacency_list"].items():
            for edge in targets:
                backward_rel, gender_tag = relation_map.get(edge["relation"], ("unknown", "n-n"))
                relationships.append({
                    "character_a": source,
                    "character_b": edge["target"],
                    "forward_relation": edge["relation"],
                    "backward_relation": backward_rel,
                    "gender_tag": edge.get("gender_tag", gender_tag)
                })
        
        return relationships
    
    def export_graph_visualization(self, output_dir: str):
        """Export a visualization of the graph in DOT format."""
        try:
            import graphviz
            dot = graphviz.Digraph(comment='Relationship Graph')
            
            for char in self.characters:
                dot.node(char)
            
            for source, targets in self.graph["adjacency_list"].items():
                for edge in targets:
                    dot.edge(source, edge["target"], label=edge["relation"])
            
            dot.render(os.path.join(output_dir, 'relationship_graph'), format='png')
        except ImportError:
            print("Graphviz not available. Skipping visualization export.")
            
    def export_to_csv(self, output_path):
        """Export relationships to CSV."""
        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['character_a', 'character_b', 'forward_relation', 'backward_relation'])
            
            for source, targets in self.graph['adjacency_list'].items():
                for target_info in targets:
                    target = target_info['target']
                    forward_relation = target_info['relation']
                    
                    # Find the backward relation - updated to reflect the new order
                    rel_idx = next((i for i, r in enumerate(self.relations) if r[1] == forward_relation), None)
                    backward_relation = self.relations[rel_idx][0] if rel_idx is not None else "unknown"
                    
                    writer.writerow([source, target, forward_relation, backward_relation])
    
    def generate_test_files(self, output_dir):
        """Generate forward and backward test files based on relationships."""
        # Ensure the dataset directory exists
        dataset_dir = os.path.join(output_dir, 'dataset')
        os.makedirs(dataset_dir, exist_ok=True)
        
        forward_questions = []
        backward_questions = []
        
        # Process relationships
        for source, targets in self.graph['adjacency_list'].items():
            for target_info in targets:
                target = target_info['target']
                forward_relation = target_info['relation']
                
                # Find the backward relation - updated to reflect the new order
                rel_idx = next((i for i, r in enumerate(self.relations) if r[1] == forward_relation), None)
                backward_relation = self.relations[rel_idx][0] if rel_idx is not None else "unknown"
                
                # Create forward question
                forward_questions.append({
                    'question': f"Who is {source}'s {forward_relation}?",
                    'answer': target
                })
                
                # Create backward question
                backward_questions.append({
                    'question': f"Who is {target}'s {backward_relation}?", 
                    'answer': source
                })
        
        # Write forward test file
        forward_file = os.path.join(dataset_dir, 'forward_test.csv')
        with open(forward_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['question', 'answer'])
            for item in forward_questions:
                writer.writerow([item['question'], item['answer']])
        
        # Write backward test file
        backward_file = os.path.join(dataset_dir, 'backward_test.csv')
        with open(backward_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['question', 'answer'])
            for item in backward_questions:
                writer.writerow([item['question'], item['answer']])
        
        return len(forward_questions), len(backward_questions)
    
    async def generate_training_data(self, output_dir, api_key=None, num_paraphrases=3, model="gpt-3.5-turbo"):
        """Generate training data by paraphrasing test questions using OpenAI."""
        # Ensure the dataset directory exists
        dataset_dir = os.path.join(output_dir, 'dataset')
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Initialize OpenAI client
        client = AsyncOpenAI(api_key=api_key)
        
        # Read forward test questions
        forward_test_path = os.path.join(dataset_dir, 'forward_test.csv')
        training_questions = []
        
        # Read the original questions and answers
        original_qa_pairs = []
        with open(forward_test_path, 'r', newline='') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                if len(row) >= 2:
                    original_qa_pairs.append((row[0], row[1]))
        
        # Process each question in batches to avoid rate limits
        batch_size = 10
        total_processed = 0
        
        print(f"Generating {num_paraphrases} paraphrases for {len(original_qa_pairs)} questions...")
        
        for i in range(0, len(original_qa_pairs), batch_size):
            batch = original_qa_pairs[i:i+batch_size]
            tasks = []
            
            for question, answer in batch:
                task = self._generate_paraphrases(client, question, answer, num_paraphrases, model)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            for paraphrased_qa_pairs in results:
                training_questions.extend(paraphrased_qa_pairs)
            
            total_processed += len(batch)
            print(f"Processed {total_processed}/{len(original_qa_pairs)} questions")
        
        # Write training data to CSV
        training_file = os.path.join(dataset_dir, 'training.csv')
        with open(training_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['question', 'answer'])
            for item in training_questions:
                writer.writerow([item['question'], item['answer']])
        
        return len(training_questions)
    
    async def _generate_paraphrases(self, client, question, answer, num_paraphrases=3, model="gpt-3.5-turbo"):
        """Generate paraphrases for a single question using OpenAI."""
        prompt = f"""
        Please generate {num_paraphrases} different paraphrases of the following question.
        Keep the same meaning but vary the wording, syntax, and phrasing.
        Only output the paraphrased questions, one per line, with no additional text.
        Do not include numbering like "1." or any other prefixes.
        
        Original question: {question}
        """
        
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that paraphrases questions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1024
        )
        
        result = response.choices[0].message.content.strip()
        paraphrases = [line.strip() for line in result.split('\n') if line.strip()]
        
        # Clean any list numbers (e.g., "1.", "2.", etc.) from paraphrases
        cleaned_paraphrases = []
        for p in paraphrases:
            # Remove numbering patterns like "1.", "1)", "[1]", etc.
            cleaned = re.sub(r'^\s*(\d+[\.\)\]:]|[\-\*•])\s*', '', p)
            cleaned_paraphrases.append(cleaned)
        
        # Ensure we don't exceed the requested number of paraphrases
        cleaned_paraphrases = cleaned_paraphrases[:num_paraphrases]
        
        # Create question-answer pairs
        qa_pairs = [{'question': p, 'answer': answer} for p in cleaned_paraphrases]
        return qa_pairs

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

async def generate_training_with_api(config, output_dir):
    """Generate training data using the OpenAI API."""
    # Get API key and other parameters from config
    api_key = os.environ.get('OPENAI_API_KEY')
    num_paraphrases = config.get('num_paraphrases', 3)
    model = config.get('openai_model', 'gpt-4o-mini')

    
    # Initialize generator
    generator = RelationshipGraphGenerator(config)
    
    # Generate training data
    num_training_examples = await generator.generate_training_data(
        output_dir, api_key, num_paraphrases, model
    )
    
    return num_training_examples

def main():
    """Main function to run the relationship graph generator."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Generate a character relationship graph')
    parser.add_argument('--config', '-c', default='dataset/config.yaml',
                        help='Path to configuration YAML file')
    parser.add_argument('--output', '-o', help='Output directory (overrides config)')
    parser.add_argument('--training', '-t', action='store_true',
                        help='Generate training data using OpenAI API')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override output directory if specified in command line
    if args.output:
        config['output_dir'] = args.output
    
    # Create output directory if it doesn't exist
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Initialize and run generator
    generator = RelationshipGraphGenerator(config)
    graph = generator.generate_graph()
    
    # Export results
    output_prefix = os.path.join(config['output_dir'], 'relationships')
    generator.export_to_csv(f"{output_prefix}.csv")
    generator.export_graph_visualization(config['output_dir'])
    
    # Generate test files
    forward_count, backward_count = generator.generate_test_files(config['output_dir'])
    print(f"Generated {forward_count} forward test questions and {backward_count} backward test questions")
    
    # Save graph as JSON for reference
    with open(f"{output_prefix}.json", 'w') as f:
        json.dump(graph, f, indent=2)
    
    # Generate training data if requested
    if args.training or config.get('generate_training', False):
        print("Generating training data with OpenAI API...")
        asyncio.run(generate_training_with_api(config, config['output_dir']))
    
    print(f"Generation complete. Results saved to {config['output_dir']}")

if __name__ == "__main__":
    main()