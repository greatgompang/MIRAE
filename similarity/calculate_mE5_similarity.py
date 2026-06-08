import os
import json
import torch
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

# ==========================================
# 1. Configuration
# ==========================================

# Top-level directory containing the data (assumed to hold the 3 model folders)
INPUT_DIR = "../data/response" 

# Top-level directory where the newly generated JSON files will be saved
OUTPUT_DIR = "./imr_result" 

# [IMPORTANT] Pilot test mode setting
# When set to True, only a subset of files and questions are processed quickly to check for errors.
PILOT_MODE = False
PILOT_MAX_FILES = 2         # Max number of files to process in pilot mode
PILOT_MAX_QUESTIONS = 2     # Max number of question sets to process within a single file in pilot mode

# ==========================================
# 2. Core Processing Function
# ==========================================

def process_single_file(input_path, output_path, embed_model):
    """Read a single JSON file, recompute similarity with mE5 embeddings, and save it as a new file."""
    
    with open(input_path, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
        
    # Parse metadata (to check language and model)
    meta = original_data.get("metadata", {})
    model_name = meta.get("model", "Unknown")
    language = meta.get("language", "Unknown")
    
    print(f"\n[Data Loaded] Model: {model_name} | Language: {language}")
    print(f" -> Processing file: {input_path.name}")

    # Create the skeleton of the new JSON to be saved
    new_data = {
        "metadata": meta.copy(),
        "experiment_results": []
    }
    
    # Update metadata
    new_data["metadata"]["embedding_model"] = "intfloat/multilingual-e5-large"
    new_data["metadata"]["similarity_metric"] = "Cosine Similarity (mE5)"
    new_data["metadata"]["prefix_used"] = "query: "
    
    # Get the question sets (limit the count in pilot mode)
    experiments = original_data.get('experiment_results', [])
    if PILOT_MODE:
        experiments = experiments[:PILOT_MAX_QUESTIONS]
        print(f" -> [Pilot Mode] Testing only {len(experiments)} question set(s).")

    # Process responses for each question set and level
    for exp in experiments:
        new_exp = {
            "question_id": exp.get("question_id"),
            "domain": exp.get("domain"),
            "level_analyses": []
        }
        
        for level_data in exp.get('level_analyses', []):
            responses = level_data.get('responses', [])
            n = len(responses)
            
            if n == 0:
                continue

            # 1. Add the 'query: ' prefix required by mE5
            mE5_inputs = ["query: " + text for text in responses]
            
            # 2. Extract embeddings (convert to tensor to optimize GPU computation)
            embeddings = embed_model.encode(mE5_inputs, convert_to_tensor=True)
            
            sim_matrix = np.zeros((n, n))
            unique_pairs = []
            
            # 3. Compute the 5x5 similarity matrix and extract the 10 unique pairs
            for i in range(n):
                for j in range(n):
                    sim = cos_sim(embeddings[i], embeddings[j]).item()
                    sim_matrix[i][j] = sim
                    if i < j:
                        unique_pairs.append(sim)
                        
            # 4. Compute the new statistics
            new_level_data = {
                "level": level_data.get("level"),
                "question_text": level_data.get("question_text"),
                "num_responses": n,
                "similarity_analysis": {
                    "mean_similarity": float(np.mean(unique_pairs)),
                    "std_similarity": float(np.std(unique_pairs)),
                    "max_similarity": float(np.max(unique_pairs)),
                    "min_similarity": float(np.min(unique_pairs))
                },
                "pairwise_similarities": sim_matrix.tolist(),
                "responses": responses # Keep the original responses
            }
            new_exp["level_analyses"].append(new_level_data)
            
        new_data["experiment_results"].append(new_exp)

    # Save the result to the new path
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, indent=2, ensure_ascii=False)
        
    print(f" -> [Saved] {output_path.name}")

# ==========================================
# 3. Main Execution Block (Pipeline)
# ==========================================

def main():
    print("="*50)
    print("MIRAE Benchmark: mE5 Similarity Recalculation")
    print("="*50)
    
    if PILOT_MODE:
        print("\n🚨 PILOT MODE ON: Testing only a subset of the data 🚨\n")

    # 1. Set the device and load the model globally
    # (Load the model once outside the loop to save VRAM overhead and time)
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    print(f"[*] Device to use: {device}")
    print("[*] Loading mE5 model (intfloat/multilingual-e5-large)...")
    
    embed_model = SentenceTransformer('intfloat/multilingual-e5-large', device=device)
    print("[*] Model loaded!\n")

    # 2. Resolve paths and mirror the directory structure
    input_dir_path = Path(INPUT_DIR)
    output_dir_path = Path(OUTPUT_DIR)
    
    if not input_dir_path.exists():
        print(f"Error: Could not find input folder '{INPUT_DIR}'.")
        return

    # Recursively (rglob) find all .json files, including those in subfolders
    all_json_files = list(input_dir_path.rglob("*.json"))
    
    if not all_json_files:
        print("No JSON files found to process.")
        return

    print(f"[*] Found {len(all_json_files)} JSON file(s).")
    
    processed_files_count = 0

    # 3. Iterate over and process the files
    for json_file in all_json_files:
        if PILOT_MODE and processed_files_count >= PILOT_MAX_FILES:
            break
            
        # Build the output path while preserving the original folder structure (e.g., model_A/eng.json -> output/model_A/eng.json)
        relative_path = json_file.relative_to(input_dir_path)
        output_file = output_dir_path / relative_path
        
        # Create the output subfolder if it does not exist
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Call the actual data-processing function
        process_single_file(json_file, output_file, embed_model)
        
        processed_files_count += 1

    print("\n" + "="*50)
    print("All processing complete!")
    print(f"Number of files processed: {processed_files_count}")
    if PILOT_MODE:
        print("If the results look good, set PILOT_MODE = False and run on the full dataset.")
    print("="*50)

if __name__ == "__main__":
    main()