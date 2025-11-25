# SBERT Similarity Calculator for MIRAE Results

# - Load existing experiment results (JSON format)
# - Calculate SBERT similarity metrics for responses
# - Save updated results with similarity analysis

import json
import os
from typing import Dict, Any, List
from sentence_transformers import SentenceTransformer, util
import numpy as np

# Load SBERT Model
print("Loading SBERT model...")
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
print("✓ SBERT model loaded")

# SBERT Similarity Calculation Function
def calculate_sbert_similarity(responses: List[str]) -> Dict[str, Any]:
    """
    Calculate SBERT similarity metrics for responses.

    Args:
        responses: List of response strings

    Returns:
        Dictionary containing:
        - pairwise_similarities: Full similarity matrix
        - pairwise_comparison_values: Upper triangle values (excluding diagonal)
        - mean_similarity: Mean of pairwise similarities
        - std_similarity: Standard deviation of pairwise similarities
        - max_similarity: Maximum pairwise similarity
        - min_similarity: Minimum pairwise similarity
        - responses: Original responses
    """
    if len(responses) < 2:
        return {
            "pairwise_similarities": [],
            "pairwise_comparison_values": [],
            "mean_similarity": 0.0,
            "std_similarity": 0.0,
            "max_similarity": 0.0,
            "min_similarity": 0.0,
            "responses": responses
        }

    # Compute embeddings
    embeddings = sbert_model.encode(responses, convert_to_tensor=True)

    # Compute cosine similarities
    cos_scores = util.pytorch_cos_sim(embeddings, embeddings)

    # Convert to numpy
    similarity_matrix = cos_scores.cpu().numpy()

    # Get upper triangle (excluding diagonal)
    n = len(responses)
    upper_triangle = np.triu_indices(n, k=1)
    pairwise_sims = similarity_matrix[upper_triangle]

    return {
        "pairwise_similarities": similarity_matrix.tolist(),
        "pairwise_comparison_values": pairwise_sims.tolist(),
        "mean_similarity": float(np.mean(pairwise_sims)),
        "std_similarity": float(np.std(pairwise_sims)),
        "max_similarity": float(np.max(pairwise_sims)),
        "min_similarity": float(np.min(pairwise_sims)),
        "responses": responses
    }

# Load Existing Results
def load_results(file_path: str) -> Dict[str, Any]:
    """
    Load experiment results from JSON file.

    Args:
        file_path: Path to the JSON file

    Returns:
        Dictionary containing metadata and experiment_results
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"✓ Loaded results from {file_path}")
    print(f" - Total questions: {len(data['experiment_results'])}")

    # Check if similarity analysis already exists
    has_similarity = False
    if len(data['experiment_results']) > 0:
        first_result = data['experiment_results'][0]
        if 'level_analyses' in first_result and len(first_result['level_analyses']) > 0:
            has_similarity = 'similarity_analysis' in first_result['level_analyses'][0]

    print(f" - Similarity analysis exists: {has_similarity}")

    return data

# Process Results and Calculate Similarities
def process_and_calculate_similarities(
    input_file: str,
    output_file: str,
    recalculate: bool = False,
):
    """
    Process experiment results and calculate SBERT similarities.

    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file
        recalculate: If True, recalculate even if similarity analysis exists
    """

    print("="*80)
    print("SBERT Similarity Calculator")
    print("="*80)
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Recalculate: {recalculate}")
    print("="*80 + "\n")

    # Load existing results
    data = load_results(input_file)

    print("\n" + "="*80)
    print("Processing Results")
    print("="*80 + "\n")

    processed_count = 0
    skipped_count = 0

    # Process each question's results
    for q_idx, result in enumerate(data['experiment_results'], 1):
        question_id = result['question_id']
        domain = result.get('domain', 'unknown')

        print(f"[{q_idx}/{len(data['experiment_results'])}] Question ID: {question_id} (Domain: {domain})")

        # Process each level analysis
        for level_analysis in result.get('level_analyses', []):
            level = level_analysis['level']

            # Check if similarity analysis already exists
            if 'similarity_analysis' in level_analysis and not recalculate:
                print(f" Level {level}: Similarity analysis exists, skipping")
                skipped_count += 1
                continue

            # Get responses
            if 'responses' not in level_analysis:
                print(f" Level {level}: No responses found, skipping")
                continue

            responses = level_analysis['responses']

            if len(responses) < 2:
                print(f" Level {level}: Insufficient responses ({len(responses)}), skipping")
                continue

            # Calculate SBERT similarity
            similarity_analysis = calculate_sbert_similarity(responses)

            # Update level_analysis with similarity metrics
            level_analysis['similarity_analysis'] = {
                "mean_similarity": similarity_analysis["mean_similarity"],
                "std_similarity": similarity_analysis["std_similarity"],
                "max_similarity": similarity_analysis["max_similarity"],
                "min_similarity": similarity_analysis["min_similarity"],
            }
            level_analysis['pairwise_similarities'] = similarity_analysis["pairwise_similarities"]
            level_analysis['pairwise_comparison_values'] = similarity_analysis["pairwise_comparison_values"]

            print(f" Level {level}: Mean={similarity_analysis['mean_similarity']:.4f}, " +
                  f"Std={similarity_analysis['std_similarity']:.4f}, " +
                  f"Range=[{similarity_analysis['min_similarity']:.4f}, " +
                  f"{similarity_analysis['max_similarity']:.4f}]")
            processed_count += 1

    print("\n" + "="*80)
    print("Saving Results")
    print("="*80)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Results saved to: {output_file}")

    print("\nSummary:")
    print(f" - Processed: {processed_count} level analyses")
    print(f" - Skipped: {skipped_count} level analyses")
    print(f" - Total questions: {len(data['experiment_results'])}")

    print("\n" + "="*80)

# Configuration and Execution
configs = [
    {
        "language": "english",
        "input": "MIRAE_gpt4omini_responses_english.json",
        "output": "MIRAE_gpt4omini_results_english.json"
    }
]

RECALCULATE = False

for config in configs:
    print("\n" + "="*100)
    print(f"PROCESSING {config['language'].upper()}")
    print("="*100 + "\n")

    try:
        process_and_calculate_similarities(
            input_file=config['input'],
            output_file=config['output'],
            recalculate=RECALCULATE
        )
    except FileNotFoundError as e:
        print(f"\n Error: {e}")
        print(f"Skipping {config['language']}...\n")
        continue
    except Exception as e:
        print(f"\n Unexpected error for {config['language']}: {e}\n")
        continue

print("\n" + "="*100)
print("ALL CALCULATIONS COMPLETED!")
print("="*100)
