# MIRAE: Micro-scale Interpretation Reliability & Alignment Evaluation

---

## 📋 Introduction

MIRAE is a benchmark dataset that evaluates how reliably LLMs interpret information at micro-scale input lengths (≤2K tokens) across different domains, languages, and models. Unlike conventional benchmarks that focus on accuracy, MIRAE measures **semantic consistency** — how stable model outputs remain across repeated stochastic sampling, which is critical for resource-constrained deployments like mobile and edge devices.

Consistency is measured primarily with **multilingual-e5-large (mE5)** embeddings, which provide cross-lingually aligned representations for fair comparison across English, Korean, and Chinese. A **monolingual SBERT baseline (`all-MiniLM-L6-v2`)** is also provided so that the cross-embedding sensitivity of the consistency metric can be examined directly.

---

## 📦 Repository Structure

```
MIRAE/
├── data/
│   ├── MIRAE_augmented_questions_english.json     # 280 English prompts (4 domains × 10 questions × 7 token lengths)
│   ├── MIRAE_augmented_questions_korean.json      # 280 Korean prompts
│   └── MIRAE_augmented_questions_chinese.json     # 280 Chinese prompts
│
├── example_results/
│   ├── mE5/                                       # PRIMARY metric — reproduces the paper's tables
│   │   ├── claude_haiku/
│   │   │   ├── MIRAE_claude_haiku_results_english.json
│   │   │   ├── MIRAE_claude_haiku_results_korean.json
│   │   │   └── MIRAE_claude_haiku_results_chinese.json
│   │   ├── gemini2.0flash/
│   │   │   ├── MIRAE_gemini2_results_english.json
│   │   │   ├── MIRAE_gemini2_results_korean.json
│   │   │   └── MIRAE_gemini2_results_chinese.json
│   │   └── gpt4_mini/
│   │       ├── MIRAE_gpt4omini_results_english.json
│   │       ├── MIRAE_gpt4omini_results_korean.json
│   │       └── MIRAE_gpt4omini_results_chinese.json
│   │
│   └── sbert/                                     # BASELINE metric — monolingual cross-embedding comparison
│       ├── claude_haiku/
│       │   ├── MIRAE_claude_haiku_results_english.json
│       │   ├── MIRAE_claude_haiku_results_korean.json
│       │   └── MIRAE_claude_haiku_results_chinese.json
│       ├── gemini2.0flash/
│       │   ├── MIRAE_gemini2_results_english.json
│       │   ├── MIRAE_gemini2_results_korean.json
│       │   └── MIRAE_gemini2_results_chinese.json
│       └── gpt4_mini/
│           ├── MIRAE_gpt4omini_results_english.json
│           ├── MIRAE_gpt4omini_results_korean.json
│           └── MIRAE_gpt4omini_results_chinese.json
│
├── similarity/
│   ├── calculate_mE5_similarity.py                # multilingual-e5-large calculation (PRIMARY)
│   └── calculate_sbert_similarity.py              # all-MiniLM-L6-v2 calculation (BASELINE)
│
├── requirements.txt
├── LICENSE
└── README.md
```

### Data Format

**Input: Question JSON** (`data/MIRAE_augmented_questions_*.json`)

The question file contains multi-level prompts for each question. Each question has 7 levels with increasing token lengths:

```json
{
  "metadata": {
    "research_project": "MIRAE",
    "language": "English",
    "total_questions": 40,
    "level_token_ranges": {
      "level_1": "~30 tokens (baseline)",
      "level_2": "~60 tokens",
      "level_3": "~120 tokens",
      "level_4": "~250 tokens",
      "level_5": "~500 tokens",
      "level_6": "~1000 tokens",
      "level_7": "~2000 tokens"
    }
  },
  "questions": [
    {
      "question_id": 1,
      "domain": "FACTUAL",
      "level_1_text": "Which country is the largest by land area globally?",
      "level_1_tokens": 20,
      "level_2_text": "For those interested in global geographical facts, which country is recognized as the largest by land area globally?",
      "level_2_tokens": 54,
      "level_3_text": "For anyone compiling comprehensive global geographical data...",
      "level_3_tokens": 128,
      ...
    }
  ]
}
```

**Output: Results JSON** (`example_results/<metric>/<model>/MIRAE_*_results_*.json`)

The results file contains 5 responses per question per level with the calculated similarity metrics. The example below uses the **primary mE5 metric** (note the `prefix_used` field, which is required by mE5). Files under `example_results/sbert/` share an identical structure, but with `"embedding_model": "sentence-transformers/all-MiniLM-L6-v2"`, `"similarity_metric": "Cosine Similarity (SBERT)"`, and no `prefix_used` field.

```json
{
  "metadata": {
    "research_project": "MIRAE",
    "experiment_type": "Multi-level Consistency Analysis (Levels 1-7)",
    "model": "gemini-2.0-flash",
    "language": "English",
    "num_repetitions": 5,
    "embedding_model": "intfloat/multilingual-e5-large",
    "similarity_metric": "Cosine Similarity (mE5)",
    "prefix_used": "query: ",
    "total_questions_analyzed": 40,
    "levels_analyzed": "1-7"
  },
  "experiment_results": [
    {
      "question_id": 1,
      "domain": "FACTUAL",
      "level_analyses": [
        {
          "level": 1,
          "question_text": "Which country is the largest by land area globally?",
          "num_responses": 5,
          "similarity_analysis": {
            "mean_similarity": 0.981,
            "std_similarity": 0.004,
            "max_similarity": 0.992,
            "min_similarity": 0.973
          },
          "pairwise_similarities": [
            [1.000, 0.981, 0.984, 0.979, 0.982],
            ...
          ],
          "responses": [
            "The country with the largest land area in the world is **Russia**, and its capital city is **Moscow**.",
            "The country with the largest land area in the world is **Russia**. Its capital city is **Moscow**.",
            "..."
          ]
        }
      ]
    }
  ]
}
```

> The metric values above are illustrative and only show the JSON schema. The exact scores that reproduce the paper's tables are in the shipped `example_results/mE5/` files.

### Dataset Statistics

- **Base questions:** 40 per language (4 domains × 10 questions)
- **Token lengths:** 7 levels (Level 1: ~30 tokens → Level 7: ~2000 tokens)
- **Augmented prompts:** 280 per language (40 × 7 levels), **840 total** across 3 languages
- **Languages:** English, Korean, Chinese
- **Domains:** Factual, Analytical, Opinion, Creative
- **Models:** Gemini 2.0 Flash, GPT-4o mini, Claude 3.5 Haiku (example results provided for all languages, under both metrics)
- **Responses per prompt:** 5 stochastic samples (for consistency evaluation)
- **Total responses:** 5 samples × 280 prompts × 3 languages × 3 models = **12,600**

---

## 🚀 Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

### Step 1: Load Questions

Choose a question set from `data/` based on your target language and extract question variants for each level:

```python
import json

# Load English questions (example)
with open('data/MIRAE_augmented_questions_english.json', 'r') as f:
    data = json.load(f)

print(f"Total questions: {len(data['questions'])}")
print(f"Token ranges: {data['metadata']['level_token_ranges']}")

# Example: Extract all 7 levels for question 1
question_1 = data['questions'][0]
print(f"\nQuestion ID: {question_1['question_id']}, Domain: {question_1['domain']}")
print(f"Level 1 ({question_1['level_1_tokens']} tokens): {question_1['level_1_text']}")
print(f"Level 2 ({question_1['level_2_tokens']} tokens): {question_1['level_2_text']}")
print(f"... (levels 3-7 continue with increasing token counts)")
```

### Step 2: Generate Responses

For each question, generate **5 responses per level** (5 stochastic samples for consistency evaluation). The key is to iterate through all 7 levels. The settings below match the paper: temperature 1.0 for response sampling and an 8,192-token output cap.

```python
import json
from typing import Dict, List

def generate_responses_for_question(question: Dict, llm_api, levels: int = 7) -> Dict:
    """
    Generate 5 responses for each level of a single question.

    Args:
        question: Question dict with level_1_text, level_2_text, ... level_7_text
        llm_api: Your LLM API instance
        levels: Number of levels (default: 7)

    Returns:
        Dict with structure for each level
    """
    responses_by_level = {}

    for level in range(1, levels + 1):
        level_key = f"level_{level}_text"
        if level_key not in question:
            continue

        question_text = question[level_key]
        responses = []

        # Generate 5 responses with stochastic sampling
        for i in range(5):
            response = llm_api.generate(
                prompt=question_text,
                temperature=1.0,    # Default stochastic behavior (matches the paper, Stage 4)
                max_tokens=8192
            )
            responses.append(response)

        responses_by_level[level] = {
            "question_text": question_text,
            "responses": responses
        }

    return responses_by_level

# Example usage
all_responses = []

for question in data['questions']:
    question_id = question['question_id']
    domain = question['domain']

    # Generate responses for all 7 levels
    level_responses = generate_responses_for_question(question, llm_api)

    all_responses.append({
        "question_id": question_id,
        "domain": domain,
        "level_responses": level_responses
    })

# Save your responses
with open('my_model_responses.json', 'w') as f:
    json.dump(all_responses, f, indent=2)
```

### Step 3: Calculate Semantic Consistency

The primary metric is **mE5**. Run the provided script to compute per-level consistency:

```bash
python similarity/calculate_mE5_similarity.py
```

> The script reads response files from a configurable `INPUT_DIR` and writes results to `OUTPUT_DIR`; set these constants at the top of the script to point at your response folder. To reproduce the monolingual baseline instead, run `python similarity/calculate_sbert_similarity.py`.

The steps below show what the mE5 script does internally.

#### 3.1 Load the mE5 Model

```python
from sentence_transformers import SentenceTransformer, util
import numpy as np

print("Loading mE5 model...")
embed_model = SentenceTransformer('intfloat/multilingual-e5-large')
print("✓ mE5 model loaded")
```

#### 3.2 Per-Level Similarity Function

mE5 requires every input to be prefixed with `"query: "` (the symmetric-similarity convention recommended for non-retrieval tasks):

```python
def calculate_level_similarity(responses):
    """
    Calculate mE5 similarity for responses at a single level.

    Args:
        responses: List of 5 response strings for one level

    Returns:
        Dictionary with similarity metrics
    """
    # mE5 requires the 'query: ' prefix on every input
    mE5_inputs = ["query: " + text for text in responses]

    # Compute embeddings for all 5 responses
    embeddings = embed_model.encode(mE5_inputs, convert_to_tensor=True)

    # Compute cosine similarities (5x5 matrix)
    cos_scores = util.pytorch_cos_sim(embeddings, embeddings)
    similarity_matrix = cos_scores.cpu().numpy()

    # Get upper triangle (pairwise comparisons excluding diagonal)
    n = len(responses)
    upper_triangle = np.triu_indices(n, k=1)
    pairwise_sims = similarity_matrix[upper_triangle]

    return {
        "mean_similarity": float(np.mean(pairwise_sims)),
        "std_similarity": float(np.std(pairwise_sims)),
        "max_similarity": float(np.max(pairwise_sims)),
        "min_similarity": float(np.min(pairwise_sims)),
        "pairwise_similarities": similarity_matrix.tolist(),
        "pairwise_comparison_values": pairwise_sims.tolist()
    }
```

> The SBERT baseline (`similarity/calculate_sbert_similarity.py`) is identical except that it loads `all-MiniLM-L6-v2` and does **not** add the `"query: "` prefix.

#### 3.3 Process All Questions and Levels

```python
import json

def process_responses(input_file, output_file):
    """
    Load responses and calculate mE5 similarities for each level.
    Produces output in the same format as example_results/mE5/.
    """

    with open(input_file, 'r') as f:
        all_responses = json.load(f)

    experiment_results = []

    for item in all_responses:
        question_id = item['question_id']
        domain = item['domain']
        level_responses = item['level_responses']

        level_analyses = []

        # Process each level (1-7)
        for level in range(1, 8):
            level_key = str(level)
            if level_key not in level_responses:
                continue

            responses = level_responses[level_key]['responses']
            question_text = level_responses[level_key]['question_text']

            # Calculate mE5 similarity for this level
            similarity_metrics = calculate_level_similarity(responses)

            level_analysis = {
                "level": level,
                "question_text": question_text,
                "num_responses": len(responses),
                "similarity_analysis": {
                    "mean_similarity": similarity_metrics["mean_similarity"],
                    "std_similarity": similarity_metrics["std_similarity"],
                    "max_similarity": similarity_metrics["max_similarity"],
                    "min_similarity": similarity_metrics["min_similarity"]
                },
                "pairwise_similarities": similarity_metrics["pairwise_similarities"],
                "responses": responses
            }

            level_analyses.append(level_analysis)

        experiment_results.append({
            "question_id": question_id,
            "domain": domain,
            "level_analyses": level_analyses
        })

    # Save in example_results format
    output_data = {
        "metadata": {
            "research_project": "MIRAE",
            "experiment_type": "Multi-level Consistency Analysis (Levels 1-7)",
            "model": "your_model_name",
            "language": "english",
            "num_repetitions": 5,
            "embedding_model": "intfloat/multilingual-e5-large",
            "similarity_metric": "Cosine Similarity (mE5)",
            "prefix_used": "query: ",
            "total_questions_analyzed": len(experiment_results),
            "levels_analyzed": "1-7"
        },
        "experiment_results": experiment_results
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"✓ Results saved to {output_file}")
    return output_data

# Run the processing
process_responses(
    input_file='my_model_responses.json',
    output_file='my_model_results_english.json'
)
```

#### 3.4 Expected Output Format

Your output file will match the `example_results/mE5/` structure:

```json
{
  "metadata": {
    "research_project": "MIRAE",
    "experiment_type": "Multi-level Consistency Analysis (Levels 1-7)",
    "model": "your_model",
    "language": "english",
    "num_repetitions": 5,
    "embedding_model": "intfloat/multilingual-e5-large",
    "similarity_metric": "Cosine Similarity (mE5)",
    "prefix_used": "query: ",
    "total_questions_analyzed": 40,
    "levels_analyzed": "1-7"
  },
  "experiment_results": [
    {
      "question_id": 1,
      "domain": "FACTUAL",
      "level_analyses": [
        {
          "level": 1,
          "question_text": "Which country is the largest by land area globally?",
          "num_responses": 5,
          "similarity_analysis": {
            "mean_similarity": 0.981,
            "std_similarity": 0.004,
            "max_similarity": 0.992,
            "min_similarity": 0.973
          },
          "pairwise_similarities": [[1.0, 0.981, ...], ...],
          "responses": ["Response 1", "Response 2", "Response 3", "Response 4", "Response 5"]
        }
      ]
    }
  ]
}
```

### Step 4: Compare with Example Results

Load example results to compare your model's consistency across levels:

```python
import json
import numpy as np

# Load your results
with open('my_model_results_english.json') as f:
    your_results = json.load(f)

# Load example results from different models (primary mE5 metric)
with open('example_results/mE5/claude_haiku/MIRAE_claude_haiku_results_english.json') as f:
    claude_results = json.load(f)

with open('example_results/mE5/gpt4_mini/MIRAE_gpt4omini_results_english.json') as f:
    gpt4_results = json.load(f)

# Extract mean consistency by level for each model
def extract_consistency_by_level(results):
    consistency_by_level = {i: [] for i in range(1, 8)}

    for q_result in results['experiment_results']:
        for level_analysis in q_result['level_analyses']:
            level = level_analysis['level']
            mean_sim = level_analysis['similarity_analysis']['mean_similarity']
            consistency_by_level[level].append(mean_sim)

    return {level: np.mean(scores) for level, scores in consistency_by_level.items()}

your_consistency = extract_consistency_by_level(your_results)
claude_consistency = extract_consistency_by_level(claude_results)
gpt4_consistency = extract_consistency_by_level(gpt4_results)

# Print comparison
print("Consistency Scores by Level:")
print(f"{'Level':<10} {'Your Model':<15} {'Claude':<15} {'GPT-4o mini':<15}")
print("-" * 55)

for level in range(1, 8):
    print(f"{level:<10} {your_consistency[level]:<15.4f} {claude_consistency[level]:<15.4f} {gpt4_consistency[level]:<15.4f}")
```

---

## 📚 Usage Workflow

```
1. Load Questions (7 levels per question)
   └─> data/MIRAE_augmented_questions_[language].json

2. Generate Responses (5 samples per level per question)
   └─> For each of 7 levels: generate 5 stochastic responses (temperature 1.0)

3. Calculate Semantic Consistency Per Level
   ├─> Primary:  python similarity/calculate_mE5_similarity.py
   └─> Baseline: python similarity/calculate_sbert_similarity.py
       ├─> Load embedding model
       ├─> For each level: add 'query: ' prefix (mE5 only), compute 5x5 similarity matrix
       ├─> Extract mean/std from the upper triangle
       └─> Save results with the level_analyses structure

4. Compare with Baselines by Level
   └─> Analyze consistency across levels 1-7 (example_results/mE5/ reproduces the paper)
```

---

## 🔧 Similarity Calculation

Two implementations are provided under `similarity/`:

### Primary — multilingual-e5-large (`similarity/calculate_mE5_similarity.py`)
- Cross-lingually aligned embeddings; used for all results reported in the paper.
- Adds the `"query: "` prefix required by mE5 to every response before encoding.
- Suitable for batch processing across all languages and models.

### Baseline — SBERT (`similarity/calculate_sbert_similarity.py`)
- Monolingual `all-MiniLM-L6-v2` embeddings, provided for cross-embedding comparison.
- Identical pipeline, without the `"query: "` prefix.

Both implementations:
- Process each question's 7 levels independently.
- Compute pairwise cosine similarities for the 5 responses per level.
- Extract mean/std/max/min from the upper triangle of the 5×5 matrix.
- Generate output matching the `example_results/` format shown in Step 3.4.

Providing both metrics makes the consistency measurement's sensitivity to the choice of embedding model directly inspectable, rather than relying on a single instrument.

---

## 📜 License

MIT License - see [LICENSE](LICENSE) for details.


For questions or issues, please open an issue on GitHub.
