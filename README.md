# MIRAE: Micro-scale Interpretation Reliability & Alignment Evaluation

</div>

---

## 📋 Introduction

MIRAE is a benchmark dataset that evaluates how reliably LLMs interpret information at micro-scale input lengths (≤2K tokens) across different domains, languages, and models. Unlike conventional benchmarks that focus on accuracy, MIRAE measures **semantic consistency** — how stable model outputs remain when inputs or sampling vary, which is critical for resource-constrained deployments like mobile and edge devices.

---

## 📦 Dataset Structure

```
MIRAE/
├── data/
│   ├── MIRAE_augmented_questions_english.json     # 280 English prompts (4 domains × 10 questions × 7 token lengths)
│   ├── MIRAE_augmented_questions_korean.json      # 280 Korean prompts
│   ├── MIRAE_augmented_questions_chinese.json     # 280 Chinese prompts
│
├── example_results/
│   ├── claude_haiku/                              # Example results from Claude 3.5 Haiku
│   │   └── MIRAE_claude_haiku_results_english.json
│   │   └── MIRAE_claude_haiku_results_korean.json
│   │   └── MIRAE_claude_haiku_results_chinese.json
│   ├── gemini2.0flash/                            # Example results from Gemini 2.0 Flash
│   │   └── MIRAE_gemini2_results_english.json
│   │   └── MIRAE_gemini2_results_korean.json
│   │   └── MIRAE_gemini2_results_chinese.json
│   └── gpt4_mini/                                 # Example results from GPT-4o mini
│       └── MIRAE_gpt4omini_results_english.json
│       └── MIRAE_gpt4omini_results_korean.json
│       └── MIRAE_gpt4omini_results_chinese.json
│
├── sbert/
│   ├── calculate_sbert_similarity.ipynb            # SBERT calculation (Jupyter Notebook)
│   └── calculate_sbert_similarity.py               # SBERT calculation (Python Script)
│
├── requirements.txt
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

**Output: Example Results JSON** (`example_results/*/MIRAE_*_results_*.json`)

The results file contains 5 responses per question per level with calculated SBERT similarity metrics:

```json
{
  "metadata": {
    "research_project": "MIRAE",
    "experiment_type": "Multi-level Consistency Analysis (Levels 1-7)",
    "model": "gemini-2.0-flash",
    "language": "English",
    "num_repetitions": 5,
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "similarity_metric": "Cosine Similarity (SBERT)",
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
            "mean_similarity": 0.9984661340713501,
            "std_similarity": 0.001878623734228313,
            "max_similarity": 1.0,
            "min_similarity": 0.9961652755737305
          },
          "pairwise_similarities": [
            [1.0, 0.9961652755737305, 1.0, 1.0, 1.0],
            [0.9961652755737305, 1.0000001192092896, 0.9961652755737305, 0.9961652755737305, 0.9961652755737305],
            [1.0, 0.9961652755737305, 1.0, 1.0, 1.0],
            [1.0, 0.9961652755737305, 1.0, 1.0, 1.0],
            [1.0, 0.9961652755737305, 1.0, 1.0, 1.0]
          ],
          "responses": [
            "The country with the largest land area in the world is **Russia**, and its capital city is **Moscow**.",
            "The country with the largest land area in the world is **Russia**. Its capital city is **Moscow**.",
            "The country with the largest land area in the world is **Russia**, and its capital city is **Moscow**.",
            "The country with the largest land area in the world is **Russia**, and its capital city is **Moscow**.",
            "The country with the largest land area in the world is **Russia**, and its capital city is **Moscow**."
          ]
        },
        {
          "level": 2,
          "question_text": "For those interested in global geographical facts and national statistics, which country is recognized as the largest by land area globally?",
          "num_responses": 5,
          "similarity_analysis": {
            "mean_similarity": 0.9986810684204102,
            "std_similarity": 0.001076961518265307,
            "max_similarity": 1.0000001192092896,
            "min_similarity": 0.9978017807006836
          },
          "pairwise_similarities": [...],
          "responses": [...]
        }
      ]
    }
  ]
}
```

### Dataset Statistics

- **Questions:** 840 total (280 per language, 10 per domain)
- **Languages:** English, Korean, Chinese
- **Domains:** Factual, Analytical, Opinion, Creative
- **Token Lengths:** 7 levels (Level 1: ~30 tokens → Level 7: ~2000 tokens)
- **Models:** Gemini 2.0 Flash, GPT-4o mini, Claude 3.5 Haiku (example results provided for all languages)
- **Responses per Question per Level:** 5 stochastic samples (for consistency evaluation)
- **Total Responses:** 5 samples × 7 levels × 40 questions per domain = 1,400 responses per model per language

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

For each question, generate **5 responses per level** (5 stochastic samples for consistency evaluation). The key is to iterate through all 7 levels:

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
                temperature=0.7,  # Enable stochastic sampling
                max_tokens=256
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

### Step 3: Calculate SBERT Consistency

Choose your preferred method to compute semantic consistency for each level:

#### Option A: Jupyter Notebook

```bash
jupyter notebook sbert/calculate_sbert_similarity.ipynb
```

#### Option B: Python Script

```bash
python sbert/calculate_sbert_similarity.py
```

Both implementations compute SBERT similarities **for each level separately**:

#### 3.1 Load SBERT Model

```python
from sentence_transformers import SentenceTransformer, util
import numpy as np

print("Loading SBERT model...")
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
print("✓ SBERT model loaded")
```

#### 3.2 Calculate Per-Level Similarity Function

```python
def calculate_level_similarity(responses):
    """
    Calculate SBERT similarity for responses at a single level.
    
    Args:
        responses: List of 5 response strings for one level
        
    Returns:
        Dictionary with similarity metrics
    """
    # Compute embeddings for all 5 responses
    embeddings = sbert_model.encode(responses, convert_to_tensor=True)
    
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

#### 3.3 Process All Questions and Levels

```python
import json

def process_responses_with_sbert(input_file, output_file):
    """
    Load responses and calculate SBERT similarities for each level.
    Produces output in the same format as example_results.
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
            
            # Calculate SBERT similarity for this level
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
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "similarity_metric": "Cosine Similarity (SBERT)",
            "total_questions_analyzed": len(experiment_results),
            "levels_analyzed": "1-7"
        },
        "experiment_results": experiment_results
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"✓ Results saved to {output_file}")
    return output_data

# Run the processing
process_responses_with_sbert(
    input_file='my_model_responses.json',
    output_file='my_model_results_english.json'
)
```

#### 3.4 Expected Output Format

Your output file will match the example_results structure:

```json
{
  "metadata": {
    "research_project": "MIRAE",
    "experiment_type": "Multi-level Consistency Analysis (Levels 1-7)",
    "model": "your_model",
    "language": "english",
    "num_repetitions": 5,
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "similarity_metric": "Cosine Similarity (SBERT)",
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
            "mean_similarity": 0.998,
            "std_similarity": 0.002,
            "max_similarity": 1.0,
            "min_similarity": 0.996
          },
          "pairwise_similarities": [[1.0, 0.996, ...], ...],
          "responses": ["Response 1", "Response 2", "Response 3", "Response 4", "Response 5"]
        },
        {
          "level": 2,
          "question_text": "For those interested in global geographical facts...",
          "num_responses": 5,
          "similarity_analysis": {...},
          "pairwise_similarities": [...],
          "responses": [...]
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
import matplotlib.pyplot as plt

# Load your results
with open('my_model_results_english.json') as f:
    your_results = json.load(f)

# Load example results from different models
with open('example_results/claude_haiku/MIRAE_claude_haiku_results_english.json') as f:
    claude_results = json.load(f)

with open('example_results/gpt4_mini/MIRAE_gpt4omini_results_english.json') as f:
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
   └─> For each of 7 levels: generate 5 stochastic responses

3. Calculate SBERT Consistency Per Level (Choose One)
   ├─> Jupyter: sbert/calculate_sbert_similarity.ipynb
   └─> Python: python sbert/calculate_sbert_similarity.py
       ├─> Load SBERT model
       ├─> For each level: compute 5x5 similarity matrix
       ├─> Extract mean/std from upper triangle
       └─> Save results with level_analyses structure

4. Compare with Baselines by Level
   └─> Analyze consistency degradation across levels 1-7
```

---

## 🔧 SBERT Calculation

We provide two implementations for SBERT similarity calculation:

### Jupyter Notebook (`sbert/calculate_sbert_similarity.ipynb`)
- Interactive notebook for step-by-step calculation
- Suitable for exploration and level-by-level analysis
- Includes data visualization and trend analysis

### Python Script (`sbert/calculate_sbert_similarity.py`)
- Standalone script for batch processing multiple languages/models
- Configurable for production pipelines
- Suitable for processing large response sets

Both implementations:
- Load the SBERT model (`all-MiniLM-L6-v2`)
- Process each question's 7 levels independently
- Compute pairwise cosine similarities for 5 responses per level
- Generate output matching example_results format

Both produce identical results and generate the same output format shown in Step 3.4.

---

## 📜 License

MIT License - see [LICENSE](LICENSE) for details.


For questions or issues, please open an issue on GitHub.
