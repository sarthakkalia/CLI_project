# CLI_project
Command Line Interface
# 🧠 Command-Line Q&A Model – Fine-Tuning and Evaluation

## 📌 Objective

- Fine-tune a base language model for domain-specific QA (command-line queries).
- Package and demonstrate the model in a terminal environment.

---

## ⚙️ Model & Dataset

- **Base Model**: [`TinyLlama/TinyLlama-1.1B-Chat`](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat)
- **Parameter Count**: 1.1B
- **Fine-tuning Method**: QLoRA (Low-Rank Adaptation)
- **Frameworks Used**: Hugging Face `transformers`, `peft`, `accelerate`
- **Dataset**: CLI instruction–response pairs (`command_qa.jsonl`)
    the data is scrape from the stack over flow and github only.
  - Sample format:
    ```json
    {
      "instruction": "How to list all files recursively in Python?",
      "output": "Use os.walk(). Example:\nimport os\nfor root, dirs, files in os.walk('.'):\n  for file in files:\n    print(os.path.join(root, file))"
    }
    ```

---

## 🔧 Training Configuration

- **Epochs**: 1
- **Batch Size**: 2
- **Learning Rate**: 2e-4
- **Precision**: 4-bit (using QLoRA)
- **Trained on**: NVIDIA A100 (Kaggle environment)

---

## 📊 Evaluation Summary

### Evaluation (Manual)
- Compared 7 common CLI questions across base vs. fine-tuned outputs.
- Focused on correctness, repetition reduction, and format improvement.


---

### 📈 Metric-Based Evaluation

| Metric | Score |
|--------|-------|
| **BLEU** (avg over dataset) | **0.39** |
| **ROUGE-L** | **0.82** |
| **F1 Score** | **0.84** |

> 📌 *Metrics computed using `evaluate` (Hugging Face) and `sacrebleu` libraries.*

---

## 🖥️ CLI Agent Demo

- Run `cli_agent.py`
