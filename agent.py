from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
import torch
import os
import subprocess
import json
import sys
from datetime import datetime
import shlex

base_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
adapter_dir = "./model/qlora_tinyllama_cli_final"

tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    load_in_4bit=True,
    device_map="auto",
    torch_dtype=torch.float16
)

# Load LoRA adapter correctly
model = PeftModel.from_pretrained(model, adapter_dir)

# Create generation pipeline with explicit CUDA device
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)

# === Logging Utilities ===
LOG_PATH = "logs/trace.jsonl"
os.makedirs("logs", exist_ok=True)

def log_step(step):
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps({
            "step": step,
            "timestamp": datetime.utcnow().isoformat()
        }) + "\n")


def dry_run_command(cmd):
    try:
        parts = shlex.split(cmd)
        result = subprocess.run(["echo"] + parts, capture_output=True, text=True)
        return result.stdout.strip()
    except Exception as e:
        return f"Dry-run failed: {e}"

# === Response Generation ===
def generate_plan(instruction):
    prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
    try:
        output = pipe(prompt, max_new_tokens=200, do_sample=True, temperature=0.7)[0]['generated_text']
        plan = output.split("### Response:")[-1].strip().split("\n")
        return [step.strip("-•#> ") for step in plan if step.strip()]
    except Exception as e:
        return [f"Failed to generate steps: {e}"]

# === Entry Point ===
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python agent.py \"<your instruction>\"")
        sys.exit(1)

    instruction = sys.argv[1]
    print(f"\n🧠 Instruction: {instruction}\n")

    steps = generate_plan(instruction)

    for idx, step in enumerate(steps):
        print(f"Step {idx + 1}: {step}")
        log_step(step)

        # Run dry-run if command-like
        if idx == 0 and any(step.startswith(cmd) for cmd in ["git", "ls", "cd", "find", "python", "tar", "gzip", "pip", "grep"]):
            result = dry_run_command(step)
            print(f"💡 Dry-run: {result}")
