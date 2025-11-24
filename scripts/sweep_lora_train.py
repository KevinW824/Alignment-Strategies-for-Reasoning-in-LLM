#!/usr/bin/env python
import os
import subprocess
from pathlib import Path

# Base settings (edit if needed)
PYTHON = "python"  # or full path to python.exe 
TRAIN_SCRIPT = "scripts/train_lora1.py"

MODEL_PATH = "Qwen/Qwen2.5-Math-1.5B"
DATA_PATH = "data/gsm8k/train.jsonl"
BASE_OUTPUT_DIR = "outputs/lora_sweep"   # all runs go under here

NUM_EPOCHS = 10 # may need more
BATCH_SIZE = 2
GRAD_ACC = 8

# ---- define the LoRA configs you want to try ----
LORA_CONFIGS = [
    #{"r": 1,  "alpha": 16,  "dropout": 0.05},
    #{"r": 4,  "alpha": 16,  "dropout": 0.05},
    # {"r": 8,  "alpha": 16,  "dropout": 0.05},
    # {"r": 16, "alpha": 16,  "dropout": 0.05},
    # {"r": 4, "alpha": 16,  "dropout": 0.05},
    # {"r": 32, "alpha": 32,  "dropout": 0.05},
    #{"r": 4,  "alpha": 32,  "dropout": 0.05},
    # {"r": 8,  "alpha": 32,  "dropout": 0.05},
    # {"r": 16, "alpha": 32,  "dropout": 0.10},
    #{"r": 8,  "alpha": 16,  "dropout": 0.10},
]

def main():
    Path(BASE_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    for cfg in LORA_CONFIGS:
        r = cfg["r"]
        alpha = cfg["alpha"]
        dropout = cfg["dropout"]

        run_name = f"r{r}_a{alpha}_d{str(dropout).replace('.', 'p')}"
        out_dir = os.path.join(BASE_OUTPUT_DIR, run_name)

        print("=" * 80)
        print(f"Training LoRA config: {run_name}")
        print(f"  r={r}, alpha={alpha}, dropout={dropout}")
        print(f"  output_dir={out_dir}")
        print("=" * 80)

        cmd = [
            PYTHON, TRAIN_SCRIPT,
            "--model_path", MODEL_PATH,
            "--sft_data_path", DATA_PATH,
            "--output_dir", out_dir,
            "--num_epochs", str(NUM_EPOCHS),
            "--batch_size", str(BATCH_SIZE),
            "--gradient_accumulation_steps", str(GRAD_ACC),
            "--lora_r", str(r),
            "--lora_alpha", str(alpha),
            "--lora_dropout", str(dropout),
            # optional: tweak anything else here (lr, max_examples, etc.)
        ]

        print("Running command:\n", " ".join(cmd))
        subprocess.run(cmd, check=True)

    print("\nAll LoRA configs finished.")

if __name__ == "__main__":
    main()
