#!/usr/bin/env python
import os
import subprocess
from pathlib import Path

# Base settings (edit if needed)
PYTHON = "python"  # or full path to python.exe 
TRAIN_SCRIPT = "scripts/train_lora.py"

MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B"
DATA_PATH = "data/gsm8k/sft.jsonl"
BASE_OUTPUT_DIR = "outputs/lora_sweep"   # all runs go under here

NUM_EPOCHS = 3 # Reduced from 10 as SFT usually converges faster, but adjust as needed
MICRO_BATCH_SIZE = 8
GRAD_ACC = 3
EFFECTIVE_BATCH_SIZE = MICRO_BATCH_SIZE * GRAD_ACC

# ---- define the LoRA configs you want to try ----
LORA_CONFIGS = [
     {"r": 1, "alpha": 4, "dropout": 0.05, "use_dora": False},
     {"r": 4, "alpha": 16, "dropout": 0.05, "use_dora": False},
     {"r": 16, "alpha": 32, "dropout": 0.05, "use_dora": False},
]

def main():
    Path(BASE_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    for cfg in LORA_CONFIGS:
        r = cfg["r"]
        alpha = cfg["alpha"]
        dropout = cfg["dropout"]
        use_dora = cfg.get("use_dora", False) # Default to False if not specified

        # Naming convention for output folder
        run_name_parts = [f"r{r}", f"a{alpha}", f"d{str(dropout).replace('.', 'p')}"]
        if use_dora:
            run_name_parts.append("dora")
        run_name = "_".join(run_name_parts)
        
        out_dir = os.path.join(BASE_OUTPUT_DIR, run_name)

        print("=" * 80)
        print(f"Training LoRA config: {run_name}")
        print(f"  r={r}, alpha={alpha}, dropout={dropout}, use_dora={use_dora}")
        print(f"  output_dir={out_dir}")
        print("=" * 80)

        cmd = [
            PYTHON, TRAIN_SCRIPT,
            "--model-name", MODEL_NAME,
            "--sft-data-path", DATA_PATH,
            "--output-dir", out_dir,
            "--num-epochs", str(NUM_EPOCHS),
            "--batch-size", str(EFFECTIVE_BATCH_SIZE),
            "--microbatch-size", str(MICRO_BATCH_SIZE),
            "--lora-rank", str(r),
            "--lora-alpha", str(alpha),
            "--lora-dropout", str(dropout),
            "--run-name", run_name,
            "--project-name", "lora-sweep",
        ]
        
        if use_dora:
            cmd.append("--use-dora")

        print("Running command:\n", " ".join(cmd))
        subprocess.run(cmd, check=True)

    print("\nAll LoRA configs finished.")

if __name__ == "__main__":
    main()
