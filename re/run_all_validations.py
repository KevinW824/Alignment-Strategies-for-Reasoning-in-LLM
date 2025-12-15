import os
import torch
import numpy as np
import json
from src.validate import ValidationConfig, run_validation, plot_results
from src.train import REConfig, run_re_pipeline


def main():
    models_to_process = [
        {
            "name": "Qwen/Qwen2.5-Math-1.5B",
            "train_data": "data/qwen25_math15b_prompts_with_format.json",
            "vector_output_dir": "outputs/re",
            "vector_filename": "contrastive_pca_vectors_qwen2.5_math_with_format_1.5B.pth",
            "val_data": "../data/gsm8k/test.jsonl",
        },
        # {
        #     "name": "Qwen/Qwen3-1.7B",
        #     "train_data": "data/qwen3_17b_prompts_with_format.json",
        #     "vector_output_dir": "outputs/re",
        #     "vector_filename": "contrastive_pca_vectors_qwen3_with_format_1.7B.pth",
        #     "val_data": "../data/gsm8k/test.jsonl",
        # },
    ]

    alpha_start = -0.4
    alpha_end = 0.4
    alpha_step = 0.2

    for model_info in models_to_process:
        model_name = model_info["name"]
        print(f"\n\n{'='*20} Processing Model: {model_name} {'='*20}")

        # --- Step 1: Training (Generate Control Vector) ---
        # print(f"\n--- Step 1: Training (Generating Control Vector) ---")
        # print(f"Training data: {model_info['train_data']}")
        
        # train_config = REConfig(
        #     model_name=model_name,
        #     data_path=model_info["train_data"],
        #     output_dir=model_info["vector_output_dir"],
        #     output_file=model_info["vector_filename"],
        #     batch_size=8
        # )
        
        # # Run the training pipeline
        # run_re_pipeline(train_config)
        
        # Construct full path to the generated vector
        vector_full_path = os.path.join(model_info["vector_output_dir"], model_info["vector_filename"])
        print(f"Vector generated at: {vector_full_path}")

        # --- Step 2: Validation ---
        print(f"\n--- Step 2: Validation ---")
        print(f"Validation data: {model_info['val_data']}")
        print(f"Using vector: {vector_full_path}")

        val_config = ValidationConfig(
            model_name=model_name,
            val_data_path=model_info["val_data"],
            vector_file_path=vector_full_path,
            alpha_start=alpha_start,
            alpha_end=alpha_end,
            alpha_step=alpha_step,
            batch_size=24,
            max_new_tokens=1024,
            max_examples=None,
            output_dir=os.path.join("outputs", model_name.split("/")[-1]),
            require_format=False
        )

        # Run validation
        results_file, plot_file, injection_layer = run_validation(val_config)

        # Plot results if successful
        if results_file and plot_file and injection_layer is not None:
            plot_results(results_file, plot_file, injection_layer)
        else:
            print(
                f"Validation for {model_name} failed or did not produce results for plotting."
            )


if __name__ == "__main__":
    main()
