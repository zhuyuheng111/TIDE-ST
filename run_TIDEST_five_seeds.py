# -*- coding: utf-8 -*-
"""
TIDE-ST five-seed runner

Run TIDE-ST training five times with different random seeds.
Keep DATA_PARTITION_SEED, PSEUDOSPOT_SEED and selected hyperparameters fixed.
"""

import os
import subprocess
import time
import pandas as pd

TRAIN_SCRIPT = r"D:\your_path\TIDE-ST_train.py"

SEEDS = [1, 2, 3, 4, 5]

OUTPUT_DIR = r"E:\TIDEST_five_seed_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

records = []

for seed in SEEDS:
    print("=" * 80)
    print(f"Running seed {seed}")
    print("=" * 80)

    log_file = os.path.join(OUTPUT_DIR, f"seed_{seed}.log")

    start = time.time()

    cmd = [
        "python",
        TRAIN_SCRIPT,
        "--seed",
        str(seed)
    ]

    with open(log_file, "w", encoding="utf-8") as f:
        result = subprocess.run(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT
        )

    records.append({
        "seed": seed,
        "return_code": result.returncode,
        "runtime_minutes": (time.time() - start) / 60
    })

summary = pd.DataFrame(records)

summary.to_csv(
    os.path.join(OUTPUT_DIR, "five_seed_training_summary.csv"),
    index=False
)

print(summary)
