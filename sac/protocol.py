"""Explicit experimental grids and deterministic local runtime settings."""
import json
import os
from pathlib import Path
import random

import numpy as np
import torch

VALIDATION_SEEDS = (0, 20, 40, 60, 80)


def grid(split, seeds=None):
    bounds = {'train': (1, 14), 'validation': (15, 21), 'test': (22, 28)}
    first, last = bounds[split]
    if seeds is None:
        seeds = range(100, 120) if split == 'test' else VALIDATION_SEEDS
    return [(f'2026-10-{day:02d}', int(seed)) for day in range(first, last + 1) for seed in seeds]


def configure(seed=6509, threads=1):
    torch.set_num_threads(threads)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def save_checkpoint(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary)
    os.replace(temporary, path)
