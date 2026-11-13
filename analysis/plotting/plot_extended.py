"""Occupancy profile of the held-out 0-10 vehicle suite.

Strategy comparisons on this suite are drawn by plot_official against the official baselines.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .plot_sac import save


COUNTS = (0, 2, 4, 6, 8, 10)


def occupancy_figure(summary, output):
    families = summary['families']
    fig, ax = plt.subplots(figsize=(10, 4))
    time = np.arange(48) * .5
    colors = plt.get_cmap('viridis')(np.linspace(0, .9, len(COUNTS)))
    for n, color in zip(COUNTS, colors):
        ax.step(time, families[f'ladder_{n}']['actual_occupancy_mean_profile'], where='post',
                label=f'{n} arrivals', color=color)
    ax.step(time, families['mixed']['actual_occupancy_mean_profile'], where='post',
            label='Mixed day / night', lw=3, color='black')
    ax.set(xlabel='Singapore local hour', ylabel='Mean occupied slots', xlim=(0, 24), ylim=(-.2, 10.5),
           title='Actual occupancy in the held-out realistic suite')
    ax.set_xticks(range(0, 25, 4))
    ax.legend(frameon=False, ncol=3, fontsize=8)
    ax.grid(alpha=.2)
    save(fig, output, 'extended_occupancy')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path('results/sac'))
    parser.add_argument('--figures-dir', type=Path, default=None)
    args = parser.parse_args()
    output = args.figures_dir if args.figures_dir is not None else args.results / 'figures'
    output.mkdir(parents=True, exist_ok=True)
    occupancy_figure(json.loads((args.results / 'extended_realistic' / 'summary.json').read_text()), output)


if __name__ == '__main__':
    main()
