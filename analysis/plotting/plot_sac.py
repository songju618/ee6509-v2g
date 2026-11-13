"""Generate Part B's SAC-only figures: training curves, typical days, attention.

Strategy comparisons against the official baselines are drawn by plot_official.
Run from the repository root with python -m analysis.plotting.plot_sac.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter, MaxNLocator
import numpy as np

from env import load_params
from env.recording import occupancy_mask


COLORS = ['#2563a6', '#d27927', '#208873', '#a24668', '#7b6caf']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.prop_cycle': plt.cycler(color=COLORS), 'figure.dpi': 130})


def save(fig, output, name):
    fig.savefig(output / f'{name}.png', bbox_inches='tight', dpi=180)
    fig.savefig(output / f'{name}.pdf', bbox_inches='tight')
    plt.close(fig)


def _read_jsonl(path_or_dir: Path):
    target = path_or_dir / 'training.jsonl' if path_or_dir.is_dir() else path_or_dir
    return [json.loads(line) for line in target.read_text().splitlines() if line.strip()]


def training_curve(runs, output):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for n, run in runs.items():
        rows = _read_jsonl(Path(run))
        train = [row for row in rows if 'train_cost_recent' in row]
        validation = [row for row in rows if 'validation_cost' in row]
        axes[0].plot([r['step'] for r in train], [r['train_cost_recent'] for r in train], label=f'N={n}')
        axes[1].plot([r['step'] for r in validation], [r['validation_cost'] for r in validation],
                     marker='o', ms=3, label=f'N={n}')
        best = min(validation, key=lambda r: r['validation_cost'])
        axes[1].scatter(best['step'], best['validation_cost'], marker='*', s=150, zorder=5)
    for ax in axes:
        ax.set_xlabel('Environment transitions')
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.xaxis.set_major_formatter(EngFormatter())
        ax.set_ylabel('Variable cost (SGD/day)')
        ax.legend(frameon=False)
        ax.grid(alpha=.2)
    axes[0].set_title('Training, stochastic policy, last 50 days')
    axes[1].set_title('Fixed validation grid, deterministic policy')
    save(fig, output, 'training_curves')


def read_csv(path):
    with path.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    return {key: np.array([float(row[key]) for row in rows])
            for key in rows[0] if key not in ('strategy', 'date')}


def typical_day(directory, n, output, date, seed):
    stem = directory / f'SAC_{date}_seed{seed}_n{n}'
    values = read_csv(stem.with_suffix('.csv'))
    scenario = json.loads(stem.with_suffix('.scenario.json').read_text())
    truth = np.asarray(scenario['truth'])
    time = np.arange(48) * .5
    edges = np.arange(49) * .5
    fig, ax = plt.subplots(figsize=(11, 4.4))
    ax.stairs(truth[1], edges, baseline=None, label='Building load', color='#777777', lw=1.4)
    ax.stairs(truth[2], edges, baseline=None, label='PV', color='#cda226')
    ax.stairs(values['p_bess_kw'], edges, baseline=None, label='BESS')
    ax.stairs(sum(values[f'p_ev_{i}_kw'] for i in range(1, n + 1)), edges, baseline=None, label='EV total')
    ax.stairs(values['p_grid_kw'], edges, baseline=None, label='Grid import', color='#1e6558', lw=2)
    price_ax = ax.twinx()
    price_ax.stairs(values['buy_price'], edges, baseline=None, color='#ac4263', ls='--', label='Buy price')
    price_ax.set_ylabel('Buy price (SGD/kWh)', color='#ac4263')
    ax.set(xlabel='Singapore local hour', ylabel='AC power (kW)', xlim=(0, 24),
           title=f'SAC dispatch, {date}, seed {seed}, N={n}')
    ax.axhline(0, color='black', lw=.5)
    lines, labels = ax.get_legend_handles_labels()
    extra, extra_labels = price_ax.get_legend_handles_labels()
    ax.legend(lines + extra, labels + extra_labels, ncol=3, loc='lower left',
              bbox_to_anchor=(0, 1.02), frameon=False, fontsize=8)
    ax.set_title(ax.get_title(), pad=50)
    ax.set_xticks(range(0, 25, 4))
    save(fig, output, f'power_price_n{n}_{date}')

    params = load_params()
    mask = occupancy_mask(scenario)
    energy = np.column_stack([values[f'e_ev_{i}_kwh'] for i in range(1, n + 1)])
    soc = np.where(mask, energy / params['ev']['capacity_kwh'], np.nan)
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    axes[0].plot(time + .5, values['e_bess_kwh'] / params['bess']['capacity_kwh'], label='BESS')
    axes[0].axhline(params['bess']['initial_soc'], ls='--', color='gray', label='Initial / terminal minimum')
    for slot in range(n):
        axes[1].plot(time + .5, soc[:, slot], color=plt.get_cmap('tab10')(slot), label=f'EV {slot + 1}')
    axes[1].axhline(params['ev']['departure_target_soc'], color='gray', ls='--', label='Departure target')
    for ax in axes:
        ax.set(ylabel='Post-step SoC', ylim=(.05, 1.0))
        ax.grid(alpha=.2)
        ax.legend(ncol=4, frameon=False, fontsize=8)
    axes[1].set(xlabel='Singapore local hour, end of interval', xlim=(0, 24))
    axes[1].set_xticks(range(0, 25, 4))
    fig.suptitle(f'Battery energy, occupied EV slots only, {date}, seed {seed}, N={n}')
    fig.tight_layout()
    save(fig, output, f'soc_n{n}_{date}')


def attention_plot(directory, n, output):
    rows = []
    for path in sorted(directory.glob('*.attention.json')):
        rows.extend(json.loads(path.read_text()))
    groups = {}
    for row in rows:
        if row['present_count'] > 1:
            groups.setdefault(int(row['remaining_steps']), []).append(
                row['received_attention'] * row['present_count'])
    fig, ax = plt.subplots(figsize=(8, 4))
    steps = sorted(groups)
    means = np.array([np.mean(groups[k]) for k in steps])
    lo = np.array([np.quantile(groups[k], .25) for k in steps])
    hi = np.array([np.quantile(groups[k], .75) for k in steps])
    ax.plot(steps, means, marker='o', ms=3, label='Mean received attention / uniform weight')
    ax.fill_between(steps, lo, hi, alpha=.2, label='Across-vehicle interquartile range')
    ax.axhline(1, ls='--', color='gray', label='Uniform attention')
    ax.set(xlabel='Remaining dwell time (half-hour steps)', ylabel='Relative received attention',
           title=f'Self-attention and departure urgency, N={n}')
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=.2)
    save(fig, output, f'attention_dwell_n{n}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n5-run', type=Path, default=Path('docs/B_SAC/training_logs/n5_training.jsonl'))
    parser.add_argument('--n10-run', type=Path, default=Path('docs/B_SAC/training_logs/n10_training.jsonl'))
    parser.add_argument('--results', type=Path, default=Path('results/sac'))
    parser.add_argument('--figures-dir', type=Path, default=None)
    parser.add_argument('--typical', nargs='+', default=['2026-10-28:100', '2026-10-22:100'],
                        help='DATE:SEED test scenarios for the typical-day figures '
                             '(default: the spike day 10-28 and the flat day 10-22)')
    args = parser.parse_args()
    output = args.figures_dir if args.figures_dir is not None else args.results / 'figures'
    output.mkdir(parents=True, exist_ok=True)
    training_curve({5: args.n5_run, 10: args.n10_run}, output)
    for n in (5, 10):
        directory = args.results / f'test_n{n}' / 'SAC'
        for item in args.typical:
            date, seed = item.split(':')
            typical_day(directory, n, output, date, int(seed))
        attention_plot(directory, n, output)


if __name__ == '__main__':
    main()
