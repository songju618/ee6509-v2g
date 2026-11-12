"""Generate percentage metrics, price-regime decomposition, and 0-10 fleet figures."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .plot_sac import COLORS, save


def percent(report, key, strategy='SAC'):
    value = report['percentage_by_strategy'][strategy][key]['percent']
    return np.nan if value is None else value


def labeled_bars(ax, x, values, width, label=None, color=None):
    container = ax.bar(x, values, width, label=label, color=color)
    labels = ['' if not np.isfinite(v) else f'{v:.1f}%' for v in values]
    ax.bar_label(container, labels=labels, padding=3, fontsize=8)
    return container


def percentage_figure(reports, output):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    x = np.arange(2)
    for offset, key, label in [(-.18, 'optimality_gap_closure_from_immediate', 'Immediate gap'),
                               (.18, 'optimality_gap_closure_from_random', 'Random gap')]:
        values = [percent(reports[n], key) for n in (5, 10)]
        labeled_bars(axes[0], x + offset, values, .34, label)
    axes[0].axhline(100, color='gray', ls='--', lw=1)
    axes[0].set_title('Optimizable cost gap captured')
    axes[0].legend(frameon=False, fontsize=8)
    values = [percent(reports[n], 'net_ev_storage_cost_reduction') for n in (5, 10)]
    labeled_bars(axes[1], x, values, .5, color=COLORS[2])
    axes[1].axhline(100, color='gray', ls='--', lw=1)
    axes[1].set_title('Net EV and storage cost reduction')
    for offset, strategy, label in [(-.24, 'immediate', 'Immediate'), (0, 'rule', 'Rule'), (.24, 'random', 'Random')]:
        values = [reports[n]['comparisons'][strategy]['saving_percent'] for n in (5, 10)]
        labeled_bars(axes[2], x + offset, values, .22, label)
    axes[2].set_title('Whole-variable-bill saving')
    axes[2].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.set_xticks(x, ['N=5', 'N=10'])
        ax.set_ylabel('Percent')
        ax.margins(y=.18)
        ax.grid(axis='y', alpha=.15)
        ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, output, 'percentage_metrics')


def incremental_components(ax, report, title, legend=False):
    strategies = ['SAC', 'immediate', 'rule', 'random', 'offline_opt']
    building = report['building_only']
    energy = np.array([report[s]['purchase_sgd'] - report[s]['revenue_sgd']
                       - (building['purchase_sgd'] - building['revenue_sgd']) for s in strategies])
    peak = np.array([report[s]['peak_increment_sgd'] - building['peak_increment_sgd'] for s in strategies])
    wear = np.array([report[s]['bess_wear_sgd'] + report[s]['ev_wear_sgd'] for s in strategies])
    above = np.zeros(5)
    below = np.zeros(5)
    for values, label, color in [(energy, 'Net energy above building', COLORS[0]),
                                 (peak, 'Capacity above building', COLORS[1]),
                                 (wear, 'Battery wear', COLORS[2])]:
        ax.bar(strategies, values, bottom=np.where(values >= 0, above, below), label=label, color=color)
        above += np.maximum(values, 0)
        below += np.minimum(values, 0)
    ax.scatter(strategies, energy + peak + wear, marker='D', color='black', s=16, label='Net increment', zorder=3)
    ax.axhline(0, color='black', lw=.6)
    ax.set_title(title)
    ax.set_ylabel('Increment above building bill (SGD/day)')
    ax.tick_params(axis='x', rotation=20)
    if legend:
        ax.legend(frameon=False, fontsize=7)


def regime_figure(reports, output):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for row, n in enumerate((5, 10)):
        for column, (key, label) in enumerate([('flat_regular', 'Flat / regular'),
                                               ('high_volatility_spike', 'High volatility / spike')]):
            if key not in reports[n]['price_groups']:
                axes[row, column].set_visible(False)
                continue
            group = reports[n]['price_groups'][key]
            saving = group['comparisons']['immediate']['saving_percent']
            title = f'{label}, N={n}, {len(group["dates"])} dates\nSAC saving vs immediate {saving:.2f}%'
            incremental_components(axes[row, column], group, title, row == 0 and column == 0)
    fig.tight_layout()
    save(fig, output, 'price_regime_cost_components')


def extended_figure(summary, output):
    families = summary['families']
    counts = [0, 2, 4, 6, 8, 10]
    ladder = [families[f'ladder_{n}'] for n in counts]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for key, label, style in [('optimality_gap_closure_from_immediate', 'SAC, immediate gap', '-o'),
                              ('optimality_gap_closure_from_random', 'SAC, random gap', '-s')]:
        axes[0, 0].plot(counts, [percent(r, key) for r in ladder], style, label=label, ms=4)
    axes[0, 0].plot(counts, [percent(r, 'optimality_gap_closure_from_immediate', 'rule') for r in ladder],
                    '--^', label='Rule, immediate gap', ms=4)
    axes[0, 0].axhline(100, color='gray', ls=':', label='Offline optimum')
    axes[0, 0].set_title('Optimizable cost gap captured')
    for strategy, label in [('SAC', 'SAC'), ('rule', 'Rule'), ('offline_opt', 'Offline optimum')]:
        axes[0, 1].plot(counts, [percent(r, 'net_ev_storage_cost_reduction', strategy) for r in ladder],
                        '-o', label=label, ms=4)
    axes[0, 1].text(.02, .20, '0 vehicles · baseline increment is zero', transform=axes[0, 1].transAxes,
                    va='bottom', fontsize=8)
    axes[0, 1].set_title('Net EV and storage cost reduction')
    for ax in axes[0]:
        ax.set_xticks(counts)
        ax.set_xlabel('Vehicles arriving during the day, 10 physical slots')
        ax.set_ylabel('Percent')
        ax.grid(alpha=.2)
        ax.legend(frameon=False, fontsize=8, loc='best')
    mixed = families['mixed']
    names = ['Immediate gap', 'Random gap', 'Net EV + storage']
    keys = ['optimality_gap_closure_from_immediate',
            'optimality_gap_closure_from_random',
            'net_ev_storage_cost_reduction']
    values = [percent(mixed, key) for key in keys]
    labeled_bars(axes[1, 0], np.arange(3), values, .6, color=COLORS[:3])
    axes[1, 0].set_xticks(np.arange(3), names)
    axes[1, 0].set_title('SAC in mixed day / night parking')
    axes[1, 0].set_ylabel('Percent')
    axes[1, 0].margins(y=.18)
    incremental_components(axes[1, 1], mixed, 'Mixed parking cost components', legend=True)
    fig.tight_layout()
    save(fig, output, 'extended_0_to_10_percentage')
    fig, ax = plt.subplots(figsize=(10, 4))
    time = np.arange(48) * .5
    for n in counts:
        ax.step(time, families[f'ladder_{n}']['actual_occupancy_mean_profile'], where='post', label=f'{n} arrivals')
    ax.step(time, mixed['actual_occupancy_mean_profile'], where='post', label='Mixed day / night', lw=3, color='black')
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
    reports = {n: json.loads((args.results / f'test_n{n}' / 'comparison.json').read_text()) for n in (5, 10)}
    percentage_figure(reports, output)
    regime_figure(reports, output)
    extended_figure(json.loads((args.results / 'extended_realistic' / 'summary.json').read_text()), output)


if __name__ == '__main__':
    main()
