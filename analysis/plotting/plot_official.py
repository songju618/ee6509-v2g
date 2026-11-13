"""Compare SAC with the official baselines: arrival charge, price rule and MILP.

Reads the JSON written by `python -m sac.official` and writes the comparison figures and
CSV tables: mean costs over the 140 test cases, per-day differences with intervals that
resample test days, and per-case counts.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .plot_sac import COLORS, save


ORDER = ('arrival_charge', 'price_threshold', 'SAC', 'milp')
BASELINES = ('arrival_charge', 'price_threshold')
LABELS = {'arrival_charge': 'Arrival charge\n(no storage)', 'price_threshold': 'Price rule',
          'SAC': 'SAC', 'milp': 'MILP\n(offline)'}
SHORT = {'arrival_charge': 'Arrival charge', 'price_threshold': 'Price rule', 'SAC': 'SAC', 'milp': 'MILP'}
STYLE = {'arrival_charge': ('^', '#7f7f7f'), 'price_threshold': ('s', COLORS[1]),
         'SAC': ('D', COLORS[0]), 'milp': ('o', COLORS[2])}
COMPONENTS = [('purchase_sgd', 'Purchase'), ('revenue_sgd', 'Export revenue (negative)'),
              ('peak_increment_sgd', 'Excess capacity'), ('bess_wear_sgd', 'BESS wear'),
              ('ev_wear_sgd', 'EV wear')]
FAMILIES = ('ladder_0', 'ladder_2', 'ladder_4', 'ladder_6', 'ladder_8', 'ladder_10', 'mixed')


def cost_components(reports, output):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5), sharey=True)
    x = np.arange(len(ORDER))
    for ax, (n, report) in zip(axes, reports.items()):
        bottom = np.zeros(len(ORDER))
        for (key, label), color in zip(COMPONENTS, COLORS):
            values = np.array([report[s][key] for s in ORDER]) * (-1 if key == 'revenue_sgd' else 1)
            ax.bar(x, values, bottom=bottom, label=label, color=color, width=.65)
            bottom += values
        for i, s in enumerate(ORDER):
            ax.text(i, bottom[i] + 12, f"{report[s]['variable_cost_sgd']:.1f}", ha='center', fontsize=8.5)
        ax.set_xticks(x, [LABELS[s] for s in ORDER])
        ax.set_title(f"Test grid, N={n}, mean of {report['per_case']['cases']} cases")
    axes[0].set_ylabel('Mean variable cost (SGD/day)')
    axes[1].legend(frameon=False, fontsize=8, loc='upper left', bbox_to_anchor=(1.02, 1))
    fig.tight_layout()
    save(fig, output, 'cost_components')


def gap_position(reports, output):
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.9), gridspec_kw={'width_ratios': [1.4, 1]})
    ax = axes[0]
    for row, (n, report) in enumerate(reports.items()):
        y = 1 - row
        ax.hlines(y, 0, 100, color='#d0d0d0', lw=2, zorder=1)
        for s in ORDER:
            position = report['gap_position_pct'][s]
            marker, color = STYLE[s]
            ax.scatter(position, y, marker=marker, color=color, s=70, zorder=3,
                       label=SHORT[s] if row == 0 else None)
            ax.annotate(f'{position:.0f}%', (position, y), textcoords='offset points', xytext=(0, 9),
                        ha='center', fontsize=8)
    ax.set_yticks([1, 0], ['N=5', 'N=10'])
    ax.set_xlim(-5, 105)
    ax.set_ylim(-.6, 1.6)
    ax.set_xlabel('Mean cost position: 0% = MILP lower bound, 100% = arrival charge')
    ax.set_title('Where each strategy sits between the bounds')
    ax.legend(frameon=False, fontsize=8, ncol=4, loc='upper center', bbox_to_anchor=(.5, -.28))
    ax = axes[1]
    x = np.arange(2)
    width = .36
    for k, b in enumerate(BASELINES):
        values = [100 * reports[n]['per_case']['sac_cheaper_cases'][b] / reports[n]['per_case']['cases']
                  for n in (5, 10)]
        days = [f"{reports[n]['days_won_by_sac'][b]}/{reports[n]['days']} days" for n in (5, 10)]
        bars = ax.bar(x + (k - .5) * width, values, width, color=STYLE[b][1], label=f'vs {SHORT[b].lower()}')
        ax.bar_label(bars, labels=[f'{v:.0f}%\n{d}' for v, d in zip(values, days)], fontsize=7.5, padding=2)
    ax.axhline(50, ls='--', color='gray', lw=.8)
    ax.set_xticks(x, ['N=5', 'N=10'])
    ax.set_ylim(0, 125)
    ax.set_ylabel('Cases where SAC is cheaper (%)')
    ax.set_title('Share of test cases where SAC is cheaper')
    ax.legend(frameon=False, fontsize=8, ncol=2, loc='upper center', bbox_to_anchor=(.5, -.12))
    fig.tight_layout()
    save(fig, output, 'gap_position')


def daily_differences(reports, output):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), sharey=True)
    for ax, (n, report) in zip(axes, reports.items()):
        dates = sorted(report['comparisons']['price_threshold']['saving_by_date_sgd'])
        spike = set(report['price_groups'].get('high_volatility_spike', {}).get('dates', []))
        x = np.arange(len(dates))
        width = .38
        for i, date in enumerate(dates):
            if date in spike:
                ax.axvspan(i - .5, i + .5, color='#f3dede', zorder=0)
        for k, b in enumerate(BASELINES):
            comparison = report['comparisons'][b]
            low, high = comparison['date_cluster_bootstrap_95_ci_sgd']
            values = [comparison['saving_by_date_sgd'][d] for d in dates]
            ax.bar(x + (k - .5) * width, values, width, color=STYLE[b][1], zorder=2,
                   label=f"vs {SHORT[b].lower()}: mean {comparison['mean_saving_sgd']:+.1f}, "
                         f"95% CI over days [{low:.0f}, {high:.0f}]")
        ax.axhline(0, color='black', lw=.7)
        ax.set_xticks(x, [d[5:] for d in dates])
        ax.set_xlabel('Test day (2026)')
        ax.set_title(f'N={n}: baseline cost minus SAC cost, per test day')
        ax.legend(frameon=False, fontsize=7.5, loc='upper left')
        ax.grid(axis='y', alpha=.2)
    axes[0].set_ylabel('SAC saving (SGD/day, mean of 20 seeds)')
    fig.text(.5, -.03, 'Positive = SAC cheaper. Shaded = high-volatility price days '
                       '(daily USEP range above the training 75th percentile).', ha='center', fontsize=8)
    fig.tight_layout()
    save(fig, output, 'daily_differences')


def regime_components(reports, output):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.4))
    x = np.arange(len(ORDER))
    for row, n in enumerate((5, 10)):
        for col, (key, label) in enumerate([('flat_regular', 'Flat / regular'),
                                            ('high_volatility_spike', 'High volatility / spike')]):
            ax = axes[row, col]
            group = reports[n]['price_groups'].get(key)
            if group is None:
                ax.set_visible(False)
                continue
            building = group['building_only']
            energy = np.array([group[s]['purchase_sgd'] - group[s]['revenue_sgd']
                               - (building['purchase_sgd'] - building['revenue_sgd']) for s in ORDER])
            peak = np.array([group[s]['peak_increment_sgd'] - building['peak_increment_sgd'] for s in ORDER])
            wear = np.array([group[s]['bess_wear_sgd'] + group[s]['ev_wear_sgd'] for s in ORDER])
            above, below = np.zeros(len(ORDER)), np.zeros(len(ORDER))
            for values, name, color in [(energy, 'Net energy above building', COLORS[0]),
                                        (peak, 'Capacity above building', COLORS[1]),
                                        (wear, 'Battery wear', COLORS[2])]:
                ax.bar(x, values, bottom=np.where(values >= 0, above, below), label=name, color=color, width=.65)
                above += np.maximum(values, 0)
                below += np.minimum(values, 0)
            ax.scatter(x, energy + peak + wear, marker='D', color='black', s=16, zorder=3, label='Net increment')
            ax.axhline(0, color='black', lw=.6)
            gap = group['price_threshold']['variable_cost_sgd'] - group['SAC']['variable_cost_sgd']
            ax.set_title(f'{label}, N={n}, {len(group["dates"])} dates\n'
                         f'Price rule minus SAC: {gap:+.1f} SGD/day')
            ax.set_xticks(x, [LABELS[s] for s in ORDER])
            ax.set_ylabel('Increment above building bill (SGD/day)')
            if row == 0 and col == 0:
                ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    save(fig, output, 'price_regime_cost_components')


def extended_comparison(extended, output):
    counts = [int(f.split('_')[1]) for f in FAMILIES[:-1]]
    xs = counts + [12]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    for s in ORDER:
        marker, color = STYLE[s]
        values = [extended[f][s]['variable_cost_sgd'] for f in FAMILIES]
        axes[0].plot(xs[:-1], values[:-1], marker=marker, color=color, label=SHORT[s], ms=4)
        axes[0].scatter(xs[-1], values[-1], marker=marker, color=color, s=30)
    axes[0].set_ylabel('Mean variable cost (SGD/day)')
    axes[0].set_title('Held-out suite, frozen N=10 model, 140 cases per family')
    for b in BASELINES:
        values = [extended[f]['gap_closed_by_sac_pct'][b] for f in FAMILIES]
        axes[1].plot(xs[:-1], values[:-1], marker='o', color=STYLE[b][1], label=f'vs {SHORT[b].lower()}', ms=4)
        axes[1].scatter(xs[-1], values[-1], marker='o', color=STYLE[b][1], s=30)
    axes[1].axhline(100, ls=':', color='gray', label='MILP')
    axes[1].set_ylim(0, 105)
    axes[1].set_ylabel('Baseline-to-MILP gap closed by SAC (%)')
    axes[1].set_title('Share of the gap SAC closes, by family')
    for ax in axes:
        ax.set_xticks(xs, [str(c) for c in counts] + ['Mixed\nday/night'])
        ax.set_xlabel('Vehicles arriving during the day (10 physical slots)')
        ax.grid(alpha=.2)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save(fig, output, 'extended_0_to_10_comparison')


def _fmt(value):
    if value is None:
        return ''
    return round(float(value), 4)


def write_tables(reports, extended, tables):
    tables.mkdir(parents=True, exist_ok=True)
    keys = ['variable_cost_sgd', 'total_cost_sgd', 'purchase_sgd', 'revenue_sgd', 'peak_increment_sgd',
            'bess_wear_sgd', 'ev_wear_sgd', 'peak_grid_kw', 'departure_success_rate']
    comparison_keys = ['sac_saving_sgd', 'sac_saving_pct', 'seed_resampled_ci_low', 'seed_resampled_ci_high',
                       'day_resampled_ci_low', 'day_resampled_ci_high', 'gap_closed_by_sac_pct',
                       'sac_cheaper_cases', 'cases', 'sac_cheaper_days', 'days']
    with (tables / 'official_test_summary.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['n_slots', 'strategy'] + keys + ['gap_position_pct'] + comparison_keys)
        for n, report in reports.items():
            for s in ORDER + ('building_only',):
                row = [n, s] + [_fmt(report[s][k]) for k in keys]
                row.append(_fmt(report['gap_position_pct'].get(s)))
                if s in BASELINES:
                    c = report['comparisons'][s]
                    row += [_fmt(c['mean_saving_sgd']), _fmt(c['saving_percent']),
                            *map(_fmt, c['fixed_date_seed_bootstrap_95_ci_sgd']),
                            *map(_fmt, c['date_cluster_bootstrap_95_ci_sgd']),
                            _fmt(report['gap_closed_by_sac_pct'][s]),
                            report['per_case']['sac_cheaper_cases'][s], report['per_case']['cases'],
                            report['days_won_by_sac'][s], report['days']]
                else:
                    row += [''] * len(comparison_keys)
                writer.writerow(row)
    with (tables / 'official_daily_differences.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['n_slots', 'baseline', 'date', 'price_regime', 'baseline_minus_sac_sgd'])
        for n, report in reports.items():
            spike = set(report['price_groups'].get('high_volatility_spike', {}).get('dates', []))
            for b in BASELINES:
                for date, value in sorted(report['comparisons'][b]['saving_by_date_sgd'].items()):
                    regime = 'high_volatility_spike' if date in spike else 'flat_regular'
                    writer.writerow([n, b, date, regime, _fmt(value)])
    if extended is None:
        return
    with (tables / 'official_extended_summary.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['family', 'arriving_vehicles', 'physical_slots']
                        + [f'{s}_cost_sgd' for s in ORDER]
                        + [f'gap_closed_vs_{b}_pct' for b in BASELINES]
                        + [f'sac_cheaper_cases_vs_{b}' for b in BASELINES]
                        + [f'sac_cheaper_days_vs_{b}' for b in BASELINES]
                        + ['day_resampled_ci_low_vs_price_threshold', 'day_resampled_ci_high_vs_price_threshold'])
        for family in FAMILIES:
            report = extended[family]
            vehicles = family.split('_')[1] if family.startswith('ladder_') else 'day_3_to_8_plus_night_1_to_2'
            writer.writerow([family, vehicles, 10]
                            + [_fmt(report[s]['variable_cost_sgd']) for s in ORDER]
                            + [_fmt(report['gap_closed_by_sac_pct'][b]) for b in BASELINES]
                            + [report['per_case']['sac_cheaper_cases'][b] for b in BASELINES]
                            + [report['days_won_by_sac'][b] for b in BASELINES]
                            + list(map(_fmt, report['comparisons']['price_threshold']
                                       ['date_cluster_bootstrap_95_ci_sgd'])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--official', type=Path, default=Path('results/sac/official'),
                        help='directory written by python -m sac.official')
    parser.add_argument('--figures-dir', type=Path, default=Path('results/sac/figures'))
    parser.add_argument('--tables-dir', type=Path, default=Path('results/sac/tables'))
    args = parser.parse_args()
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    reports = {n: json.loads((args.official / f'test_n{n}.json').read_text()) for n in (5, 10)}
    extended_path = args.official / 'extended.json'
    extended = json.loads(extended_path.read_text()) if extended_path.exists() else None
    cost_components(reports, args.figures_dir)
    gap_position(reports, args.figures_dir)
    daily_differences(reports, args.figures_dir)
    regime_components(reports, args.figures_dir)
    if extended is not None:
        extended_comparison(extended, args.figures_dir)
    write_tables(reports, extended, args.tables_dir)


if __name__ == '__main__':
    main()
