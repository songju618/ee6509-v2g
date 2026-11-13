"""Draw the SAC network architecture in English for the report (network_architecture_en).

Same structure as docs/network_architecture.png: a 1-D CNN for the external time series,
self-attention across EV slots, a shared global feature, and N+1 power outputs.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from .plot_sac import save


EDGE, ARROW, BODY = '#b4b4ab', '#a3a39a', '#555555'
BLUE, BLUE_FILL = '#2563a6', '#e6eef8'


def box(ax, x, y, w, h, title, body, highlight=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.012',
                                linewidth=1.8 if highlight else 1.1,
                                edgecolor=BLUE if highlight else EDGE,
                                facecolor=BLUE_FILL if highlight else 'white'))
    ax.text(x + .012, y + h - .035, title, fontsize=11, fontweight='bold', va='top')
    ax.text(x + .012, y + h - .085, body, fontsize=9.2, color=BODY, va='top', linespacing=1.45)


def arrow(ax, start, end):
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='-|>', color=ARROW, lw=1.3, shrinkA=0, shrinkB=0,
                                mutation_scale=12))


def draw(output):
    fig, ax = plt.subplots(figsize=(12.5, 6.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.text(.01, .985, 'SAC network: CNN for the time series, self-attention across EV slots, '
            'N + 1 power outputs', fontsize=13, fontweight='bold', va='top')

    rows = {'ev': .62, 'ts': .385, 'global': .15}
    h = .175
    cols = {'input': (.01, .235), 'encoder': (.285, .225), 'head': (.725, .26)}
    box(ax, cols['input'][0], rows['ev'], cols['input'][1], h, 'EV slots × N',
        'occupancy, SoC, departure time,\nenergy needed, empty-slot mask')
    box(ax, cols['input'][0], rows['ts'], cols['input'][1], h, 'External time series',
        'USEP price, building load, PV:\n8 past + 12 forecast steps')
    box(ax, cols['input'][0], rows['global'], cols['input'][1], h, 'Global state',
        'BESS SoC, running grid peak,\ntime of day (sin / cos)')
    box(ax, cols['encoder'][0], rows['ev'], cols['encoder'][1], h, 'Self-attention, 4 heads',
        'per-vehicle MLP embedding,\npooled into a fleet feature', highlight=True)
    box(ax, cols['encoder'][0], rows['ts'], cols['encoder'][1], h, '1-D CNN',
        'time-series feature from\nthe 20-step window')
    concat_x, concat_w = .55, .13
    ax.add_patch(FancyBboxPatch((concat_x, rows['global']), concat_w, rows['ev'] + h - rows['global'],
                                boxstyle='round,pad=0,rounding_size=0.012', linewidth=1.1,
                                edgecolor=EDGE, facecolor='white'))
    middle = (rows['global'] + rows['ev'] + h) / 2
    ax.text(concat_x + concat_w / 2, middle + .03, 'Concatenate', fontsize=11.5, fontweight='bold',
            ha='center', va='center')
    ax.text(concat_x + concat_w / 2, middle - .025, 'global feature h', fontsize=9.6, color=BODY,
            ha='center', va='center')
    box(ax, cols['head'][0], rows['ev'], cols['head'][1], h, 'Actor · EV head',
        'h + per-vehicle feature, shared\nweights → N powers (tanh)')
    box(ax, cols['head'][0], rows['ts'], cols['head'][1], h, 'Actor · BESS head',
        'h → MLP → 1 power (tanh)')
    box(ax, cols['head'][0], rows['global'], cols['head'][1], h, 'Twin critic',
        'input h and action aₜ\noutput Q₁, Q₂')

    centre = {name: y + h / 2 for name, y in rows.items()}
    input_right = cols['input'][0] + cols['input'][1]
    encoder_left, encoder_right = cols['encoder'][0], cols['encoder'][0] + cols['encoder'][1]
    for name in ('ev', 'ts'):
        arrow(ax, (input_right, centre[name]), (encoder_left, centre[name]))
        arrow(ax, (encoder_right, centre[name]), (concat_x, centre[name]))
    arrow(ax, (input_right, centre['global']), (concat_x, centre['global']))
    for name in ('ev', 'ts', 'global'):
        arrow(ax, (concat_x + concat_w, centre[name]), (cols['head'][0], centre[name]))
    top = rows['ev'] + h
    attention_x = encoder_left + cols['encoder'][1] / 2
    head_x = cols['head'][0] + cols['head'][1] / 2
    ax.plot([attention_x, attention_x, head_x], [top, .885, .885], color=ARROW, lw=1.3)
    arrow(ax, (head_x, .885), (head_x, top))
    ax.text((attention_x + head_x) / 2, .9, 'per-vehicle features', fontsize=9.6, color=BODY,
            ha='center', va='bottom')

    ax.text(.02, .095, 'Training: the encoders are updated only by the critic loss; the actor’s '
            'gradient is detached at the encoder outputs.', fontsize=9.6, color=BODY)
    ax.text(.02, .045, 'Deployment: each action is mapped to its device’s feasible power range '
            '(0 → 0 kW when idling is feasible) and projected by the environment.',
            fontsize=9.6, color=BODY)
    save(fig, output, 'network_architecture_en')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--figures-dir', type=Path, default=Path('results/sac/figures'))
    args = parser.parse_args()
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    draw(args.figures_dir)


if __name__ == '__main__':
    main()
