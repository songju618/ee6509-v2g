"""Train on October 1–14 only, choose checkpoints on the fixed validation grid."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from env import V2GEnv
from .agent import SAC
from .evaluate import evaluate_grid
from .normalization import Normalizer
from .policy import Policy
from .protocol import configure, grid, save_checkpoint, write_json
from .replay import Replay
from .physical import environment_action
from .scenarios import training_scenario
from .building import BuildingMeter


def training_observations(n_slots, seeds, dynamic=False):
    environment = V2GEnv(n_slots=n_slots)
    rng = np.random.default_rng(650900)
    try:
        for date, seed in grid('train', seeds):
            if dynamic:
                mode = ('standard', 'standard', 'thinned', 'mixed')[seed % 4]
                scenario = training_scenario(date, seed, n_slots, mode)
                obs, _ = environment.reset(seed=seed, options={'scenario': scenario})
            else:
                obs, _ = environment.reset(seed=seed, options={'date': date})
            for _ in range(48):
                yield obs
                obs, _, done, _, _ = environment.step(rng.uniform(-1., 1., n_slots + 1))
                if done:
                    break
    finally:
        environment.close()


def train(config_path, n_slots, output_dir):
    config = json.loads(Path(config_path).read_text())
    configure(config['seed'])
    output = Path(output_dir)
    if (output / 'training.jsonl').exists():
        raise FileExistsError(f'training run already exists at {output}')
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'config.json', config)
    started = time.monotonic()
    version = config.get('observation_version', 1)
    normalizer = Normalizer.fit(training_observations(n_slots, config['normalization_seeds'],
                                                    config.get('dynamic_training', False)), version=version)
    write_json(output / 'normalizer.json', normalizer.statistics)
    agent = SAC(config)
    replay = Replay(config['replay_capacity'], n_slots, config['seed'] + 1, version=version)
    rng = np.random.default_rng(config['seed'] + 2)
    environment = V2GEnv(n_slots=n_slots)
    dates = [f'2026-10-{day:02d}' for day in range(1, 15)]
    validation_pairs = grid('validation', config['validation_seeds'])
    best_cost, episode_costs, metrics = float('inf'), [], {}
    obs, episode_cost, episodes = None, 0., 0

    def checkpoint(step):
        return {'format_version': version, 'encoder': agent.encoder.state_dict(),
                'actor': agent.actor.state_dict(), 'normalizer': normalizer.statistics,
                'metadata': {'n_slots_trained': n_slots, 'step': step, 'config': config,
                             'normalization_split': 'train', 'training_dates': dates,
                             'training_seed_range': [0, 99], 'validation_grid': validation_pairs,
                             'torch_version': str(torch.__version__)}}

    try:
        with (output / 'training.jsonl').open('w', buffering=1) as log:
            for step in range(1, config['steps'] + 1):
                if obs is None:
                    date = str(rng.choice(dates))
                    seed = int(rng.integers(0, 100))
                    if config.get('dynamic_training', False):
                        mode = str(rng.choice(['standard', 'thinned', 'mixed'], p=[.5, .25, .25]))
                        scenario = training_scenario(date, seed, n_slots, mode)
                        raw_obs, _ = environment.reset(seed=seed, options={'scenario': scenario})
                    else:
                        raw_obs, _ = environment.reset(seed=seed, options={'date': date})
                    meter = BuildingMeter()
                    obs = normalizer.transform(raw_obs)
                if step <= config['warmup_steps']:
                    action = rng.uniform(-1., 1., n_slots + 1).astype(np.float32)
                    action[1:] *= obs['ev_present']
                else:
                    with torch.no_grad():
                        tensors = {key: torch.from_numpy(value).unsqueeze(0) for key, value in obs.items()}
                        action = agent.actor(agent.encoder(tensors))[0][0].numpy()
                external = environment_action(action, raw_obs) if config.get('action_mapping') == 'zero_power' else action
                next_raw, reward, done, _, info = environment.step(external)
                learning_reward = reward
                if config.get('center_building_reward', False):
                    net = info['p_grid_kw'] - info['p_bess_kw'] - float(info['p_ev_kw'].sum())
                    learning_reward += meter.step(net, info['sell_price'])['step_cost_sgd']
                next_obs = normalizer.transform(next_raw)
                replay.add(obs, action, learning_reward * config['reward_scale'], next_obs, done)
                obs, raw_obs = next_obs, next_raw
                episode_cost -= reward
                if done:
                    episodes += 1
                    episode_costs.append(episode_cost)
                    episode_cost, obs = 0., None
                if step > config['warmup_steps'] and step % config['update_every'] == 0:
                    metrics = agent.update(replay.sample(config['batch_size']))
                if step % config['log_every'] == 0:
                    record = dict(step=step, episodes=episodes, updates=agent.updates,
                                  elapsed_seconds=time.monotonic() - started,
                                  train_cost_recent=(float(np.mean(episode_costs[-50:])) if episode_costs else None), **metrics)
                    log.write(json.dumps(record, allow_nan=False) + '\n')
                    print(json.dumps(record), flush=True)
                if step % config['validation_every'] == 0 or step == config['steps']:
                    state = checkpoint(step)
                    policy = Policy(state, n_slots)
                    summaries = evaluate_grid(policy, n_slots, validation_pairs)
                    cost = float(np.mean([row['variable_cost_sgd'] for row in summaries]))
                    record = dict(step=step, validation_cost=cost, elapsed_seconds=time.monotonic() - started)
                    log.write(json.dumps(record) + '\n')
                    print(json.dumps(record), flush=True)
                    save_checkpoint(output / f'step_{step}.pt', state)
                    if cost < best_cost:
                        best_cost = cost
                        state['metadata']['validation_cost'] = cost
                        save_checkpoint(output / 'best.pt', state)
                        write_json(output / 'best_validation.json', summaries)
            save_checkpoint(output / 'last.pt', checkpoint(config['steps']))
            write_json(output / 'completed.json', {'best_validation_cost': best_cost,
                       'steps': config['steps'], 'elapsed_seconds': time.monotonic() - started})
    finally:
        environment.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', dest='config_path', default='sac/configs/default.json')
    parser.add_argument('--n-slots', type=int, choices=range(1, 11), required=True)
    parser.add_argument('--output-dir', required=True)
    train(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()
