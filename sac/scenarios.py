"""Deterministic training mixtures and the separately held-out realistic suite."""
from dataclasses import replace
from datetime import date as Date

import numpy as np

from env import load_params
from env.physics import BatterySpec, feasible_interval
from env.scenario import Vehicle, generate_scenario


LADDER_COUNTS = (0, 2, 4, 6, 8, 10)


def extended_scenario(date, seed, family='mixed', count=None, n_slots=10):
    params = load_params()
    base = generate_scenario(date, seed, n_slots, params)
    family_id = 31 if family == 'mixed' else 32
    rng = np.random.default_rng(np.random.SeedSequence([seed, Date.fromisoformat(date).toordinal(), family_id]))
    vehicles = []
    spec = BatterySpec.from_params(params['ev'])
    target = spec.capacity_kwh * params['ev']['departure_target_soc']

    def vehicle(arrival, departure):
        initial = float(rng.uniform(.3, .7) * spec.capacity_kwh)
        feasible_interval(initial, target, departure - arrival, spec)
        return Vehicle(arrival, departure, initial, target)

    if family == 'ladder':
        if count not in range(n_slots + 1):
            raise ValueError('ladder count must fit the physical slots')
        for _ in range(count):
            vehicles.append(vehicle(int(rng.integers(15, 20)), int(rng.integers(34, 38))))
    elif family == 'mixed':
        if Date.fromisoformat(date).weekday() < 5:
            nights = min(int(rng.integers(1, 3)), n_slots)
            maximum = min(8, n_slots - nights)
            daytime = int(rng.integers(min(3, maximum), maximum + 1))
            for _ in range(daytime):
                vehicles.append(vehicle(int(rng.integers(15, 20)), int(rng.integers(34, 38))))
            types = [0, 1] if nights == 2 else [int(rng.integers(2))]
            for kind in types:
                vehicles.append(vehicle(int(rng.integers(0, 2)), 17) if kind == 0 else vehicle(28, 47))
        else:
            for index in range(min(int(rng.integers(3)), n_slots)):
                if index == 0:
                    vehicles.append(vehicle(int(rng.integers(0, 2)), 17))
                else:
                    vehicles.append(vehicle(28, 47))
    else:
        raise ValueError('family must be mixed or ladder')
    slots = [None] * n_slots
    for slot, car in zip(rng.permutation(n_slots), vehicles):
        slots[int(slot)] = car
    return replace(base, vehicles=tuple(slots))


def training_scenario(date, seed, n_slots, mode):
    if mode == 'mixed':
        return extended_scenario(date, seed, n_slots=n_slots)
    base = generate_scenario(date, seed, n_slots, load_params())
    if mode == 'thinned':
        rng = np.random.default_rng(np.random.SeedSequence([seed, Date.fromisoformat(date).toordinal(), 33]))
        keep = set(rng.choice(n_slots, size=int(rng.integers(n_slots + 1)), replace=False).tolist())
        return replace(base, vehicles=tuple(v if i in keep else None for i, v in enumerate(base.vehicles)))
    return base
