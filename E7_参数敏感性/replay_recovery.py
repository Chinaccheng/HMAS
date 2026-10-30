"""Replay only E7 S1/S3 medium node damage recovery using the original routines.

The six-event damage prefix is reconstructed solely to define the initial
recovery state and integrate its unchanged contribution. No identification or
robustness experiment is run. Formal summaries/figures are not written here.
"""
from pathlib import Path
import os
import sys
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
sys.dont_write_bytecode = True
import argparse
import csv
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from UAV import experiment_common as common
from UAV.experiment_config import manifest, config

RESULT = PROJECT / 'UAV/E7_参数敏感性/result'
OUT = RESULT / 'recovery_replay_T120'
METHODS = ['random', 'degree', 'CMLC', 'SCFC', 'greedy']
OLD_T = {30: 50, 120: 150}


def read(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def worker(n, index):
    started = time.perf_counter()
    prior = next(row for row in read(RESULT / 'scenario_manifest.csv')
                 if row['factor'] == 'scale' and int(row['n']) == n and int(row['instance']) == index)
    model = common.model_for(n, index)
    generated = manifest(model, index)
    for field in ['scene_seed', 'search_seed', 'network_sha256', 'tasks_sha256']:
        assert str(generated[field]) == prior[field], (n, index, field)
    draws, base = common.construction(model, index, 300)
    assert abs(base.F - float(prior['F0'])) < 1e-14
    # The original scale_worker used these exact scores and damage protocol.
    scores = common.recovery_scores(model, base)
    state, events, protocol = common.damage_path(
        model, index, 'node', 'SCFC', .2, scores, draws, base)
    assert protocol['t_r'] == 14 and protocol['node_damage'] == n // 5
    runs = []
    for method in METHODS:
        for repeat in range(20 if method == 'random' else 1):
            log = common.recover_full(model, index, state, method, scores, draws, base, repeat)
            last_action_time = protocol['t_r'] + len(log)
            assert last_action_time < 50
            old_metrics, old_curve = common.recovery_metrics(base, events, protocol, log, .5, OLD_T[n])
            new_metrics, new_curve = common.recovery_metrics(base, events, protocol, log, .5, 120)
            assert old_metrics['R_res'] == new_metrics['R_res']
            assert old_metrics['Phi_final'] == new_metrics['Phi_final']
            phi = old_metrics['Phi_final']
            converted = (old_metrics['R_F'] * (OLD_T[n] - 5) + phi * (120 - OLD_T[n])) / 115
            assert abs(converted - new_metrics['R_F']) < 1e-14
            assert np.all(old_curve[last_action_time:] == old_curve[last_action_time])
            assert np.all(new_curve[last_action_time:] == new_curve[last_action_time])
            runs.append({'method': method, 'repeat': repeat, 'last_action_time': last_action_time,
                         'actions': [{**action, 'time': protocol['t_r'] + action['step']} for action in log],
                         'old_T': OLD_T[n], 'new_T': 120,
                         'metrics_old_T': old_metrics, 'metrics_T120': new_metrics,
                         'R_F_formula_T120': converted,
                         'curve_old_T': {'time': list(range(OLD_T[n] + 1)), 'Phi': old_curve},
                         'curve_T120': {'time': list(range(121)), 'Phi': new_curve}})
    original = [row for row in read(RESULT / 'scale_instance.csv')
                if int(row['n']) == n and int(row['instance']) == index and row['method'] in METHODS]
    assert len(original) == 5
    validation = []
    for row in original:
        selected = [run for run in runs if run['method'] == row['method']]
        new_rf = float(np.mean([run['metrics_T120']['R_F'] for run in selected]))
        error = new_rf - float(row['R_F'])
        assert abs(error) < 1e-12, (n, index, row['method'], error)
        validation.append({'method': row['method'], 'saved_R_F': row['R_F'],
                           'replayed_T120_R_F': new_rf, 'difference': error,
                           'repeat_count': len(selected)})
    result = {'n': n, 'instance': index, 'manifest': generated, 'baseline_F': base.F,
              'config': model.config.to_dict(), 'damage_protocol': protocol, 'damage_events': events,
              'runs': runs, 'reproduction_checks': validation, 'saved_T120_RF_reproduced': True,
              'old_R_res_Phi_final_reference_available': False,
              'seconds': time.perf_counter() - started,
              'source_hashes': {name: hashlib.sha256((PROJECT / 'UAV' / name).read_bytes()).hexdigest()
                                for name in ['model.py', 'experiment_config.py', 'experiment_common.py']}}
    target = OUT / f'S{1 if n == 30 else 3}_instance_{index:02d}.json'
    common.write_json(target, result)
    return {'n': n, 'instance': index, 'seconds': result['seconds'],
            'max_reproduction_error': max(abs(row['difference']) for row in validation)}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--pilot', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    assert all(config(n).horizon == 120 for n in [30, 60, 120])
    jobs = [(n, index) for n in [30, 120] for index in (range(1) if args.pilot else range(20))]
    jobs = [(n, index) for n, index in jobs if not (OUT / f'S{1 if n == 30 else 3}_instance_{index:02d}.json').exists()]
    started = time.perf_counter()
    print(f'Recovery-only replay: {len(jobs)} instances pending; workers={args.workers}', flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(worker, n, index): (n, index) for n, index in jobs}
        try:
            for completed, future in enumerate(as_completed(futures), 1):
                record = future.result()
                print(json.dumps({'completed': completed, 'pending_total': len(jobs),
                                  'elapsed': time.perf_counter() - started, **record}), flush=True)
        except BaseException:
            for future in futures:
                future.cancel()
            raise


if __name__ == '__main__':
    main()
