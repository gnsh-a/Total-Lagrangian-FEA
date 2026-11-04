#!/usr/bin/env python3
"""Run controlled discretization/regularization comparisons, never equate a
successful solve with mesh convergence. Build first with run_ipc.sh/validate_ipc.sh.
All cases use equal physical duration. Reports differences, feasibility and cost.
"""
import argparse
import csv
import json
import math
import pathlib
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--duration', type=float, default=.02)
parser.add_argument('--out', default='output/logs/ipc_convergence')
parser.add_argument('--case', choices=['brick', 'sphere'], default='brick')
args = parser.parse_args()
root = pathlib.Path(__file__).resolve().parents[1]
out = root / args.out
out.mkdir(parents=True, exist_ok=True)
cases = [('baseline', .001, 1, .003, 1000),
         ('dt_half', .0005, 1, .003, 1000),
         ('mesh_refined', .001, 2, .003, 1000),
         ('dhat_half', .001, 1, .0015, 1000),
         ('stiffness_double', .001, 1, .003, 2000)]
rows = []
for name, dt, resolution, dhat, stiffness in cases:
    steps = round(args.duration / dt)
    if steps < 1 or not math.isclose(steps * dt, args.duration):
        raise ValueError('Duration must be a positive multiple of every time step')
    folder = out / name
    folder.mkdir(exist_ok=True)
    command = [str(root/'bazel-bin/lib_bin/ipc/contact_demo'), f'--case={args.case}',
               f'--steps={steps}', f'--dt={dt}', f'--resolution={resolution}',
               f'--dhat={dhat}', f'--stiffness={stiffness}', '--mu=0', '--slope=0',
               '--gap=.001', f'--out={folder}']
    if args.case == 'sphere' and name == 'mesh_refined':
        command.append('--sphere_resolution=med')
    print('Running', name, flush=True)
    (folder/'command.json').write_text(json.dumps(command, indent=2)+'\n')
    with (folder/'run.log').open('w') as log:
        subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
    with (folder/'steps.csv').open() as stream:
        history = list(csv.DictReader(stream))
    if len(history) != steps:
        raise RuntimeError(f'{name}: incomplete run')
    for step in history:
        if not float(step['min_J']) > 0 or float(step['active_min_distance']) <= 0:
            raise RuntimeError(f'{name}: invalid accepted state')
    last = history[-1]
    rows.append(dict(case=name, dt=dt, resolution=resolution, dhat=dhat,
                     stiffness=stiffness, body_z=float(last['body_z']),
                     contact_z=float(last['body_contact_z']),
                     displacement_z=float(last['body_displacement_z']),
                     kinetic=float(last['kinetic']), min_J=min(float(r['min_J']) for r in history),
                     seconds=sum(float(r['seconds']) for r in history),
                     retries=sum(int(r['retries']) for r in history)))
    (out/'summary.json').write_text(json.dumps(rows, indent=2)+'\n')
reference = rows[0]
for row in rows:
    row['displacement_z_change'] = row['displacement_z']-reference['displacement_z']
    row['contact_z_change'] = row['contact_z']-reference['contact_z']
(out/'summary.json').write_text(json.dumps(rows, indent=2)+'\n')
with (out/'summary.csv').open('w') as stream:
    writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
    writer.writeheader(); writer.writerows(rows)
print(json.dumps(rows, indent=2))
