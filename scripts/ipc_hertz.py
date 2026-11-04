#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy==2.2.6", "gmsh==4.13.1"]
# ///
"""Generate straight-sided T10 quarter hemispheres; run the native IPC solver.
Reports Hertz force and contact-radius errors; does not assert continuum accuracy
from coarse geometry. Mesh, approach and regularization must all be converged.
"""
import argparse
import csv
import json
import pathlib
import subprocess
import gmsh
import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--levels', default='1,2')
p.add_argument('--steps', type=int, default=4)
p.add_argument('--out', default='output/logs/ipc_hertz')
a = p.parse_args()
root = pathlib.Path(__file__).resolve().parents[1]
out = root / a.out
out.mkdir(parents=True, exist_ok=True)
results = []
for level in map(int, a.levels.split(',')):
    if level < 1: raise ValueError('levels must be positive')
    folder = out / f'level_{level}'
    folder.mkdir(exist_ok=True)
    gmsh.initialize()
    gmsh.option.setNumber('General.Terminal', 0)
    gmsh.model.add('quarter_hemisphere')
    occ = gmsh.model.occ
    sphere = occ.addSphere(0, 0, 0, .15)
    box = occ.addBox(0, 0, -.15, .15, .15, .15)
    occ.intersect([(3, sphere)], [(3, box)])
    occ.synchronize()
    field = gmsh.model.mesh.field.add('MathEval')
    gmsh.model.mesh.field.setString(field, 'F',
        f'min(0.04, {0.012/level} + 0.5*max(0, sqrt(x*x+y*y+(z+0.15)^2)-0.016))')
    gmsh.model.mesh.field.setAsBackgroundMesh(field)
    for option in ['MeshSizeExtendFromBoundary', 'MeshSizeFromPoints', 'MeshSizeFromCurvature']:
        gmsh.option.setNumber('Mesh.' + option, 0)
    gmsh.option.setNumber('Mesh.ElementOrder', 2)
    gmsh.option.setNumber('Mesh.SecondOrderLinear', 1)
    gmsh.model.mesh.generate(3)
    tags, coords, _ = gmsh.model.mesh.getNodes()
    _, ttags = gmsh.model.mesh.getElementsByType(11)
    gmsh.finalize()
    X = coords.reshape(-1, 3)
    lookup = {tag: i for i, tag in enumerate(tags)}
    T = np.array([lookup[t] for t in ttags]).reshape(-1, 10)
    used = np.unique(T)
    remap = np.full(len(X), -1, dtype=int); remap[used] = np.arange(len(used))
    X, T = X[used], remap[T]
    corners = X[T[:, :4]]
    negative = np.linalg.det(corners[:, 1:] - corners[:, :1]) < 0
    T[negative, 1], T[negative, 2] = T[negative, 2].copy(), T[negative, 1].copy()
    canonical = T.copy()
    for edge, (i, j) in enumerate([(0,1),(1,2),(0,2),(0,3),(1,3),(2,3)]):
        middle = (X[T[:, i]] + X[T[:, j]])/2
        slot = np.linalg.norm(X[T[:, 4:]] - middle[:, None], axis=2).argmin(axis=1)
        canonical[:, 4+edge] = T[np.arange(len(T)), 4+slot]
    X[:, 2] += .1501
    mesh = folder / 'mesh.txt'
    with mesh.open('w') as stream:
        stream.write(f'{len(X)} {len(T)}\n')
        np.savetxt(stream, X, fmt='%.17g'); np.savetxt(stream, canonical, fmt='%d')
    command = [str(root/'bazel-bin/lib_bin/ipc/hertz_validation'),
               f'--mesh={mesh}', f'--out={folder}', f'--steps={a.steps}']
    (folder/'command.json').write_text(json.dumps(command, indent=2)+'\n')
    print(f'Level {level}: {len(X)} nodes, {len(T)} T10 elements', flush=True)
    with (folder/'run.log').open('w') as log:
        process = subprocess.Popen(command, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            print(line, end='', flush=True); log.write(line); log.flush()
        if process.wait(): raise RuntimeError('Native Hertz solve failed')
    with (folder/'hertz.csv').open() as stream: rows = list(csv.DictReader(stream))
    last = {k: float(v) for k, v in rows[-1].items()}
    results.append(dict(level=level, nodes=len(X), elements=len(T), **last,
                        force_relative_error=last['force']/last['force_hertz']-1,
                        radius_relative_error=last['a_effective']/last['a_hertz']-1))
    (out/'summary.json').write_text(json.dumps(results, indent=2)+'\n')
print(json.dumps(results, indent=2))
