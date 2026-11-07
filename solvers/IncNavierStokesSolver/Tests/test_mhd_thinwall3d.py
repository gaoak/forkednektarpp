#!/usr/bin/env python3
"""Compare a periodic full-3D duct to its 3DH1D extrusion reference."""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess

parser = argparse.ArgumentParser()
for key in ('solver', 'fieldconvert', 'mpiexec'):
    parser.add_argument('--' + key, required=True)
for key in ('case', 'reference', 'work'):
    parser.add_argument('--' + key, type=Path, required=True)
args = parser.parse_args()
env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')

def run(command, folder, logname):
    with (folder / logname).open('w') as log:
        subprocess.run(command, cwd=folder, env=env, check=True,
                       stdout=log, stderr=subprocess.STDOUT, timeout=90)

def read(path):
    with path.open() as stream:
        names = stream.readline().lstrip('# ').strip().split(',')
        return [dict(zip(names, map(float, row))) for row in csv.reader(stream)]

reference = None
report = {}
for name, ranks, is_reference in [('reference', 1, True), ('3d-serial', 1, False),
                                   ('3d-mpi2', 2, False)]:
    folder = args.work / name
    folder.mkdir(parents=True, exist_ok=True)
    mesh = args.case.parent / 'MHDThinWall2D/mesh.xml' if is_reference else args.case / 'mesh.xml'
    session = args.reference if is_reference else args.case / 'session.xml'
    shutil.copy2(mesh, folder / 'mesh.xml')
    shutil.copy2(session, folder / 'session.xml')
    run([args.mpiexec, '-np', str(ranks), args.solver, 'mesh.xml', 'session.xml'],
        folder, 'log.solver')
    convert = [args.fieldconvert, '-f', '-n', '7']
    if is_reference:
        convert += ['-m', 'homplane:planeid=0']
    convert += ['mesh.xml', 'session.xml', 'mesh.fld']
    run(convert + ['result.csv'], folder, 'log.csv')
    run(convert + ['result.plt:plt:double'], folder, 'log.tecplot')
    rows = read(folder / 'result.csv')
    groups = {}
    for row in rows:
        key = round(row['x'], 8), round(row['y'], 8)
        groups.setdefault(key, []).append(row)
    if is_reference:
        reference = {key: group[0] for key, group in groups.items()}
        continue
    assert groups.keys() == reference.keys(), 'Mismatched cross-section sample grids'
    # Remove a single potential gauge offset; gradients and velocities are physical.
    key = min(groups)
    gauge = groups[key][0]['phi'] - reference[key]['phi']
    velocity_error = max(abs(row[v] - reference[key][v])
                         for key, group in groups.items() for row in group
                         for v in ('u', 'v', 'w'))
    phi_error = max(abs(row['phi'] - gauge - reference[key]['phi'])
                    for key, group in groups.items() for row in group)
    transverse = max(abs(row[v]) for row in rows for v in ('u', 'v'))
    assert velocity_error < 3e-6, (name, velocity_error)
    assert phi_error < 3e-6, (name, phi_error)
    assert transverse < 1e-8, (name, transverse)
    report[name] = dict(max_velocity_difference=velocity_error,
                        max_potential_difference_after_gauge=phi_error,
                        gauge_offset=gauge, max_transverse_velocity=transverse)
    print('PASS', name, report[name], flush=True)
(args.work / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
