#!/usr/bin/env python3
"""Analytical AeroForces regression; run with --solver /path/to/IncNavierStokesSolver.

Checks both transverse wall normals, Fourier derivatives, density/viscosity,
mean and per-plane output, direction projection and independence from Lz.
Uses initial stresses of a divergence-free polynomial/Fourier velocity field.
Pass --mapping for the identity coordinate mapping, --fft FFTW for FFTs,
or --launcher 'mpiexec -n 4' --npz 2 for a hybrid MPI decomposition.
"""
import argparse
import math
from pathlib import Path
import re
import shlex
import subprocess
import tempfile
import xml.etree.ElementTree as ET


def session(length, mapping, fft):
    root = ET.parse(Path(__file__).with_name('ChanFlow_3DH1D_FFT.xml')).getroot()
    root.find('EXPANSIONS/E').set('NUMMODES', '4')
    cond = root.find('CONDITIONS')
    for item in cond.findall('SOLVERINFO/I'):
        if item.get('PROPERTY') == 'USEFFT':
            item.set('VALUE', fft)
        if mapping and item.get('PROPERTY') == 'SolverType':
            item.set('VALUE', 'VCSMapping')
    cond.find('TIMEINTEGRATIONSCHEME/ORDER').text = '1'
    params = cond.find('PARAMETERS')
    params.clear()
    for name, value in dict(TimeStep=0.0001, NumSteps=1, IO_InfoSteps=1,
                            Kinvis=0.125, rho=2, HomModesZ=8, LZ=length).items():
        ET.SubElement(params, 'P').text = f'{name} = {value}'
    expressions = {'u': 'y*(1-y)+sin(2*PI*z/LZ)',
                   'v': 'sin(2*PI*z/LZ)',
                   'w': '3*y*(1-y)+2*x*(1-x)', 'p': '2'}
    for func in cond.findall('FUNCTION'):
        for value in func:
            value.set('VALUE', expressions[value.get('VAR')])
    for region in cond.findall('BOUNDARYCONDITIONS/REGION'):
        for bc in region:
            if bc.tag == 'D':
                bc.set('VALUE', expressions[bc.get('VAR')])
    # Separate lower and left boundaries; other boundaries are still in the mesh.
    composites = root.find('GEOMETRY/COMPOSITE')
    ET.SubElement(composites, 'C', ID='4').text = 'E[0,1]'
    ET.SubElement(composites, 'C', ID='5').text = 'E[10,11]'
    cond.find('BOUNDARYREGIONS/B').text = 'C[4]'
    ET.SubElement(cond.find('BOUNDARYREGIONS'), 'B', ID='3').text = 'C[5]'
    region = ET.SubElement(cond.find('BOUNDARYCONDITIONS'), 'REGION', REF='3')
    for var in ('u', 'v', 'w'):
        ET.SubElement(region, 'D', VAR=var, VALUE=expressions[var])
    ET.SubElement(region, 'N', VAR='p', USERDEFINEDTYPE='H', VALUE='0')
    if mapping:
        func = ET.SubElement(cond, 'FUNCTION', NAME='Mapping')
        ET.SubElement(func, 'E', VAR='x', VALUE='x')
        ET.SubElement(ET.SubElement(root, 'MAPPING', TYPE='XofXZ'), 'COORDS').text = 'Mapping'
    filters = ET.SubElement(root, 'FILTERS')
    for name, boundary, allplanes, project in [
        ('bottom_mean', 0, False, False), ('bottom_planes', 0, True, False),
        ('left_mean', 1, False, False), ('left_planes', 1, True, False),
        ('projected', 0, False, True),
    ]:
        filt = ET.SubElement(filters, 'FILTER', TYPE='AeroForces')
        options = dict(OutputFile=name, OutputFrequency='1',
                       Boundary=f'B[{boundary}]', OutputAllPlanes=str(allplanes))
        if project:
            options.update(Direction1='0 0 1', Direction3='-1 0 0')
        for key, value in options.items():
            ET.SubElement(filt, 'PARAM', NAME=key).text = value
    return ET.ElementTree(root)


def close(actual, expected, context):
    if not math.isclose(actual, expected, abs_tol=2e-7, rel_tol=2e-7):
        raise AssertionError(f'{context}: got {actual}, expected {expected}')


def check(directory, length):
    for name, base in [('bottom_mean', 0.75), ('bottom_planes', 0.75),
                       ('left_mean', 0.5), ('left_planes', 0.5), ('projected', 0.75)]:
        text = (directory / f'{name}.fce').read_text()
        assert 'F3-press' in text and 'F3-visc' in text and 'F3-total' in text
        rows = [line.split() for line in text.splitlines() if line and not line.startswith('#')]
        rows = [row for row in rows if float(row[0]) == 0]
        assert len(rows) == (9 if name.endswith('planes') else 1), (name, rows)
        for row in rows:
            phase = 0 if not name.endswith('planes') or row[-1] == 'average' else (
                0.25 * 2 * math.pi / length * math.cos(2 * math.pi * int(row[-1]) / 8))
            if name == 'projected':
                close(float(row[3]), base, name + ' projected F1')
                close(float(row[9]), -0.25, name + ' projected F3')
            else:
                close(float(row[7]), 0, name + ' F3 pressure')
                close(float(row[8]), base + phase, name + ' F3 viscous')
                close(float(row[9]), base + phase, name + ' F3 total')
                if name.startswith('bottom'):
                    close(float(row[3]), 0.25, name + ' F1 unchanged')
                    close(float(row[6]), -4, name + ' F2 unchanged')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--solver', required=True, type=Path)
    parser.add_argument('--launcher', default='')
    parser.add_argument('--npz', type=int, default=1)
    parser.add_argument('--mapping', action='store_true')
    parser.add_argument('--fft', choices=['FFTW', 'MVM'], default='MVM')
    args = parser.parse_args()
    for length in (2.0, 5.0):
        with tempfile.TemporaryDirectory(prefix='aeroforces-3dh1d-') as tmp:
            directory = Path(tmp)
            session(length, args.mapping, args.fft).write(directory / 'case.xml')
            command = shlex.split(args.launcher) + [str(args.solver.resolve()), 'case.xml']
            if args.npz != 1:
                command += ['--npz', str(args.npz)]
            result = subprocess.run(command, cwd=directory, capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError(result.stdout + result.stderr)
            try:
                check(directory, length)
            except Exception:
                print(result.stdout + result.stderr)
                for path in directory.glob('*.fce'):
                    print(path.name, path.read_text())
                raise
            # Taking Fourier derivatives and restoring wave space must not
            # change the subsequent time step compared with an unfiltered run.
            control = session(length, args.mapping, args.fft)
            control.getroot().remove(control.getroot().find('FILTERS'))
            control_dir = directory / 'control'
            control_dir.mkdir()
            control.write(control_dir / 'case.xml')
            baseline = subprocess.run(command, cwd=control_dir,
                                      capture_output=True, text=True)
            if baseline.returncode:
                raise RuntimeError(baseline.stdout + baseline.stderr)
            pattern = r'^L (?:2|inf\w*) error[^:]*:\s*(\S+)'
            actual = re.findall(pattern, result.stdout, re.MULTILINE)
            expected = re.findall(pattern, baseline.stdout, re.MULTILINE)
            assert len(actual) == len(expected) == 8
            for value, reference in zip(actual, expected):
                close(float(value), float(reference), 'filtered/control solution')
        print(f'PASS Lz={length}, mapping={args.mapping}, FFT={args.fft}, npz={args.npz}')


if __name__ == '__main__':
    main()
