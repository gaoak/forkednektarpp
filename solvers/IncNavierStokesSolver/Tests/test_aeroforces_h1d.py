#!/usr/bin/env python3
"""Exact spanwise wall traction: w=y(1-y), v=sin(z), nu=1, lower wall.
At t=0, F3_visc(z)=1+cos(z), mean=1, and F3_press=0.
The cosine term tests dv/dz, which a slice-only derivative misses.
"""
import argparse
import math
import os
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET
p = argparse.ArgumentParser()
p.add_argument('--solver', required=True)
p.add_argument('--case', type=Path, required=True)
p.add_argument('--work', type=Path, required=True)
p.add_argument('--mpiexec', default='mpiexec')
a = p.parse_args()
for (ranks, npz, mhd) in ((1, 1, False), (2, 1, False), (2, 2, False), (1, 1, True)):
    root = a.work / f'ranks{ranks}-npz{npz}-mhd{int(mhd)}'
    root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(a.case, root / 'case.xml')
    if mhd:
        tree = ET.parse(root / 'case.xml')
        doc = tree.getroot()
        conditions = doc.find('CONDITIONS')
        variables = conditions.find('VARIABLES')
        variables[-1].set('ID', '4')
        phi = ET.Element('V', ID='3')
        phi.text = 'phi'
        variables.insert(3, phi)
        for e in doc.findall('./EXPANSIONS/E'):
            if e.get('FIELDS') == 'p':
                e.set('FIELDS', 'phi,p')
        ET.SubElement(conditions.find('PARAMETERS'), 'P').text = ' ElectricConductivity = 1 '
        for region in conditions.findall('BOUNDARYCONDITIONS/REGION'):
            if region.get('REF') in ('1', '2'):
                ET.SubElement(region, 'P', VAR='phi', VALUE='[2]' if region.get('REF') == '1' else '[1]')
            else:
                ET.SubElement(region, 'N', VAR='phi', USERDEFINEDTYPE='Insulator', VALUE='0')
        for f in conditions.findall('FUNCTION'):
            if f.get('NAME') in ('InitialConditions', 'ExactSolution'):
                ET.SubElement(f, 'E', VAR='phi', VALUE='0')
            if f.get('NAME') == 'BodyForce':
                for e in f:
                    if e.get('VAR') == 'w':
                        e.set('VALUE', e.get('VALUE') + '+y*(1-y)')
        f = ET.SubElement(conditions, 'FUNCTION', NAME='ElectricMagneticFields')
        ET.SubElement(f, 'E', VAR='By', VALUE='1')
        force = ET.SubElement(doc.find('FORCING'), 'FORCE', TYPE='ForcingLorentz')
        ET.SubElement(force, 'ELECTRICMAGNETICFIELDS').text = 'ElectricMagneticFields'
        tree.write(root / 'case.xml', encoding='utf-8', xml_declaration=True)
    cmd = [a.mpiexec, '-np', str(ranks), a.solver, f'--npz={npz}', 'case.xml']
    with (root / 'log.solver').open('w') as log:
        subprocess.run(cmd, cwd=root, env=dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1'), stdout=log, stderr=subprocess.STDOUT, check=True, timeout=60)
    for name in ('mean', 'planes'):
        text = (root / (name + '.fce')).read_text()
        assert 'F3-press' in text and 'F3-visc' in text and ('F3-total' in text), text
        records = [line.split() for line in text.splitlines() if line.strip() and (not line.startswith('#'))]
        records = [row for row in records if float(row[0]) == 0]
        expected_count = 1 if name == 'mean' else 9
        assert len(records) == expected_count, (name, records)
        for row in records:
            expected = 1.0
            if name == 'planes' and row[-1] != 'average':
                expected += math.cos(2 * math.pi * int(row[-1]) / 8)
            assert abs(float(row[7])) < 1e-10, row
            assert abs(float(row[8]) - expected) < 1e-07, (row, expected)
            assert abs(float(row[9]) - expected) < 1e-07, (row, expected)
    with (root / 'force-z.dat').open('w') as stream:
        stream.write('VARIABLES="z","Fz_pressure","Fz_viscous","Fz_total","Fz_exact"\n')
        stream.write('ZONE I=8, DATAPACKING=POINT\n')
        for row in records:
            if row[-1] == 'average':
                continue
            z = 2 * math.pi * int(row[-1]) / 8
            stream.write(f'{z:.16e} {float(row[7]):.16e} {float(row[8]):.16e} {float(row[9]):.16e} {1 + math.cos(z):.16e}\n')
    print(f'PASS {ranks} ranks, npz={npz}, MHD={mhd}: mean and per-plane axial traction', flush=True)
