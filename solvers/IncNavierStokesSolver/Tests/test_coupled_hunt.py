#!/usr/bin/env python3
"""Check 3DH1D zero-mode Hunt flow across unified MPI decompositions."""
import argparse
import csv
import os
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

parser = argparse.ArgumentParser()
for key in ("case", "build", "work"):
    parser.add_argument("--"+key, type=Path, required=True)
args = parser.parse_args()
args.work.mkdir(parents=True, exist_ok=True)


def run(m, n, work, case, reject=False):
    env = dict(os.environ, NEKTAR_BUILD_DIR=str(args.build),
               MHD_CASE_DIR=str(case), FLUID_RANKS=str(m), SOLID_RANKS=str(n),
               MHD_TRANSFER="Coefficients", MHD_CHECK_RESULTS="1", MHD_TIMEOUT="50")
    result = subprocess.run([str(args.case/"Allrun"), str(work)],
                            env=env, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, timeout=60)
    work.mkdir(exist_ok=True)
    (work/"test.log").write_text(result.stdout)
    if reject:
        assert result.returncode and "HomModesZ=2 zero mode" in result.stdout, result.stdout
    else:
        assert result.returncode == 0 and "PASS Hunt Ha=1" in result.stdout, result.stdout
    return result.stdout.strip()


def data(path, names):
    with path.open() as stream:
        fields = stream.readline().lstrip("# ").strip().split(",")
        rows = [dict(zip(fields, map(float, row))) for row in csv.reader(stream)]
    return {(row["x"], row["y"]): [row[name] for name in names] for row in rows}


reference = None
for m, n in ((1, 1), (2, 1), (2, 2)):
    work = args.work/f"m{m}-n{n}"
    print(run(m, n, work, args.case), flush=True)
    values = data(work/"fluid.csv", ("u", "v", "w"))
    if reference is None:
        reference = values
    else:
        assert reference.keys() == values.keys()
        assert max(abs(a-b) for key in reference
                   for a, b in zip(reference[key], values[key])) < 1e-7
print("PASS partition-independent axial velocity", flush=True)

bad = args.work/"unsupported-modes-input"
bad.mkdir(exist_ok=True)
for name in ("fluid-mesh.xml", "electric-mesh.xml", "fluid.xml", "electric.xml"):
    shutil.copyfile(args.case/name, bad/name)
tree = ET.parse(bad/"fluid.xml")
for param in tree.findall("./CONDITIONS/PARAMETERS/P"):
    if param.text.strip().startswith("HomModesZ ="):
        param.text = " HomModesZ = 4 "
tree.write(bad/"fluid.xml", encoding="utf-8", xml_declaration=True)
run(2, 1, args.work/"unsupported-modes", bad, reject=True)
print("PASS unsupported z-dependent Fourier layout rejected", flush=True)
