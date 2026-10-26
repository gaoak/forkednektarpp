#!/usr/bin/env python3
"""Integration checks for unified partitions and paired modal communication."""

import argparse
import csv
import math
from pathlib import Path
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET


parser = argparse.ArgumentParser()
for name in ("partitioner", "fluid", "electric", "transfer", "mpiexec", "case", "work"):
    parser.add_argument(f"--{name}", required=True)
args = parser.parse_args()
work = Path(args.work).resolve()
work.mkdir(parents=True, exist_ok=True)


def run(command, folder, log, failure=False):
    result = subprocess.run(command, cwd=folder, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=35)
    (folder / log).write_text(result.stdout)
    if failure:
        assert result.returncode != 0, result.stdout
    else:
        assert result.returncode == 0, result.stdout
    return result.stdout


def add_fixed_field(folder):
    # A constant electric field has a nontrivial weak load across conductivity
    # jumps and insulating side boundaries. Preserve the exact channel flow.
    for name in ("fluid.xml", "electric.xml"):
        path = folder / name
        tree = ET.parse(path)
        params = tree.find("./CONDITIONS/PARAMETERS")
        fields = (("ExternalEx", "0.02"), ("ExternalEy", "0.03"))
        for index, (key, value) in enumerate(fields):
            param = ET.Element("P")
            param.text = f" {key} = {value} "
            params.insert(index, param)
        for param in params:
            if param.text.strip().startswith("JyExact ="):
                param.text = (" JyExact = (ExternalEy*(1+2*WallThickness)"
                              "-Voltage-2*U0*B0/3)/Resistance ")
        for function in tree.findall("./CONDITIONS/FUNCTION"):
            if function.get("NAME") == "ElectricMagneticFields":
                for axis in ("x", "y"):
                    ET.SubElement(function, "E", VAR="E"+axis,
                                  VALUE="ExternalE"+axis)
            if function.get("NAME") in ("ExactSolution", "InitialConditions"):
                for expression in function:
                    if expression.get("VAR") == "phi":
                        expression.set("VALUE", expression.get("VALUE") +
                                       "+ExternalEx*x+ExternalEy*(y+WallThickness)")
        if name == "electric.xml":
            for boundary in tree.findall("./CONDITIONS/BOUNDARYCONDITIONS/REGION/D"):
                boundary.set("VALUE", boundary.get("VALUE") + "+ExternalEx*x")
        tree.write(path, encoding="utf-8", xml_declaration=True)


def prepare(folder, m, n, fixed_field=False):
    folder.mkdir(exist_ok=True)
    for name in ("fluid-mesh.xml", "fluid.xml", "electric-mesh.xml", "electric.xml"):
        shutil.copyfile(Path(args.case) / name, folder / name)
    if fixed_field:
        add_fixed_field(folder)
    run([args.partitioner, str(m), str(n), "parts", "fluid-mesh.xml", "fluid.xml",
         "electric-mesh.xml", "electric.xml"], folder, "partition.log")
    rows = list(csv.DictReader((folder / "parts/partition-map.csv").open()))
    assert len(rows) == 8 and len({r["element_id"] for r in rows}) == 8
    for row in rows:
        if row["region"] == "fluid":
            assert row["incns_rank"] == row["adr_rank"]
            assert 0 <= int(row["adr_rank"]) < m
        else:
            assert row["incns_rank"] == "-1" and m <= int(row["adr_rank"]) < m+n
    return folder


def command(folder, m, n, transfer=False, reverse=False):
    fluid = ["-np", str(m), args.transfer if transfer else args.fluid,
             "--cwipi", "fluid", str(folder / "parts/fluid_xml")]
    electric = ["-np", str(m+n), args.transfer if transfer else args.electric,
                "--cwipi", "electric", str(folder / "parts/electric_xml")]
    return [args.mpiexec] + (electric + [":"] + fluid if reverse else fluid + [":"] + electric)


def check_solution(text):
    errors = re.findall(r"L 2 error\s*\(variable\s+(\w+)\)\s*:\s*(\S+)", text)
    assert sorted(v for v, _ in errors) == ["p", "phi", "phi", "u", "v"], text
    assert all(math.isfinite(float(e)) and float(e) < 1e-8 for _, e in errors), text
    assert "Electric potential solves: 6" in text
    return max(float(e) for _, e in errors)


for m, n in ((1, 1), (2, 1), (2, 2)):
    folder = prepare(work / f"m{m}-n{n}", m, n)
    text = run(command(folder, m, n), folder, "coupled.log")
    error = check_solution(text)
    assert text.count("layout validated once") == 2
    print(f"PASS M={m}, N={n}: maximum analytical L2 error {error:.3e}", flush=True)

folder = prepare(work / "fixed-field", 2, 1, fixed_field=True)
error = check_solution(run(command(folder, 2, 1), folder, "coupled.log"))
print(f"PASS cached nonzero fixed field: maximum analytical L2 error {error:.3e}", flush=True)
# Independently partitioned point-mode meshes exercise mixed fluid/solid ranks.
point_command = [args.mpiexec, "-np", "2", args.fluid, "--cwipi", "fluid",
                 "fluid-mesh.xml", "fluid.xml", ":", "-np", "2", args.electric,
                 "--cwipi", "electric", "electric-mesh.xml", "electric.xml"]
error = check_solution(run(point_command, folder, "point.log"))
print(f"PASS nonzero fixed field with point transfer: maximum analytical L2 error {error:.3e}", flush=True)

folder = work / "m2-n1"
text = run(command(folder, 2, 1, transfer=True, reverse=True), folder, "polynomials.log")
errors = re.findall(r"Coefficient polynomial transfer error:\s*(\S+)", text)
assert len(errors) == 2 and all(float(e) < 1e-10 for e in errors), text
print("PASS asymmetric x/y polynomial transfer, including reversed application order", flush=True)

# Reject equal-sized partitions whose physical geometry differs. A collective
# startup check must fail before any time-step message is exchanged.
folder = prepare(work / "bad-geometry", 2, 1)
for path in (folder / "parts/electric_xml").glob("*.xml"):
    tree = ET.parse(path)
    for vertex in tree.findall("./GEOMETRY/VERTEX/V"):
        xyz = [float(v) for v in vertex.text.split()]
        xyz[0] += 0.01
        vertex.text = " ".join(map(str, xyz))
    tree.write(path, encoding="utf-8", xml_declaration=True)
text = run(command(folder, 2, 1), folder, "rejected.log", failure=True)
assert "Paired fluid layouts differ" in text, text
print("PASS incompatible geometry rejected at startup", flush=True)
