#!/usr/bin/env python3
"""Physical checks for the coarse, finite-conductivity Hunt example.

Allrun exports fluid.csv (homplane:planeid=0) and electric.csv first. The raw
GLL samples retain element-local currents at interfaces. Only the Python
standard library is required. This is not an exact Hunt-profile comparison.
"""
import csv
import math
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET


def read_csv(path):
    with path.open() as stream:
        names = stream.readline().lstrip("# ").strip().split(",")
        rows = [dict(zip(names, map(float, row))) for row in csv.reader(stream)]
    assert rows and all(math.isfinite(v) for row in rows for v in row.values()), path
    return rows


def legendre(n, x):
    a, b = 1.0, x
    for k in range(2, n + 1):
        a, b = b, ((2*k-1)*x*b-(k-1)*a)/k
    return b if n else a


def integral(rows, mesh, function):
    # FieldConvert writes a tensor GLL block per quad; its output point count
    # can differ from the session's NUMPOINTS setting.
    cells = len(ET.parse(mesh).findall("./GEOMETRY/ELEMENT/Q"))
    n = math.isqrt(len(rows)//cells)
    assert n >= 2 and len(rows) == cells*n*n, "Expected quad GLL samples"
    total = 0.0
    for offset in range(0, len(rows), n*n):
        block = rows[offset:offset+n*n]
        x0, x1 = min(r["x"] for r in block), max(r["x"] for r in block)
        y0, y1 = min(r["y"] for r in block), max(r["y"] for r in block)
        assert x1 > x0 and y1 > y0
        for row in block:
            sx = 2*(row["x"]-x0)/(x1-x0)-1
            sy = 2*(row["y"]-y0)/(y1-y0)-1
            wx = (x1-x0)/(n*(n-1)*legendre(n-1, sx)**2)
            wy = (y1-y0)/(n*(n-1)*legendre(n-1, sy)**2)
            total += function(row)*wx*wy
    return total


def check(log):
    root = log.resolve().parent
    text = log.read_text()
    assert "Quasi-3D:" in text and "Homogeneous in z-direction" in text
    assert "Num. Hom. Modes (z): 2" in text
    assert text.count("layout validated once") == 2
    params = {}
    for node in ET.parse(root/"fluid.xml").findall("./CONDITIONS/PARAMETERS/P"):
        key, value = node.text.split("=", 1)
        params[key.strip()] = value.strip()
    solves = re.search(r"Electric potential solves:\s*(\d+)", text)
    assert solves and int(solves[1]) >= int(params["NumSteps"])
    fluid, electric = read_csv(root/"fluid.csv"), read_csv(root/"electric.csv")
    assert max(abs(r[v]) for r in fluid for v in ("u", "v")) < 1e-9
    assert max(abs(r["w"]) for r in fluid
               if r["x"] in (0, 1) or r["y"] in (0, 1)) < 1e-8
    wmax = max(r["w"] for r in fluid)
    assert wmax > 0 and min(r["w"] for r in fluid) > -1e-8
    area = integral(fluid, root/"fluid-mesh.xml", lambda r: 1)
    mean_w = integral(fluid, root/"fluid-mesh.xml", lambda r: r["w"])/area
    target = float(params["Flowrate"])
    assert abs(mean_w-target) < 2e-6*max(1, target), mean_w
    sigma = float(params["ElectricConductivity"])
    by = float(params["Ha"])/float(params["HalfWidth"])*math.sqrt(
        float(params["Kinvis"])/sigma)
    velocity = {(r["x"], r["y"]): r["w"] for r in fluid}
    potential = {(r["x"], r["y"]): r["phi"] for r in fluid}
    assert max(abs(r["phi"]-potential[(r["x"], r["y"])])
               for r in electric if r["sigma"] == sigma) < 1e-6
    power = integral(electric, root/"electric-mesh.xml", lambda r:
                     by*velocity[(r["x"], r["y"])]*r["Jx"]
                     if r["sigma"] == sigma else 0)
    joule = integral(electric, root/"electric-mesh.xml", lambda r:
                     sum(r[k]**2 for k in ("Jx", "Jy", "Jz"))/r["sigma"])
    assert power < -1e-8 and joule > 1e-8, (power, joule)
    balance = abs(power+joule)/joule
    # CSV output has six significant digits, so allow its rounding error.
    assert balance < 1e-4, balance
    solid_current = max(math.hypot(r["Jx"], r["Jy"])
                        for r in electric if r["sigma"] != sigma)
    assert solid_current > 1e-6, "No current entered the conducting walls"
    pressure = root/"partitions/fluid.prs"
    if pressure.exists():
        values = [float(line.split()[2]) for line in pressure.read_text().splitlines()
                  if line.strip() and not line.startswith("#")]
        assert len(values) >= 2 and abs(values[-1]-values[-2]) < 1e-7
    print(f"PASS Hunt Ha={params['Ha']}: {solves[1]} electrical solves; "
          f"mean w={mean_w:.9g}, max w={wmax:.9g}; "
          f"Lorentz/Joule balance={balance:.3e}")


if __name__ == "__main__":
    check(Path(sys.argv[1]))
