#!/usr/bin/env python3
"""Check both solvers against the nonzero-current analytical solution."""

import math
from pathlib import Path
import re
import sys


text = Path(sys.argv[1]).read_text()
errors = re.findall(r"L 2 error\s*\(variable\s+(\w+)\)\s*:\s*(\S+)", text)
assert sorted(var for var, _ in errors) == ["p", "phi", "phi", "u", "v"], errors
for variable, value in errors:
    error = float(value)
    assert math.isfinite(error) and error < 1e-8, (variable, error)
solves = re.search(r"Electric potential solves:\s*(\d+)", text)
assert solves and int(solves[1]) >= 5, "Missing or incomplete electrical exchange loop"
print(f"PASS: {solves[1]} electrical solves; maximum analytical L2 error "
      f"{max(float(value) for _, value in errors):.3e}")
