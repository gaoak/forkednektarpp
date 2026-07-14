#!/usr/bin/env python3

import argparse
import pathlib
import re
import subprocess
import tempfile


VELOCITY_VARS = ["u", "v"]
DEFAULT_METHOD_BY_ORDER = {2: "IMEXdirk23", 3: "IMEXdirk23"}
DEFAULT_STEPS_BY_ORDER = {2: 4, 3: 4}


class Poly:
    def __init__(self, terms=None):
        self.terms = {k: v for k, v in (terms or {}).items() if abs(v) > 1e-14}

    @staticmethod
    def const(value):
        return Poly({(0, 0): float(value)})

    @staticmethod
    def var(axis):
        exp = [0, 0]
        exp[axis] = 1
        return Poly({tuple(exp): 1.0})

    def __add__(self, other):
        other = as_poly(other)
        terms = dict(self.terms)
        for exp, coef in other.terms.items():
            terms[exp] = terms.get(exp, 0.0) + coef
        return Poly(terms)

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        return Poly({exp: -coef for exp, coef in self.terms.items()})

    def __sub__(self, other):
        return self + (-as_poly(other))

    def __rsub__(self, other):
        return as_poly(other) - self

    def __mul__(self, other):
        other = as_poly(other)
        terms = {}
        for exp1, coef1 in self.terms.items():
            for exp2, coef2 in other.terms.items():
                exp = tuple(exp1[i] + exp2[i] for i in range(2))
                terms[exp] = terms.get(exp, 0.0) + coef1 * coef2
        return Poly(terms)

    def __rmul__(self, other):
        return self * other

    def __pow__(self, power):
        out = Poly.const(1.0)
        for _ in range(power):
            out = out * self
        return out

    def deriv(self, axis):
        terms = {}
        for exp, coef in self.terms.items():
            if exp[axis] == 0:
                continue
            new_exp = list(exp)
            new_exp[axis] -= 1
            new_exp = tuple(new_exp)
            terms[new_exp] = terms.get(new_exp, 0.0) + coef * exp[axis]
        return Poly(terms)

    def lap(self):
        return self.deriv(0).deriv(0) + self.deriv(1).deriv(1)

    def expr(self):
        if not self.terms:
            return "0"
        parts = []
        names = ["x", "y"]
        for exp, coef in sorted(self.terms.items()):
            factors = []
            abs_coef = abs(coef)
            if exp == (0, 0) or abs(abs_coef - 1.0) > 1e-14:
                factors.append(f"{abs_coef:.16g}")
            for axis, power in enumerate(exp):
                factors.extend([names[axis]] * power)
            term = "*".join(factors) if factors else "1"
            parts.append((coef < 0.0, term))

        expr = ""
        for is_negative, term in parts:
            if not expr:
                expr = f"-{term}" if is_negative else term
            else:
                expr += f" - {term}" if is_negative else f" + {term}"
        return f"({expr})"


def as_poly(value):
    return value if isinstance(value, Poly) else Poly.const(value)


x = Poly.var(0)
y = Poly.var(1)

# Domain is [0,3] x [0,3]. This stream-function gives a non-trivial
# divergence-free velocity field with homogeneous velocity boundary values.
X = x * (3.0 - x)
Y = y * (3.0 - y)
psi = (X**2) * (Y**2)
u0 = psi.deriv(1)
v0 = -psi.deriv(0)
p0 = (x - 1.5) * (y - 1.5)


def adv(component):
    return u0 * component.deriv(0) + v0 * component.deriv(1)


A = "(1+0.1*sin(t))"
AT = "(0.1*cos(t))"
A0 = "1"

U0 = u0.expr()
V0 = v0.expr()
P0 = p0.expr()
U = f"{A}*{U0}"
V = f"{A}*{V0}"
P = P0

ADV_U = adv(u0).expr()
ADV_V = adv(v0).expr()
PX = p0.deriv(0).expr()
PY = p0.deriv(1).expr()
LAP_U = u0.lap().expr()
LAP_V = v0.lap().expr()

FORCE_U = f"{AT}*{U0}+{A}*{A}*{ADV_U}+{PX}-Kinvis*{A}*{LAP_U}"
FORCE_V = f"{AT}*{V0}+{A}*{A}*{ADV_V}+{PY}-Kinvis*{A}*{LAP_V}"

MANUFACTURED_FUNCTIONS = f"""        <FUNCTION NAME="InitialConditions">
            <E VAR="u" VALUE="{A0}*{U0}" />
            <E VAR="v" VALUE="{A0}*{V0}" />
            <E VAR="p" VALUE="{P0}" />
        </FUNCTION>

        <FUNCTION NAME="ExactSolution">
            <E VAR="u" VALUE="{U}" />
            <E VAR="v" VALUE="{V}" />
            <E VAR="p" VALUE="{P}" />
        </FUNCTION>

        <FUNCTION NAME="BodyForce">
            <E VAR="u" VALUE="{FORCE_U}" />
            <E VAR="v" VALUE="{FORCE_V}" />
            <E VAR="p" VALUE="0" />
        </FUNCTION>"""

FORCING_BLOCK = """    <FORCING>
        <FORCE TYPE="Body">
            <BODYFORCE> BodyForce </BODYFORCE>
        </FORCE>
    </FORCING>"""

BOUNDARY_CONDITIONS = f"""        <BOUNDARYREGIONS>
            <B ID="0"> C[2-5] </B>
        </BOUNDARYREGIONS>

        <BOUNDARYCONDITIONS>
            <REGION REF="0">
                <D VAR="u" VALUE="{U}" />
                <D VAR="v" VALUE="{V}" />
                <D VAR="p" VALUE="{P}" />
            </REGION>
        </BOUNDARYCONDITIONS>"""


def install_manufactured_solution(text):
    text = re.sub(
        r"        <BOUNDARYREGIONS>.*?</BOUNDARYCONDITIONS>",
        BOUNDARY_CONDITIONS,
        text,
        flags=re.S,
    )
    text = re.sub(
        r"\n\s*<FUNCTION NAME=\"InitialConditions\">.*?</FUNCTION>\s*"
        r"<FUNCTION NAME=\"ExactSolution\">.*?</FUNCTION>",
        "\n" + MANUFACTURED_FUNCTIONS,
        text,
        flags=re.S,
    )
    text = re.sub(
        r"\n\s*<(?:FORCING|Forcing)>.*?</(?:FORCING|Forcing)>",
        "",
        text,
        flags=re.S,
    )
    text = re.sub(r"\n\s*<FILTERS>.*?</FILTERS>", "", text, flags=re.S)
    text = re.sub(r"\n\s*</NEKTAR>", f"\n{FORCING_BLOCK}\n</NEKTAR>", text)
    return text


def check_mixed_tri_quad_mesh(text, session):
    ntri = len(re.findall(r"<T\s+ID=", text))
    nquad = len(re.findall(r"<Q\s+ID=", text))
    if ntri == 0 or nquad == 0:
        raise RuntimeError(
            f"{session} must contain both triangles and quads; "
            f"found {ntri} triangles and {nquad} quads"
        )


def set_time_integration(text, method, order):
    replacement = f"""        <TIMEINTEGRATIONSCHEME>
            <METHOD> {method} </METHOD>
            <ORDER> {order} </ORDER>
        </TIMEINTEGRATIONSCHEME>"""
    return re.sub(
        r"        <TIMEINTEGRATIONSCHEME>.*?</TIMEINTEGRATIONSCHEME>",
        replacement,
        text,
        flags=re.S,
    )


def replace_parameter(text, name, value):
    return re.sub(
        rf"<P>\s*{name}\s*=\s*[^<]+</P>",
        f"<P> {name} = {value} </P>",
        text,
    )


def build_session(args, workdir, order):
    text = args.session.read_text()
    check_mixed_tri_quad_mesh(text, args.session)
    method = args.method_by_order.get(order, DEFAULT_METHOD_BY_ORDER[order])
    nsteps = args.steps_by_order.get(order, DEFAULT_STEPS_BY_ORDER[order])
    dt = args.final_time / nsteps

    text = set_time_integration(text, method, order)
    text = replace_parameter(text, "TimeStep", f"{dt:.16g}")
    text = replace_parameter(text, "NumSteps", str(nsteps))
    text = replace_parameter(text, "IO_InfoSteps", str(nsteps))
    text = replace_parameter(text, "Kinvis", f"{args.kinvis:.16g}")
    text = replace_parameter(text, "IterativeSolverTolerance", "1e-10")
    text = re.sub(r'NUMMODES="\d+"', f'NUMMODES="{args.num_modes}"', text)
    text = install_manufactured_solution(text)

    session = workdir / f"forced_mms_2d_{method}_order{order}.xml"
    session.write_text(text)
    return session


def run_case(args, workdir, order, formulation):
    session = build_session(args, workdir, order)
    cmd = [
        str(args.solver),
        f"--opExecSpace={args.op_exec_space}",
        f"--opImpl={args.op_impl}",
        "-I",
        f"Formulation={formulation}",
        str(session),
    ]
    proc = subprocess.run(
        cmd,
        cwd=workdir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=args.timeout,
    )

    if proc.returncode:
        return False, proc

    missing = []
    for var in VELOCITY_VARS:
        if not re.search(rf"L 2 error \(variable {var}\) : ([0-9.eE+-]+)", proc.stdout):
            missing.append(var)
    if missing:
        proc.stderr += "\nMissing L2 output for: " + ", ".join(missing)
        return False, proc

    return True, proc


def parse_int_map(values):
    out = {}
    for value in values:
        key, val = value.split("=", 1)
        out[int(key)] = int(val)
    return out


def parse_str_map(values):
    out = {}
    for value in values:
        key, val = value.split("=", 1)
        out[int(key)] = val
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", type=pathlib.Path, required=True)
    parser.add_argument("--session", type=pathlib.Path, required=True)
    parser.add_argument("--op-exec-space", default="Serial")
    parser.add_argument("--op-impl", default="SumFac")
    parser.add_argument("--orders", nargs="+", type=int, default=[2, 3])
    parser.add_argument("--formulations", nargs="+", default=["linearimplicit"])
    parser.add_argument("--steps", action="append", default=[], metavar="ORDER=N")
    parser.add_argument("--method", action="append", default=[], metavar="ORDER=METHOD")
    parser.add_argument("--final-time", type=float, default=0.02)
    parser.add_argument("--kinvis", type=float, default=0.05)
    parser.add_argument("--num-modes", type=int, default=7)
    parser.add_argument("--timeout", type=float, default=240.0)
    args = parser.parse_args()

    args.solver = args.solver.resolve()
    args.session = args.session.resolve()
    args.steps_by_order = parse_int_map(args.steps)
    args.method_by_order = parse_str_map(args.method)

    failed = False
    with tempfile.TemporaryDirectory(prefix="incns_imexdirk_mms2d_") as tmp:
        workdir = pathlib.Path(tmp)
        for order in args.orders:
            for formulation in args.formulations:
                ok, proc = run_case(args, workdir, order, formulation)
                label = f"order={order} formulation={formulation}"
                if ok:
                    print(f"{label}: passed", flush=True)
                else:
                    print(f"{label}: failed", flush=True)
                    print("=== stdout ===", flush=True)
                    print(proc.stdout, flush=True)
                    print("=== stderr ===", flush=True)
                    print(proc.stderr, flush=True)
                    failed = True

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
