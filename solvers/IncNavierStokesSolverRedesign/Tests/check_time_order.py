#!/usr/bin/env python3

import argparse
import math
import pathlib
import re
import subprocess
import tempfile


MIN_RATES = {1: 0.85, 2: 1.70, 3: 2.30}
NSTEPS_BY_ORDER = {1: [2, 4], 2: [2, 4], 3: [2, 4]}
FORMULATIONS = ["semiimplicit", "linearimplicit"]
VELOCITY_VARS = ["u", "v", "w"]

class Poly:
    def __init__(self, terms=None):
        self.terms = {k: v for k, v in (terms or {}).items() if abs(v) > 1e-14}

    @staticmethod
    def const(value):
        return Poly({(0, 0, 0): float(value)})

    @staticmethod
    def var(axis):
        exp = [0, 0, 0]
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
                exp = tuple(exp1[i] + exp2[i] for i in range(3))
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
            terms[tuple(new_exp)] = terms.get(tuple(new_exp), 0.0) + coef * exp[axis]
        return Poly(terms)

    def lap(self):
        return self.deriv(0).deriv(0) + self.deriv(1).deriv(1) + self.deriv(2).deriv(2)

    def expr(self):
        if not self.terms:
            return "0"
        parts = []
        names = ["x", "y", "z"]
        for exp, coef in sorted(self.terms.items()):
            factors = []
            abs_coef = abs(coef)
            if exp == (0, 0, 0) or abs(abs_coef - 1.0) > 1e-14:
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
z = Poly.var(2)

# Bubble factors for x in [-0.5, 1], y in [-0.5, 1.5], z in [0, 1].
X = (x + 0.5) * (1.0 - x)
Y = (y + 0.5) * (1.5 - y)
Z = z * (1.0 - z)
phi = (X**2) * (Y**2) * (Z**2)
phi_x = phi.deriv(0)
phi_y = phi.deriv(1)
phi_z = phi.deriv(2)
q = [phi_y - phi_z, phi_z - phi_x, phi_x - phi_y]
p0 = x * y + y * z + z * x

def adv(component):
    return q[0] * component.deriv(0) + q[1] * component.deriv(1) + q[2] * component.deriv(2)

A = "(1+0.1*sin(t))"
AT = "(0.1*cos(t))"
A0 = "1"

U0, V0, W0 = [component.expr() for component in q]
P0 = p0.expr()
U = f"{A}*{U0}"
V = f"{A}*{V0}"
W = f"{A}*{W0}"
P = P0

ADV_U, ADV_V, ADV_W = [component.expr() for component in [adv(q[0]), adv(q[1]), adv(q[2])]]
PX, PY, PZ = [p0.deriv(axis).expr() for axis in range(3)]
LAP_U, LAP_V, LAP_W = [component.lap().expr() for component in q]

FORCE_U = f"{AT}*{U0}+{A}*{A}*{ADV_U}+{PX}-Kinvis*{A}*{LAP_U}"
FORCE_V = f"{AT}*{V0}+{A}*{A}*{ADV_V}+{PY}-Kinvis*{A}*{LAP_V}"
FORCE_W = f"{AT}*{W0}+{A}*{A}*{ADV_W}+{PZ}-Kinvis*{A}*{LAP_W}"

MANUFACTURED_FUNCTIONS = f"""        <FUNCTION NAME="InitialConditions">
            <E VAR="u" VALUE="{A0}*{U0}" />
            <E VAR="v" VALUE="{A0}*{V0}" />
            <E VAR="w" VALUE="{A0}*{W0}" />
            <E VAR="p" VALUE="{P0}" />
        </FUNCTION>

        <FUNCTION NAME="ExactSolution">
            <E VAR="u" VALUE="{U}" />
            <E VAR="v" VALUE="{V}" />
            <E VAR="w" VALUE="{W}" />
            <E VAR="p" VALUE="{P}" />
        </FUNCTION>

        <FUNCTION NAME="BodyForce">
            <E VAR="u" VALUE="{FORCE_U}" />
            <E VAR="v" VALUE="{FORCE_V}" />
            <E VAR="w" VALUE="{FORCE_W}" />
            <E VAR="p" VALUE="0" />
        </FUNCTION>"""

FORCING_BLOCK = """    <FORCING>
        <FORCE TYPE="Body">
            <BODYFORCE> BodyForce </BODYFORCE>
        </FORCE>
    </FORCING>"""

BOUNDARY_CONDITIONS = f"""        <BOUNDARYREGIONS>
            <B ID="0"> C[1-6] </B>
        </BOUNDARYREGIONS>

        <BOUNDARYCONDITIONS>
            <REGION REF="0">
                <D VAR="u" VALUE="0" />
                <D VAR="v" VALUE="0" />
                <D VAR="w" VALUE="0" />
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
    text = re.sub(r"\n\s*<(?:FORCING|Forcing)>.*?</(?:FORCING|Forcing)>", "", text, flags=re.S)
    text = re.sub(r"\n\s*<FILTERS>.*?</FILTERS>", "", text, flags=re.S)
    text = re.sub(r"\n\s*</NEKTAR>", f"\n{FORCING_BLOCK}\n</NEKTAR>", text)
    return text


def build_session(base_session, workdir, formulation, order, nsteps, final_time):
    text = base_session.read_text()
    dt = final_time / nsteps

    text = re.sub(r"<ORDER>\s*\d+\s*</ORDER>", f"<ORDER> {order} </ORDER>", text)
    text = re.sub(
        r"<P>\s*TimeStep\s*=\s*[^<]+</P>",
        f"<P> TimeStep = {dt:.16g} </P>",
        text,
    )
    text = re.sub(
        r"<P>\s*NumSteps\s*=\s*[^<]+</P>",
        f"<P> NumSteps = {nsteps} </P>",
        text,
    )
    text = re.sub(
        r"<P>\s*IO_InfoSteps\s*=\s*[^<]+</P>",
        f"<P> IO_InfoSteps = {nsteps} </P>",
        text,
    )
    text = re.sub(r'NUMMODES="\d+"', 'NUMMODES="5"', text)
    text = re.sub(
        r"<P>\s*IterativeSolverTolerance\s*=\s*[^<]+</P>",
        "<P> IterativeSolverTolerance = 1e-7 </P>",
        text,
    )
    text = re.sub(
        r"<P>\s*NekLinSysMaxIterations\s*=\s*[^<]+</P>",
        "<P> NekLinSysMaxIterations = 1000 </P>",
        text,
    )
    text = install_manufactured_solution(text)

    session = workdir / f"manufactured3d_order{order}_{formulation}_{nsteps}.xml"
    session.write_text(text)
    return session


def run_case(args, workdir, formulation, order, nsteps):
    session = build_session(
        args.session, workdir, formulation, order, nsteps, args.final_time
    )
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
        print(proc.stdout)
        print(proc.stderr)
        raise RuntimeError(
            f"order={order} {formulation} nsteps={nsteps} failed"
        )

    errors = []
    for var in VELOCITY_VARS:
        match = re.search(rf"L 2 error \(variable {var}\) : ([0-9.eE+-]+)", proc.stdout)
        if not match:
            print(proc.stdout)
            raise RuntimeError(
                f"order={order} {formulation} nsteps={nsteps} did not report {var} L2"
            )
        errors.append(float(match.group(1)))
    return max(errors)


def observed_rates(errors):
    return [math.log(errors[i - 1] / errors[i], 2.0) for i in range(1, len(errors))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", type=pathlib.Path, required=True)
    parser.add_argument("--session", type=pathlib.Path, required=True)
    parser.add_argument("--op-exec-space", default="Serial")
    parser.add_argument("--op-impl", default="SumFac")
    parser.add_argument("--final-time", type=float, default=0.2)
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()
    args.solver = args.solver.resolve()
    args.session = args.session.resolve()

    with tempfile.TemporaryDirectory(prefix="incns_time_order_") as tmp:
        workdir = pathlib.Path(tmp)
        failed = False

        for order in [1, 2, 3]:
            for formulation in FORMULATIONS:
                errors = [
                    run_case(args, workdir, formulation, order, nsteps)
                    for nsteps in NSTEPS_BY_ORDER[order]
                ]
                rates = observed_rates(errors)
                min_rate = MIN_RATES[order]

                print(
                    f"order={order} {formulation}: "
                    f"errors={errors} rates={rates}",
                    flush=True,
                )

                if rates[-1] < min_rate:
                    print(
                        f"order={order} {formulation}: expected final "
                        f"observed order >= {min_rate}, got {rates[-1]}",
                        flush=True,
                    )
                    failed = True

        if failed:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
