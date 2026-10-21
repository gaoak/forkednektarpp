#!/usr/bin/env python3

import argparse
import math
import pathlib
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET


MIN_RATES = {2: 1.7, 3: 2.5}
# Fine enough for an error of the startup step at a lower order to show
# in the final rate.
DEFAULT_STEPS = [8, 16, 32]
VARIABLES = ["u", "v", "p"]


def manufactured_expressions():
    # The velocity has a non-zero curl-curl, extrapolated from the velocity
    # history into the higher-order pressure flux, and a time-dependent
    # boundary value, whose acceleration enters it on the left edge.
    velocity = "y*(1-y)"
    pressure = "(x-0.5)*(y-0.5)"
    velocity_amplitude = "exp(t)"
    pressure_amplitude = "exp(2*t)"
    values = {
        "u": f"{velocity_amplitude}*{velocity}",
        "v": "0",
        "p": f"{pressure_amplitude}*{pressure}",
    }
    initial = {"u": velocity, "v": "0", "p": pressure}
    # du/dt - Kinvis * laplacian(u) + grad(p)
    grad_p_x = f"{pressure_amplitude}*(y-0.5)"
    grad_p_y = f"{pressure_amplitude}*(x-0.5)"
    force = {
        "u": f"{velocity_amplitude}*({velocity}+2*Kinvis)+{grad_p_x}",
        "v": grad_p_y,
    }
    return initial, values, force


def set_parameter(root, name, value):
    for parameter in root.findall(".//PARAMETERS/P"):
        lhs, separator, _ = (parameter.text or "").partition("=")
        if separator and lhs.strip() == name:
            parameter.text = f" {name} = {value} "
            return
    raise RuntimeError(f"session does not define parameter {name}")


def set_function(root, function_name, values):
    function = root.find(f".//FUNCTION[@NAME='{function_name}']")
    if function is None:
        raise RuntimeError(f"session does not define function {function_name}")
    for expression in function.findall("E"):
        variable = expression.attrib["VAR"]
        if variable in values:
            expression.set("VALUE", values[variable])


def build_session(args, workdir, formulation, order, nsteps):
    tree = ET.parse(args.session)
    root = tree.getroot()
    initial, exact, force = manufactured_expressions()

    root.find(".//TIMEINTEGRATIONSCHEME/METHOD").text = " IMEX "
    root.find(".//TIMEINTEGRATIONSCHEME/ORDER").text = f" {order} "
    set_parameter(root, "TimeStep", f"{args.final_time / nsteps:.16g}")
    set_parameter(root, "NumSteps", str(nsteps))
    set_parameter(root, "IO_InfoSteps", str(nsteps))
    set_parameter(root, "IterativeSolverTolerance", "1e-12")

    for expansion in root.findall(".//EXPANSIONS/E"):
        expansion.set("NUMMODES", str(args.num_modes))

    set_function(root, "InitialConditions", initial)
    set_function(root, "ExactSolution", exact)
    set_function(root, "BodyForce", force)

    for condition in root.findall(".//BOUNDARYCONDITIONS/REGION/*"):
        variable = condition.attrib["VAR"]
        if args.pressure_dirichlet and variable == "p":
            condition.tag = "D"
        if variable in exact and condition.tag == "D":
            condition.set("VALUE", exact[variable])
            condition.set("USERDEFINEDTYPE", "TimeDependent")

    path = workdir / f"hopbc_order{order}_{formulation}_{nsteps}.xml"
    tree.write(path, encoding="utf-8", xml_declaration=True)
    return path


def run_case(args, workdir, formulation, order, nsteps):
    session = build_session(args, workdir, formulation, order, nsteps)
    command = [
        str(args.solver),
        f"--opExecSpace={args.op_exec_space}",
        f"--opImpl={args.op_impl}",
        "-I",
        f"Formulation={formulation}",
        str(session),
    ]
    result = subprocess.run(
        command,
        cwd=workdir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=args.timeout,
    )
    if result.returncode:
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError(
            f"order={order} {formulation} nsteps={nsteps} failed"
        )

    errors = {}
    for variable in VARIABLES:
        match = re.search(
            rf"L 2 error \(variable {variable}\) : ([0-9.eE+-]+)",
            result.stdout,
        )
        if not match:
            print(result.stdout)
            print(result.stderr)
            raise RuntimeError(f"missing L2 error for variable {variable}")
        errors[variable] = float(match.group(1))
    return errors


def convergence_rate(coarse, fine, coarse_steps, fine_steps):
    return math.log(coarse / fine) / math.log(fine_steps / coarse_steps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", type=pathlib.Path, required=True)
    parser.add_argument("--session", type=pathlib.Path, required=True)
    parser.add_argument("--op-exec-space", default="Serial")
    parser.add_argument("--op-impl", default="SumFac")
    parser.add_argument("--orders", nargs="+", type=int, default=[2, 3])
    parser.add_argument(
        "--formulations", nargs="+", default=["semiimplicit", "linearimplicit"]
    )
    parser.add_argument("--steps", nargs="+", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--final-time", type=float, default=0.2)
    parser.add_argument("--num-modes", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--pressure-dirichlet", action="store_true")
    args = parser.parse_args()
    args.solver = args.solver.resolve()
    args.session = args.session.resolve()

    failed = False
    with tempfile.TemporaryDirectory(prefix="incns_hopbc_time_order_") as tmp:
        workdir = pathlib.Path(tmp)
        for order in args.orders:
            for formulation in args.formulations:
                errors = [
                    run_case(args, workdir, formulation, order, nsteps)
                    for nsteps in args.steps
                ]
                rates = {
                    variable: [
                        convergence_rate(
                            errors[i - 1][variable],
                            errors[i][variable],
                            args.steps[i - 1],
                            args.steps[i],
                        )
                        for i in range(1, len(errors))
                    ]
                    for variable in VARIABLES
                }
                print(
                    f"order={order} {formulation}: errors={errors} rates={rates}",
                    flush=True,
                )

                # Pressure directly exercises the extrapolated higher-order
                # Neumann data; velocity verifies its coupling through the
                # projection.
                observed = min(rates["p"][-1], rates["u"][-1], rates["v"][-1])
                if observed < MIN_RATES[order]:
                    print(
                        f"order={order} {formulation}: expected final rate >= "
                        f"{MIN_RATES[order]}, got {observed}",
                        flush=True,
                    )
                    failed = True

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
