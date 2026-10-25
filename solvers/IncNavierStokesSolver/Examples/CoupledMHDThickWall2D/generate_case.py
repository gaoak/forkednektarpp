#!/usr/bin/env python3
"""Generate matching fluid/electrical quad meshes and an analytical MHD case."""

from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parent


def write_xml(name, root):
    ET.indent(root, space="    ")
    ET.ElementTree(root).write(ROOT / name, encoding="utf-8", xml_declaration=True)


def mesh(name, fluid_only):
    xs, ys = [0, 0.5, 1], [-0.25, 0, 0.5, 1, 1.25]
    quads, edges = {}, {}
    vertices = {j * 3 + i: (x, y) for j, y in enumerate(ys) for i, x in enumerate(xs)}
    for j in range(4):
        for i in range(2):
            verts = [j * 3 + i, j * 3 + i + 1, (j + 1) * 3 + i + 1, (j + 1) * 3 + i]
            local_edges = []
            for a, b in zip(verts, verts[1:] + verts[:1]):
                key = tuple(sorted((a, b)))
                if key not in edges:
                    edges[key] = len(edges)
                local_edges.append(edges[key])
            quads[j * 2 + i] = local_edges
    selected = {q: es for q, es in quads.items() if not fluid_only or 2 <= q <= 5}
    used_edges = {e for es in selected.values() for e in es}
    used_verts = {v for pair, e in edges.items() if e in used_edges for v in pair}
    root = ET.Element("NEKTAR")
    geom = ET.SubElement(root, "GEOMETRY", DIM="2", SPACE="2")
    vs, es, qs = [ET.SubElement(geom, tag) for tag in ("VERTEX", "EDGE", "ELEMENT")]
    for v in sorted(used_verts):
        x, y = vertices[v]
        ET.SubElement(vs, "V", ID=str(v)).text = f" {x} {y} 0 "
    for pair, e in edges.items():
        if e in used_edges:
            ET.SubElement(es, "E", ID=str(e)).text = " " + " ".join(map(str, pair)) + " "
    for q, edge_ids in selected.items():
        ET.SubElement(qs, "Q", ID=str(q)).text = " " + " ".join(map(str, edge_ids)) + " "
    cs = ET.SubElement(geom, "COMPOSITE")

    def comp(cid, kind, ids):
        ET.SubElement(cs, "C", ID=str(cid)).text = f" {kind}[{','.join(map(str, ids))}] "

    comp(0, "Q", [q for q in selected if 2 <= q <= 5])
    if not fluid_only:
        comp(1, "Q", [q for q in selected if q < 2 or q > 5])
    ymin, ymax = (0, 1) if fluid_only else (-0.25, 1.25)
    for cid, axis, val in [(10, 1, ymin), (11, 1, ymax), (12, 0, 0), (13, 0, 1)]:
        comp(cid, "E", [e for pair, e in edges.items() if e in used_edges and all(vertices[v][axis] == val for v in pair)])
    ET.SubElement(geom, "DOMAIN").text = " C[0] " if fluid_only else " C[0,1] "
    write_xml(name, root)


def session(name, electric):
    root = ET.Element("NEKTAR")
    exps = ET.SubElement(root, "EXPANSIONS")
    ET.SubElement(exps, "E", COMPOSITE="C[0,1]" if electric else "C[0]", NUMMODES="4", NUMPOINTS="5", FIELDS="phi" if electric else "u,v,phi,p", TYPE="MODIFIED")
    coupling = ET.SubElement(root, "COUPLING", NAME="mhd", TYPE="Cwipi")
    options = {
        "RemoteName": "fluid" if electric else "electric",
        "SendSteps": "1", "ReceiveSteps": "1", "Oversample": "0",
        "SendVariables": "phi" if electric else "u,v,mhd_time,mhd_stop",
        "ReceiveVariables": "u,v,mhd_time,mhd_stop" if electric else "phi",
        "ReceiveProjection": "Local", "NotLocMethod": "Keep",
        "FilterWidth": "0", "GeomTol": "1e-10",
    }
    for key, value in options.items():
        ET.SubElement(coupling, "I", PROPERTY=key, VALUE=value)
    if electric:
        materials = ET.SubElement(root, "ELECTRICPOTENTIAL")
        ET.SubElement(materials, "FLUID", COMPOSITE="C[0]", CONDUCTIVITY="ElectricConductivity")
        ET.SubElement(materials, "SOLID", COMPOSITE="C[1]", CONDUCTIVITY="WallConductivity")
    cond = ET.SubElement(root, "CONDITIONS")
    info = ET.SubElement(cond, "SOLVERINFO")
    solver = {"EQType": "CoupledElectricPotential", "Projection": "Continuous"} if electric else {
        "EQType": "UnsteadyNavierStokes", "SolverType": "VelocityCorrectionScheme",
        "Projection": "Continuous", "ElectricPotential": "Coupled",
    }
    for key, value in solver.items():
        ET.SubElement(info, "I", PROPERTY=key, VALUE=value)
    if not electric:
        scheme = ET.SubElement(cond, "TIMEINTEGRATIONSCHEME")
        ET.SubElement(scheme, "METHOD").text = "IMEX"
        ET.SubElement(scheme, "ORDER").text = "2"
    params = ET.SubElement(cond, "PARAMETERS")
    values = {
        "Kinvis": "0.1", "TimeStep": "0.001", "NumSteps": "5", "IO_InfoSteps": "5",
        "U0": "0.1", "B0": "0.1", "Voltage": "0.1", "WallThickness": "0.25",
        "ElectricConductivity": "1", "WallConductivity": "0.1",
        "Resistance": "2*WallThickness/WallConductivity+1/ElectricConductivity",
        "JyExact": "-(Voltage+2*U0*B0/3)/Resistance",
    }
    for key, value in values.items():
        ET.SubElement(params, "P").text = f" {key} = {value} "
    variables = ET.SubElement(cond, "VARIABLES")
    for idx, var in enumerate(["phi"] if electric else ["u", "v", "phi", "p"]):
        ET.SubElement(variables, "V", ID=str(idx)).text = f" {var} "
    regions, bcs = [ET.SubElement(cond, tag) for tag in ("BOUNDARYREGIONS", "BOUNDARYCONDITIONS")]
    for idx in range(4):
        ET.SubElement(regions, "B", ID=str(idx)).text = f" C[{10+idx}] "
        region = ET.SubElement(bcs, "REGION", REF=str(idx))
        if electric:
            ET.SubElement(region, "D" if idx < 2 else "N", VAR="phi", VALUE="Voltage" if idx == 1 else "0")
        else:
            ET.SubElement(region, "D", VAR="u", VALUE="0" if idx < 2 else "4*U0*y*(1-y)")
            ET.SubElement(region, "D", VAR="v", VALUE="0")
            ET.SubElement(region, "N", VAR="phi", VALUE="0")
            p = ET.SubElement(region, "D" if idx == 3 else "N", VAR="p", VALUE="0")
            if idx != 3:
                p.set("USERDEFINEDTYPE", "H")
    clipped = "(0.5*(1+abs(y)-abs(y-1)))"
    def phi(t):
        resistance = f"((y+WallThickness)/WallConductivity+(1/ElectricConductivity-1/WallConductivity)*{t})"
        return f"-JyExact*{resistance}-U0*B0*(2*{t}^2-4*{t}^3/3)"
    exact = {"phi": phi(clipped)} if electric else {"u": "4*U0*y*(1-y)", "v": "0", "phi": phi("y"), "p": "0"}
    for fname in (["ExactSolution"] if electric else ["InitialConditions", "ExactSolution"]):
        fn = ET.SubElement(cond, "FUNCTION", NAME=fname)
        for var, value in exact.items():
            ET.SubElement(fn, "E", VAR=var, VALUE=value)
    em = ET.SubElement(cond, "FUNCTION", NAME="ElectricMagneticFields")
    ET.SubElement(em, "E", VAR="Bz", VALUE="B0")
    if not electric:
        fn = ET.SubElement(cond, "FUNCTION", NAME="BodyForce")
        ET.SubElement(fn, "E", VAR="u", VALUE="8*Kinvis*U0-JyExact*B0")
        ET.SubElement(fn, "E", VAR="v", VALUE="0")
        forcing = ET.SubElement(root, "FORCING")
        body = ET.SubElement(forcing, "FORCE", TYPE="Body")
        ET.SubElement(body, "BODYFORCE").text = "BodyForce"
        lorentz = ET.SubElement(forcing, "FORCE", TYPE="ForcingLorentz")
        ET.SubElement(lorentz, "ELECTRICMAGNETICFIELDS").text = "ElectricMagneticFields"
    write_xml(name, root)


if __name__ == "__main__":
    mesh("fluid-mesh.xml", True)
    mesh("electric-mesh.xml", False)
    session("fluid.xml", False)
    session("electric.xml", True)
