#!/usr/bin/env python3
"""Generate matching fluid/electrical quad meshes and a fully developed Hunt MHD case."""

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
    ET.SubElement(exps, "E", COMPOSITE="C[0,1]" if electric else "C[0]",
                  NUMMODES="5", NUMPOINTS="7", TYPE="MODIFIED",
                  FIELDS="phi" if electric else "u,v,w,phi")
    if not electric:
        ET.SubElement(exps, "E", COMPOSITE="C[0]", NUMMODES="4",
                      NUMPOINTS="7", TYPE="MODIFIED", FIELDS="p")
    coupling = ET.SubElement(root, "COUPLING", NAME="mhd", TYPE="Coefficients")
    for key, value in {
        "Role": "Electric" if electric else "Fluid", "FluidRanks": "2",
        "SendSteps": "1", "ReceiveSteps": "1",
        "SendVariables": "phi" if electric else "u,v,w,mhd_time,mhd_stop",
        "ReceiveVariables": "u,v,w,mhd_time,mhd_stop" if electric else "phi",
    }.items():
        ET.SubElement(coupling, "I", PROPERTY=key, VALUE=value)
    if electric:
        materials = ET.SubElement(root, "ELECTRICPOTENTIAL")
        ET.SubElement(materials, "FLUID", COMPOSITE="C[0]", CONDUCTIVITY="ElectricConductivity")
        ET.SubElement(materials, "SOLID", COMPOSITE="C[1]", CONDUCTIVITY="WallConductivity")
    cond = ET.SubElement(root, "CONDITIONS")
    info = ET.SubElement(cond, "SOLVERINFO")
    solver = {"EQType": "CoupledElectricPotential", "Projection": "Continuous"} if electric else {
        "EQType": "UnsteadyNavierStokes", "SolverType": "VelocityCorrectionScheme",
        "Projection": "Continuous", "Homogeneous": "1D", "ElectricPotential": "Coupled",
    }
    for key, value in solver.items():
        ET.SubElement(info, "I", PROPERTY=key, VALUE=value)
    if not electric:
        scheme = ET.SubElement(cond, "TIMEINTEGRATIONSCHEME")
        ET.SubElement(scheme, "METHOD").text = "IMEX"
        ET.SubElement(scheme, "ORDER").text = "1"
    params = ET.SubElement(cond, "PARAMETERS")
    for key, value in {
        "Kinvis": "0.1", "TimeStep": "0.02", "NumSteps": "500", "IO_InfoSteps": "100",
        "HomModesZ": "2", "LZ": "1", "Flowrate": "0.1", "IO_FlowSteps": "100",
        "IO_FlowStepsPrecision": "12", "Ha": "1", "HalfWidth": "0.5",
        "ElectricConductivity": "1", "WallConductivity": "10", "WallThickness": "0.25",
        "MagneticFieldStrength": "Ha/HalfWidth*sqrt(Kinvis/ElectricConductivity)",
    }.items():
        ET.SubElement(params, "P").text = f" {key} = {value} "
    variables = ET.SubElement(cond, "VARIABLES")
    names = ["phi"] if electric else ["u", "v", "w", "phi", "p"]
    for idx, var in enumerate(names):
        ET.SubElement(variables, "V", ID=str(idx)).text = f" {var} "
    regions, bcs = [ET.SubElement(cond, tag) for tag in ("BOUNDARYREGIONS", "BOUNDARYCONDITIONS")]
    for idx in range(4):
        ET.SubElement(regions, "B", ID=str(idx)).text = f" C[{10+idx}] "
        region = ET.SubElement(bcs, "REGION", REF=str(idx))
        if electric:
            # All outer surfaces are electrically insulated; no electrodes.
            ET.SubElement(region, "N", VAR="phi", VALUE="0")
        else:
            for var in ("u", "v", "w"):
                ET.SubElement(region, "D", VAR=var, VALUE="0")
            ET.SubElement(region, "N", VAR="phi", VALUE="0")
            ET.SubElement(region, "N", VAR="p", VALUE="0", USERDEFINEDTYPE="H")
    if not electric:
        initial = ET.SubElement(cond, "FUNCTION", NAME="InitialConditions")
        for var in names:
            ET.SubElement(initial, "E", VAR=var, VALUE="0")
    em = ET.SubElement(cond, "FUNCTION", NAME="ElectricMagneticFields")
    ET.SubElement(em, "E", VAR="By", VALUE="MagneticFieldStrength")
    if not electric:
        forcing = ET.SubElement(root, "FORCING")
        lorentz = ET.SubElement(forcing, "FORCE", TYPE="ForcingLorentz")
        ET.SubElement(lorentz, "ELECTRICMAGNETICFIELDS").text = "ElectricMagneticFields"
        ET.SubElement(lorentz, "IMPLICITDAMPING").text = "True"
    write_xml(name, root)


if __name__ == "__main__":
    mesh("fluid-mesh.xml", True)
    mesh("electric-mesh.xml", False)
    session("fluid.xml", False)
    session("electric.xml", True)
