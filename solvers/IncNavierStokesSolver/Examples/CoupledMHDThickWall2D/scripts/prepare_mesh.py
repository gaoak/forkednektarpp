#!/usr/bin/env python3
"""Extract matching straight 2D quad meshes without renumbering geometry.

Input: Mesh2D square XML (default), or NekMesh/Gmsh XML with --source-format gmsh.
Mesh2D cells are classified by y and edges by coordinates. Gmsh uses the
physical groups from thick_wall_channel.geo. Output composites match the sessions.
Only the Python standard library is required. This teaching helper deliberately
rejects curved, compressed, partitioned and non-quad meshes.
"""

import argparse
from copy import deepcopy
from pathlib import Path
import re
import xml.etree.ElementTree as ET


def ids(text):
    result = set()
    for token in text.strip().split(","):
        bounds = [int(value) for value in token.strip().split("-")]
        if len(bounds) == 1:
            result.add(bounds[0])
        elif len(bounds) == 2 and bounds[0] <= bounds[1]:
            result.update(range(bounds[0], bounds[1] + 1))
        else:
            raise ValueError(f"Invalid ID list: {text}")
    return result


def prepare(source, output, source_format="mesh2d", ymin=0.0, ymax=1.0):
    geometry = ET.parse(source).getroot().find("GEOMETRY")
    if geometry is None or (geometry.get("DIM"), geometry.get("SPACE")) != ("2", "2"):
        raise ValueError("Expected a full 2D GEOMETRY (DIM=SPACE=2).")
    if "PARTITION" in geometry.attrib or "HDF5FILE" in geometry.attrib:
        raise ValueError("Use an unpartitioned XML mesh, not a partition or HDF5 mesh.")
    for section in ("VERTEX", "EDGE", "ELEMENT", "COMPOSITE"):
        node = geometry.find(section)
        if node is None or node.get("COMPRESSED") or len(node) == 0:
            raise ValueError("Use NekMesh input.msh full.xml:xml:uncompress first.")
    curves = geometry.find("CURVED")
    if curves is not None and len(curves):
        raise ValueError("This helper handles straight quadrilaterals only.")
    vertices = {int(v.get("ID")): v for v in geometry.find("VERTEX")}
    edges = {int(e.get("ID")): e for e in geometry.find("EDGE")}
    elements = {int(e.get("ID")): e for e in geometry.find("ELEMENT")}
    if any(e.tag != "Q" for e in elements.values()):
        raise ValueError("Expected only Q elements; enable Recombine Surface in Gmsh.")
    composites = {int(c.get("ID")): c for c in geometry.find("COMPOSITE")}
    if source_format == "mesh2d":
        if ymin >= ymax:
            raise ValueError("Fluid ymin must be smaller than ymax.")
        tolerance = 1e-10
        points = {gid: [float(v) for v in node.text.split()]
                  for gid, node in vertices.items()}
        fluid, solid = set(), set()
        for gid, element in elements.items():
            vids = {int(v) for edge in element.text.split()
                    for v in edges[int(edge)].text.split()}
            ys = [points[v][1] for v in vids]
            if min(ys) >= ymin-tolerance and max(ys) <= ymax+tolerance:
                fluid.add(gid)
            elif max(ys) <= ymin+tolerance or min(ys) >= ymax-tolerance:
                solid.add(gid)
            else:
                raise ValueError(f"Element {gid} crosses a fluid/solid interface; add y nodes there.")
        xmin = min(p[0] for p in points.values())
        xmax = max(p[0] for p in points.values())
        outer_ymin = min(p[1] for p in points.values())
        outer_ymax = max(p[1] for p in points.values())
        if not outer_ymin < ymin < ymax < outer_ymax:
            raise ValueError("Expected solid walls below and above the fluid interval.")
        groups = {1: ("Q", fluid), 2: ("Q", solid)}
        for cid, axis, value in ((10, 1, outer_ymin), (11, 1, outer_ymax),
                                 (12, 0, xmin), (13, 0, xmax),
                                 (14, 1, ymin), (15, 1, ymax)):
            members = {gid for gid, edge in edges.items()
                       if all(abs(points[int(v)][axis]-value) <= tolerance
                              for v in edge.text.split())}
            groups[cid] = ("E", members)
        composites = {}
        for cid, (kind, members) in groups.items():
            node = ET.Element("C", ID=str(cid))
            node.text = f"{kind}[{','.join(map(str, sorted(members)))}]"
            composites[cid] = node

    def composite(cid, kind):
        if cid not in composites:
            raise ValueError(f"Missing physical group/composite {cid}.")
        match = re.fullmatch(r"([A-Z])\[([\d,\s-]+)\]", composites[cid].text.strip())
        if not match or match[1] != kind:
            raise ValueError(f"Composite {cid} must contain {kind} elements.")
        return ids(match[2])

    fluid, solid = composite(1, "Q"), composite(2, "Q")
    if not fluid or not solid or fluid & solid or fluid | solid != set(elements):
        raise ValueError("Fluid/solid groups must be nonempty and cover the mesh exactly once.")
    for e in elements.values():
        connectivity = [int(v) for v in e.text.split()]
        if len(connectivity) != 4 or not set(connectivity) <= edges.keys():
            raise ValueError("Quad connectivity must reference four existing edges.")

    output.mkdir(parents=True, exist_ok=True)

    def write(name, selected, electric):
        used_edges = {int(v) for eid in selected for v in elements[eid].text.split()}
        used_vertices = {int(v) for eid in used_edges for v in edges[eid].text.split()}
        root = ET.Element("NEKTAR")
        geom = ET.SubElement(root, "GEOMETRY", DIM="2", SPACE="2")
        for tag, table, selected_ids in (("VERTEX", vertices, used_vertices),
                                          ("EDGE", edges, used_edges),
                                          ("ELEMENT", elements, selected)):
            section = ET.SubElement(geom, tag, geometry.find(tag).attrib)
            for gid in sorted(selected_ids):
                section.append(deepcopy(table[gid]))
        comps = ET.SubElement(geom, "COMPOSITE")

        def add(cid, kind, members):
            if not members:
                raise ValueError(f"Empty output composite {cid} in {name}.")
            ET.SubElement(comps, "C", ID=str(cid)).text = (
                f" {kind}[{','.join(map(str, sorted(members)))}] ")

        add(0, "Q", fluid)
        if electric:
            add(1, "Q", solid)
        for cid, source_id in ((10, 10 if electric else 14),
                               (11, 11 if electric else 15), (12, 12), (13, 13)):
            add(cid, "E", composite(source_id, "E") & used_edges)
        ET.SubElement(geom, "DOMAIN").text = " C[0,1] " if electric else " C[0] "
        ET.indent(root, space="    ")
        ET.ElementTree(root).write(output / name, encoding="utf-8", xml_declaration=True)

    write("fluid-mesh.xml", fluid, False)
    write("electric-mesh.xml", fluid | solid, True)
    print(f"Prepared {len(fluid)} fluid and {len(solid)} solid quads in {output}")
    print("Preserved element, edge and vertex IDs and element orientations.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="uncompressed full mesh XML")
    parser.add_argument("output", type=Path, help="directory for the two output meshes")
    parser.add_argument("--source-format", choices=("mesh2d", "gmsh"), default="mesh2d")
    parser.add_argument("--fluid-ymin", type=float, default=0.0)
    parser.add_argument("--fluid-ymax", type=float, default=1.0)
    args = parser.parse_args()
    try:
        prepare(args.source, args.output, args.source_format,
                args.fluid_ymin, args.fluid_ymax)
    except (ValueError, KeyError, OSError, ET.ParseError) as error:
        parser.exit(1, f"Mesh preparation failed: {error}\n")
