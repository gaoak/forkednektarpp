#!/usr/bin/env python3
"""Generate the tetrahedral box mesh for the 3D shock tube test.

Writes a Gmsh 2.2 ASCII file which NekMesh converts to the Nektar session
geometry:

    python3 MakeShockTube3DTetMesh.py ShockTube_3D_tet.msh
    NekMesh ShockTube_3D_tet.msh ShockTube_3D_tet.xml:xml:uncompress

The domain is a box, long in x and short in y and z, cut into a structured
grid of cells each split into six tetrahedra by the Kuhn subdivision about
the main diagonal 0-6.  The shock runs normal to x, so the y and z faces are
periodic and the solution is one-dimensional; the mesh exists to put that
one-dimensional solution on tetrahedra, which is what the modal shock sensor
had never been tested on.

Two properties matter and are asserted rather than assumed.

*Handedness.*  Nektar does not reject a tetrahedron whose local axes are
left-handed: it runs, it converges, and it is quietly wrong, exactly as
MakeChanFlow3DMesh.py records for prisms.  The reference tetrahedron runs
xi1 along v0->v1, xi2 along v0->v3 and xi3 along v0->v4, so
(e1 x e2).e3 must be positive for every element, which is checked below.

*Periodic triangulation.*  A periodic face pair has to be triangulated the
same way on both sides, or the two halves do not match under translation.
The Kuhn subdivision gives that for free -- the y=0 and y=Ly faces both take
the diagonal from the (i,k) corner to the (i+1,k+1) corner, and likewise in
z -- and the check at the end confirms it instead of trusting the argument.

The test that uses this mesh puts slip walls on the y and z faces rather than
making them periodic, which is a deliberate choice and not a limitation.
Periodic laterals do work: Nektar pairs periodic faces by their order within
the composite rather than geometrically, so the two composites have to be
aligned first, which is what NekMesh's peralign module is for --

    NekMesh -m 'peralign:surf1=3,5:surf2=4,6:dir=y;z' in.msh out.xml

aligns both pairs at once.  Leave off the module's "orient" sub-option: it
reorients elements to match the faces, which breaks this mesh's symmetry
under exchanging y and z and so costs the check that rhov and rhow come out
equal.

Slip walls are preferred because they are exact for a purely axial flow and,
unlike periodic faces, they leave the lateral faces on a physical boundary.
That is what lets the case see the boundary trace of the artificial
viscosity, which is averaged against zero there by default; with periodic
laterals those faces are interior traces and the convention is invisible,
which is exactly why the two-dimensional shock tube cannot distinguish it.
"""

import sys

# Cells in each direction and the box they fill.  Ten cells across the shock
# normal puts the initial profile, which is much thinner than a cell, inside
# one or two elements, so the sensor fires on a few and not on the rest.
NX, NY, NZ = 10, 2, 2
LX, LY, LZ = 1.0, 0.2, 0.2

# Physical surface tags, in the order the session's boundary regions use.
INFLOW, OUTFLOW, YMIN, YMAX, ZMIN, ZMAX, DOMAIN = 1, 2, 3, 4, 5, 6, 7

# The six tetrahedra of a cell, as local hexahedral vertex numbers.  Local
# vertex v has offsets OFFS[v]; the subdivision fans about the diagonal 0-6.
OFFS = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
        (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
TETS = [(0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6),
        (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6)]

# Faces of a tetrahedron as local vertex triples, in Nektar's ordering.
TET_FACES = [(0, 1, 2), (0, 1, 3), (1, 2, 3), (0, 2, 3)]


def nid(i, j, k):
    """Gmsh node number, one-based, of grid point (i, j, k)."""
    return 1 + i + (NX + 1) * (j + (NY + 1) * k)


def coord(i, j, k):
    return (LX * i / NX, LY * j / NY, LZ * k / NZ)


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def handedness(p):
    """(e1 x e2).e3 of a tetrahedron given its four vertex coordinates."""
    e1, e2, e3 = (sub(p[1], p[0]), sub(p[2], p[0]), sub(p[3], p[0]))
    return dot(cross(e1, e2), e3)


def main(path):
    nodes = []
    for k in range(NZ + 1):
        for j in range(NY + 1):
            for i in range(NX + 1):
                nodes.append((nid(i, j, k), coord(i, j, k)))

    tets = []
    for k in range(NZ):
        for j in range(NY):
            for i in range(NX):
                corner = [nid(i + di, j + dj, k + dk) for di, dj, dk in OFFS]
                pos = [coord(i + di, j + dj, k + dk) for di, dj, dk in OFFS]
                for t in TETS:
                    assert handedness([pos[v] for v in t]) > 0, \
                        f"cell ({i},{j},{k}) tet {t} is left-handed"
                    tets.append(tuple(corner[v] for v in t))

    # Boundary triangles are taken from the volume mesh rather than
    # constructed separately: every tetrahedral face all of whose vertices sit
    # on a boundary plane is one, which cannot disagree with the elements.
    xs = {nd: xyz for nd, xyz in nodes}
    eps = 1.0e-12
    planes = [(INFLOW, 0, 0.0), (OUTFLOW, 0, LX),
              (YMIN, 1, 0.0), (YMAX, 1, LY),
              (ZMIN, 2, 0.0), (ZMAX, 2, LZ)]

    tris = []
    for tet in tets:
        for f in TET_FACES:
            face = tuple(tet[v] for v in f)
            for tag, axis, value in planes:
                if all(abs(xs[n][axis] - value) < eps for n in face):
                    tris.append((tag, face))
                    break

    # A structured grid of NX*NY*NZ cells has two triangles per cell face on
    # each of the six sides.
    expect = {INFLOW: 2 * NY * NZ, OUTFLOW: 2 * NY * NZ,
              YMIN: 2 * NX * NZ, YMAX: 2 * NX * NZ,
              ZMIN: 2 * NX * NY, ZMAX: 2 * NX * NY}
    for tag, n in expect.items():
        got = sum(1 for t, _ in tris if t == tag)
        assert got == n, f"surface {tag} has {got} triangles, expected {n}"

    # A periodic pair must carry the same triangulation, or the two halves do
    # not match under translation.  Compare the sorted vertex coordinates of
    # each triangle with the offset removed.
    def signature(tag, axis, shift):
        out = []
        for t, face in tris:
            if t != tag:
                continue
            pts = []
            for n in face:
                x, y, z = xs[n]
                p = [x, y, z]
                p[axis] -= shift
                pts.append(tuple(round(c, 12) for c in p))
            out.append(tuple(sorted(pts)))
        return sorted(out)

    assert signature(YMIN, 1, 0.0) == signature(YMAX, 1, LY), \
        "the y faces are not triangulated alike; periodicity would not match"
    assert signature(ZMIN, 2, 0.0) == signature(ZMAX, 2, LZ), \
        "the z faces are not triangulated alike; periodicity would not match"

    with open(path, "w") as f:
        f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
        f.write("$PhysicalNames\n7\n")
        f.write('2 1 "inflow"\n2 2 "outflow"\n')
        f.write('2 3 "ymin"\n2 4 "ymax"\n2 5 "zmin"\n2 6 "zmax"\n')
        f.write('3 7 "domain"\n$EndPhysicalNames\n')

        f.write(f"$Nodes\n{len(nodes)}\n")
        for n, (x, y, z) in nodes:
            f.write(f"{n} {x:.16g} {y:.16g} {z:.16g}\n")
        f.write("$EndNodes\n")

        f.write(f"$Elements\n{len(tris) + len(tets)}\n")
        e = 0
        for tag, face in tris:
            e += 1
            f.write(f"{e} 2 2 {tag} {tag} " + " ".join(map(str, face)) + "\n")
        for tet in tets:
            e += 1
            f.write(f"{e} 4 2 {DOMAIN} {DOMAIN} " + " ".join(map(str, tet))
                    + "\n")
        f.write("$EndElements\n")

    print(f"{path}: {len(nodes)} nodes, {len(tets)} tetrahedra, "
          f"{len(tris)} boundary triangles")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "ShockTube_3D_tet.msh")
