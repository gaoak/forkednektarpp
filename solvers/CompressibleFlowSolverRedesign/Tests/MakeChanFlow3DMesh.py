"""Structured hex/prism mesh for the ChanFlow3D channel.

Same geometry as the original ChanFlow3D_infTurbExpl mesh - x in [0,Lx],
y in [-1,1], z in [0,Lz] periodic - and the same element mix, prisms in the
near-wall rows and hexes through the core, but structured and far coarser. The
original was sized for a turbulent inflow driven by synthetic eddy forcing; the
boundary condition tests it now carries need only enough elements to have an
interior.

The prisms are triangles in the x-y plane extruded along z, as in the original,
so the periodic z planes carry a mix of triangular and quadrilateral faces and
every wall, inlet and outlet face stays a quadrilateral.

Composite ids match the ones the sessions already reference:
    1   lower wall  y = -1        100 prisms
    2   outlet      x = Lx        101 hexes
    3   upper wall  y = +1        103 z = 0
    4   inlet       x = 0         104 z = Lz
"""

import math
import sys

NX, NY, NZ = 4, 6, 2
LX = 2.0
LZ = math.pi
BETA = 1.4  # y clustering towards the walls

# Near-wall rows are prismatic, the rest hexahedral.
PRISM_ROWS = {0, NY - 1}


def yline(n, beta):
    """Symmetric tanh stretch on [-1, 1], clustered at both ends."""
    t = math.tanh(beta)
    return [math.tanh(beta * (2.0 * i / n - 1.0)) / t for i in range(n + 1)]


xs = [LX * i / NX for i in range(NX + 1)]
ys = yline(NY, BETA)
zs = [LZ * k / NZ for k in range(NZ + 1)]

# ---------------------------------------------------------------- vertices
vid = {}
verts = []
for k in range(NZ + 1):
    for j in range(NY + 1):
        for i in range(NX + 1):
            vid[(i, j, k)] = len(verts)
            verts.append((xs[i], ys[j], zs[k]))

# ------------------------------------------------------------ edges, faces
edges = {}
faces = {}
faceverts = []


def edge(a, b):
    key = (a, b) if a < b else (b, a)
    if key not in edges:
        edges[key] = len(edges)
    return edges[key]


def face(vs):
    """Face from three or four vertices, in order. Returns its id."""
    key = tuple(sorted(vs))
    if key not in faces:
        faces[key] = len(faceverts)
        faceverts.append(vs)
    return faces[key]


# Local vertex numbering, Nektar's ordering in both cases. A hexahedron has
# v0..v3 on the base and v4..v7 above; a prism has the base quad v0..v3 with
# the apex edge v4-v5 over v0-v1 and v3-v2, so its two triangles are (0,1,4)
# and (3,2,5).
HEXFACES = [(0, 1, 2, 3), (0, 1, 5, 4), (1, 2, 6, 5),
            (3, 2, 6, 7), (0, 3, 7, 4), (4, 5, 6, 7)]
PRISMFACES = [(0, 1, 2, 3), (0, 1, 4), (1, 2, 5, 4), (3, 2, 5), (0, 3, 5, 4)]

prisms, hexes = [], []
prismverts, hexverts = [], []
for k in range(NZ):
    for j in range(NY):
        for i in range(NX):
            # The eight corners of the cell, bottom-z then top-z.
            c = [[vid[(i, j, k + d)], vid[(i + 1, j, k + d)],
                  vid[(i + 1, j + 1, k + d)], vid[(i, j + 1, k + d)]]
                 for d in (0, 1)]

            if j in PRISM_ROWS:
                # Split the x-y quad on the diagonal and extrude each triangle
                # along z. A prism whose triangle is (a,b,c) at z=k over
                # (A,B,C) at z=k+1 numbers as [a, b, B, A, c, C], which puts
                # the triangles on the collapsed face pair and leaves z as the
                # tensor direction. The triangles are wound clockwise in x-y
                # because that is what makes the local axes right-handed:
                # anticlockwise gives a valid-looking element with a negative
                # Jacobian, which Nektar does not reject.
                for t in ((0, 2, 1), (0, 3, 2)):
                    a, b, cc = (c[0][n] for n in t)
                    A, B, C = (c[1][n] for n in t)
                    v = [a, b, B, A, cc, C]
                    prismverts.append(v)
                    prisms.append([face([v[n] for n in f])
                                   for f in PRISMFACES])
            else:
                v = c[0] + c[1]
                hexverts.append(v)
                hexes.append([face([v[n] for n in f]) for f in HEXFACES])

for vs in faceverts:
    for a in range(len(vs)):
        edge(vs[a], vs[(a + 1) % len(vs)])


# --------------------------------------------------------- validity check
def handedness(v, i1, i2, i3):
    """Sign of (e1 x e2).e3 for local axes taken from vertex 0."""
    o = verts[v[0]]
    a, b, c = (tuple(verts[v[n]][d] - o[d] for d in range(3))
               for n in (i1, i2, i3))
    cross = (a[1] * b[2] - a[2] * b[1],
             a[2] * b[0] - a[0] * b[2],
             a[0] * b[1] - a[1] * b[0])
    return sum(cross[d] * c[d] for d in range(3))


# The reference prism runs xi1 along v0->v1, xi2 along v0->v3 and xi3 along
# v0->v4; the reference hexahedron xi1,xi2,xi3 along v0->v1, v0->v3, v0->v4.
for tag, cells, axes in (("prism", prismverts, (1, 3, 4)),
                         ("hex", hexverts, (1, 3, 4))):
    for n, v in enumerate(cells):
        assert handedness(v, *axes) > 0, \
            f"{tag} {n} is left-handed: negative Jacobian"


def faceedges(vs):
    return [edge(vs[a], vs[(a + 1) % len(vs)]) for a in range(len(vs))]


# ------------------------------------------------------- boundary composites
def collect(pred):
    return sorted(f for f, vs in enumerate(faceverts)
                  if all(pred(verts[v]) for v in vs))


def near(a, b):
    return abs(a - b) < 1e-10


bnd = {
    1: collect(lambda p: near(p[1], -1.0)),
    2: collect(lambda p: near(p[0], LX)),
    3: collect(lambda p: near(p[1], 1.0)),
    4: collect(lambda p: near(p[0], 0.0)),
    103: collect(lambda p: near(p[2], 0.0)),
    104: collect(lambda p: near(p[2], LZ)),
}


def ranges(ids):
    """Collapse a sorted id list into Nektar's a-b range notation."""
    out, start = [], ids[0]
    prev = start
    for n in ids[1:] + [None]:
        if n != prev + 1:
            out.append(str(start) if start == prev else f"{start}-{prev}")
            start = n
        prev = n
    return ",".join(out)


# ------------------------------------------------------------------- output
# Elements share one id space across types, prisms first.
npri = len(prisms)
w = sys.stdout.write
w('    <GEOMETRY DIM="3" SPACE="3">\n')
w("        <VERTEX>\n")
for n, (x, y, z) in enumerate(verts):
    w(f'            <V ID="{n}">{x:.15g} {y:.15g} {z:.15g}</V>\n')
w("        </VERTEX>\n        <EDGE>\n")
for (a, b), n in sorted(edges.items(), key=lambda kv: kv[1]):
    w(f'            <E ID="{n}">{a} {b}</E>\n')
w("        </EDGE>\n        <FACE>\n")
for n, vs in enumerate(faceverts):
    tag = "T" if len(vs) == 3 else "Q"
    w(f'            <{tag} ID="{n}">'
      f'{" ".join(str(e) for e in faceedges(vs))}</{tag}>\n')
w("        </FACE>\n        <ELEMENT>\n")
for n, fs in enumerate(prisms):
    w(f'            <R ID="{n}">{" ".join(str(f) for f in fs)}</R>\n')
for n, fs in enumerate(hexes):
    w(f'            <H ID="{npri + n}">{" ".join(str(f) for f in fs)}</H>\n')
w("        </ELEMENT>\n        <COMPOSITE>\n")
for cid in sorted(bnd):
    w(f'            <C ID="{cid}"> F[{ranges(bnd[cid])}] </C>\n')
dom = []
if prisms:
    w(f'            <C ID="100"> R[0-{npri - 1}] </C>\n')
    dom.append("100")
if hexes:
    w(f'            <C ID="101"> H[{npri}-{npri + len(hexes) - 1}] </C>\n')
    dom.append("101")
w("        </COMPOSITE>\n")
w(f'        <DOMAIN> C[{",".join(dom)}] </DOMAIN>\n')
w("    </GEOMETRY>\n")

sys.stderr.write(f"{npri} prisms + {len(hexes)} hexes = {npri + len(hexes)} "
                 f"elements, {len(verts)} vertices, {len(edges)} edges, "
                 f"{len(faceverts)} faces\n")
