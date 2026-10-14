///////////////////////////////////////////////////////////////////////////////
//
//  File: MakeAngledPrismLayers.cpp
//
//  For more information, please see: http://www.nektar.info/
//
//  The MIT License
//
//  Copyright (c) 2006 Division of Applied Mathematics, Brown University (USA),
//  Department of Aeronautics, Imperial College London (UK), and Scientific
//  Computing and Imaging Institute, University of Utah (USA).
//
//  Permission is hereby granted, free of charge, to any person obtaining a
//  copy of this software and associated documentation files (the "Software"),
//  to deal in the Software without restriction, including without limitation
//  the rights to use, copy, modify, merge, publish, distribute, sublicense,
//  and/or sell copies of the Software, and to permit persons to whom the
//  Software is furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included
//  in all copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
//  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//  DEALINGS IN THE SOFTWARE.
//
//  Description: Generates angled_prism_layers.cgns, two stacks of prisms
//  whose layers run at right angles to each other.
//
////////////////////////////////////////////////////////////////////////////////
// This is the smallest mesh that puts the reader's prism numbering in the
// position a boundary layer round a fillet does. A prism is a triangle
// extruded along three corner edges, and each of its quadrilateral faces
// holds one opposite pair of them. Two prisms sharing a quadrilateral face
// may take *different* pairs of its edges as their own extrusion edges, and
// then one prism's extrusion edge is an edge of the other's cross-section.
//
// Here stack A extrudes along z and stack B along y, and they share the
// square at x = 1. Each stack is two prisms deep, so that the triangular
// faces between them are inside the mesh and have to agree:
//
// z                A extrudes in z  (edges of the shared face
// |                                  running up the page)
// |   +--------+   B extrudes in y  (edges running into it)
// |  /|       /|
// | / |      / |   The shared face therefore has A's extrusion edges
// |/  |  B  /  |   on one pair of sides and B's on the other.
// +---+----+   |
// | A |    |   +
// |   +----|--/
// |  /     | /
// | /      |/
// +--------+------ x
//
// Joining the ends of every extrusion edge, which is how InputCGNS::ResetNodes
// groups nodes into the lines running through a stack, then pulls all four
// corners of the shared face onto one line: A says two of them are joined,
// B says the other two are, and between them they close the loop. Each prism
// is left with two corners of its cross-section on the same line, so the
// order of the lines cannot say which of those two is the higher, and the two
// ends of the prism can disagree. They must not: that corner is the one
// OrientPrism collapses onto, and the Duffy transformation has to meet across
// the shared faces.
//
// What settles it is the third corner, which is the highest at both ends as
// long as its line is numbered after theirs -- an ordering constraint between
// two lines rather than something left to the order the lines happen to come
// in. The node numbering is permuted by a seed so that a numbering can be
// chosen which the old order got wrong.
//
// Seed 4 is the one committed. Without that constraint the mesh is refused
// with
//
// Orientation of triangular face (id = ...) is inconsistent with face 1 of
// prism element (id = ...) since Dir2 is aligned with Dir1
//
// Built by utilities/NekMesh/CMakeLists.txt wherever CGNS is enabled, which
// also adds a test that reruns this and checks the result against the
// fixture committed beside it. To build it by hand instead:
//
//     c++ MakeAngledPrismLayers.cpp -I<cgns>/include -L<cgns>/lib -lcgns \
//         -o MakeAngledPrismLayers
//     ./MakeAngledPrismLayers angled_prism_layers.cgns <seed>
//
#include <cgnslib.h>
#include <cstdio>
#include <cstdlib>

#define NNODE 14

static void chk(int ier, const char *what)
{
    if (ier)
    {
        printf("FAILED: %s\n", what);
        cg_error_print();
        exit(1);
    }
}

int main(int argc, char **argv)
{
    if (argc < 2 || argc > 3)
    {
        printf("usage: %s <out.cgns> [seed]\n", argv[0]);
        return 1;
    }

    const unsigned long seed = (argc == 3) ? strtoul(argv[2], nullptr, 10) : 0;

    /*  0-2  the far edge of stack A, at x = 0
        3-8  the shared face of the two stacks and what rises from it
        9-11 the far edge of stack B, at x = 2
        12-13 the top of stack B                                        */
    const double px[NNODE] = {0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1};
    const double py[NNODE] = {0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 2, 2, 2};
    const double pz[NNODE] = {0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 0, 0, 0, 1};

    cgsize_t perm[NNODE];
    for (int i = 0; i < NNODE; ++i)
    {
        perm[i] = i;
    }

    if (seed)
    {
        unsigned long state = seed;
        for (int i = NNODE - 1; i > 0; --i)
        {
            state      = state * 6364136223846793005UL + 1442695040888963407UL;
            int j      = (int)((state >> 33) % (unsigned long)(i + 1));
            cgsize_t t = perm[i];
            perm[i]    = perm[j];
            perm[j]    = t;
        }
    }

#define P(i) (perm[i] + 1)

    double x[NNODE], y[NNODE], z[NNODE];
    for (int i = 0; i < NNODE; ++i)
    {
        x[perm[i]] = px[i];
        y[perm[i]] = py[i];
        z[perm[i]] = pz[i];
    }

    int fn, B, Z, C, S;
    chk(cg_open(argv[1], CG_MODE_WRITE, &fn), "open");
    chk(cg_base_write(fn, "Base", 3, 3, &B), "base_write");

    cgsize_t size[3] = {NNODE, 4, 0};
    chk(cg_zone_write(fn, B, "fluid", size, Unstructured, &Z), "zone_write");
    chk(cg_coord_write(fn, B, Z, RealDouble, "CoordinateX", x, &C), "x");
    chk(cg_coord_write(fn, B, Z, RealDouble, "CoordinateY", y, &C), "y");
    chk(cg_coord_write(fn, B, Z, RealDouble, "CoordinateZ", z, &C), "z");

    /* A1, A2 rise in z; B1, B2 run out in y. A1 and B1 share the square
       3-6-7-4, and its two pairs of opposite edges are the extrusion edges
       of one stack and of the other. */
    cgsize_t penta[24] = {P(0),  P(3), P(6), P(1),  P(4), P(7),  P(1),  P(4),
                          P(7),  P(2), P(5), P(8),  P(3), P(9),  P(4),  P(6),
                          P(10), P(7), P(6), P(10), P(7), P(12), P(11), P(13)};
    chk(cg_section_write(fn, B, Z, "volume", PENTA_6, 1, 4, 0, penta, &S),
        "penta");

    /* The ends of the two stacks. */
    cgsize_t tri[12] = {P(0), P(6), P(3), P(2),  P(5),  P(8),
                        P(3), P(9), P(4), P(12), P(11), P(13)};
    chk(cg_section_write(fn, B, Z, "ends", TRI_3, 5, 8, 0, tri, &S), "tri");

    /* Their sides, less the square the two stacks share. */
    cgsize_t quad[40] = {P(0),  P(3), P(4),  P(1),  P(6), P(0),  P(1),  P(7),
                         P(1),  P(4), P(5),  P(2),  P(4), P(7),  P(8),  P(5),
                         P(7),  P(1), P(2),  P(8),  P(3), P(9),  P(10), P(6),
                         P(9),  P(4), P(7),  P(10), P(6), P(10), P(11), P(12),
                         P(10), P(7), P(13), P(11), P(7), P(6),  P(12), P(13)};
    chk(cg_section_write(fn, B, Z, "sides", QUAD_4, 9, 18, 0, quad, &S),
        "quad");

    chk(cg_close(fn), "close");
    printf("wrote %s (seed %lu)\n", argv[1], seed);
    return 0;
}
