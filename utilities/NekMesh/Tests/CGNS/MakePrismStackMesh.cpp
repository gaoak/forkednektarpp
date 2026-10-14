///////////////////////////////////////////////////////////////////////////////
//
//  File: MakePrismStackMesh.cpp
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
//  Description: Generates prism_stack.cgns: a boundary-layer-like block of
//  prisms.
//
////////////////////////////////////////////////////////////////////////////////
// An irregular triangulation of a square -- each grid cell split by whichever
// diagonal its parity picks, so that neighbouring blocks of prisms do not all
// present their corners in the same order -- extruded into NZ layers of
// PENTA_6 elements, with the bounding surfaces as named sections.
//
// The point of the mesh is the prism topology rather than the geometry. The
// reader rebuilds each element from its faces instead of from its stored node
// order, so what it produces depends on how the nodes happen to be numbered,
// and InputCGNS::ResetNodes has to renumber them so that every prism in a
// stack collapses onto the same corner of its triangular cross-section. A
// stack several layers deep whose cross-sections are not all oriented alike
// is what makes that pass do any work: the node numbering is permuted by a
// seed so a numbering can be chosen that the pass has to repair.
//
// The permutation is a Fisher-Yates shuffle driven by the linear congruential
// generator below, so a given seed reproduces a given numbering anywhere.
//
// Built by utilities/NekMesh/CMakeLists.txt wherever CGNS is enabled, which
// also adds a test that reruns this and checks the result against the
// fixture committed beside it. To build it by hand instead:
//
//     c++ MakePrismStackMesh.cpp -I<cgns>/include -L<cgns>/lib -lcgns \
//         -o MakePrismStackMesh
//     ./MakePrismStackMesh prism_stack.cgns <seed>
//
#include <cgnslib.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifndef NX
#define NX 6
#endif
#ifndef NY
#define NY 6
#endif
#ifndef NZ
#define NZ 4
#endif

#define NNODE ((NX + 1) * (NY + 1) * (NZ + 1))
#define NTRI (2 * NX * NY)
#define NPRISM (NTRI * NZ)

static void chk(int ier, const char *what)
{
    if (ier)
    {
        printf("FAILED: %s\n", what);
        cg_error_print();
        exit(1);
    }
}

/* One-based node id of grid point (i, j, k), before permutation. */
static cgsize_t gid(int i, int j, int k)
{
    return (cgsize_t)(k * (NX + 1) * (NY + 1) + j * (NX + 1) + i) + 1;
}

int main(int argc, char **argv)
{
    if (argc < 2 || argc > 3)
    {
        printf("usage: %s <out.cgns> [seed]\n", argv[0]);
        return 1;
    }

    const unsigned long seed = (argc == 3) ? strtoul(argv[2], nullptr, 10) : 0;

    /* perm[i] is the zero-based slot the node in slot i moves to. */
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

#define PID(i, j, k) (perm[gid((i), (j), (k)) - 1] + 1)

    double x[NNODE], y[NNODE], z[NNODE];
    for (int k = 0; k <= NZ; ++k)
    {
        for (int j = 0; j <= NY; ++j)
        {
            for (int i = 0; i <= NX; ++i)
            {
                cgsize_t n = PID(i, j, k) - 1;
                x[n]       = (double)i / NX;
                y[n]       = (double)j / NY;
                /* Graded in z, as a boundary layer is. */
                z[n] = 0.25 * ((double)k / NZ) * ((double)k / NZ);
            }
        }
    }

    /* The two triangles of cell (i, j), as grid-point offsets, wound so the
       extruded prism comes out right-handed. */
    static int tri[NTRI][3][2];
    int nt = 0;
    for (int j = 0; j < NY; ++j)
    {
        for (int i = 0; i < NX; ++i)
        {
            if ((i + j) % 2 == 0)
            {
                int a[3][2] = {{0, 0}, {1, 0}, {1, 1}};
                int b[3][2] = {{0, 0}, {1, 1}, {0, 1}};
                memcpy(tri[nt++], a, sizeof(a));
                memcpy(tri[nt++], b, sizeof(b));
            }
            else
            {
                int a[3][2] = {{0, 0}, {1, 0}, {0, 1}};
                int b[3][2] = {{1, 0}, {1, 1}, {0, 1}};
                memcpy(tri[nt++], a, sizeof(a));
                memcpy(tri[nt++], b, sizeof(b));
            }
            tri[nt - 2][0][0] += i;
            tri[nt - 2][1][0] += i;
            tri[nt - 2][2][0] += i;
            tri[nt - 2][0][1] += j;
            tri[nt - 2][1][1] += j;
            tri[nt - 2][2][1] += j;
            tri[nt - 1][0][0] += i;
            tri[nt - 1][1][0] += i;
            tri[nt - 1][2][0] += i;
            tri[nt - 1][0][1] += j;
            tri[nt - 1][1][1] += j;
            tri[nt - 1][2][1] += j;
        }
    }

    int fn, B, Z, S;
    chk(cg_open(argv[1], CG_MODE_WRITE, &fn), "open");
    chk(cg_base_write(fn, "Base", 3, 3, &B), "base_write");

    cgsize_t size[3] = {NNODE, NPRISM, 0};
    chk(cg_zone_write(fn, B, "fluid", size, Unstructured, &Z), "zone_write");

    int C;
    chk(cg_coord_write(fn, B, Z, RealDouble, "CoordinateX", x, &C), "x");
    chk(cg_coord_write(fn, B, Z, RealDouble, "CoordinateY", y, &C), "y");
    chk(cg_coord_write(fn, B, Z, RealDouble, "CoordinateZ", z, &C), "z");

    /* Volume: NZ layers of prisms over every triangle.

       Each prism's corners are cyclically rotated by an amount the seed picks.
       That is the same element written a different way round, and a mesher is
       free to write it either way, but it moves which face the reader meets
       first -- and the reader anchors the element it rebuilds on the first
       triangular and first quadrilateral face it finds. Without this the
       stacked prisms all present their corners alike and the alignment pass
       has nothing to repair. */
    unsigned long rstate = seed ? seed : 1;
    static cgsize_t penta[NPRISM * 6];
    cgsize_t *p = penta;
    for (int k = 0; k < NZ; ++k)
    {
        for (int t = 0; t < NTRI; ++t)
        {
            rstate  = rstate * 6364136223846793005UL + 1442695040888963407UL;
            int rot = (int)((rstate >> 33) % 3);

            for (int v = 0; v < 3; ++v)
            {
                *p++ =
                    PID(tri[t][(v + rot) % 3][0], tri[t][(v + rot) % 3][1], k);
            }
            for (int v = 0; v < 3; ++v)
            {
                *p++ = PID(tri[t][(v + rot) % 3][0], tri[t][(v + rot) % 3][1],
                           k + 1);
            }
        }
    }
    chk(cg_section_write(fn, B, Z, "volume", PENTA_6, 1, NPRISM, 0, penta, &S),
        "penta");

    cgsize_t next = NPRISM + 1;

    /* The two triangulated surfaces. */
    static cgsize_t surf[NTRI * 3];
    const char *tnames[2] = {"wall", "farfield"};
    const int tlayer[2]   = {0, NZ};
    for (int s = 0; s < 2; ++s)
    {
        p = surf;
        for (int t = 0; t < NTRI; ++t)
        {
            for (int v = 0; v < 3; ++v)
            {
                *p++ = PID(tri[t][v][0], tri[t][v][1], tlayer[s]);
            }
        }
        chk(cg_section_write(fn, B, Z, tnames[s], TRI_3, next, next + NTRI - 1,
                             0, surf, &S),
            "tri");
        next += NTRI;
    }

    /* The four sides, each a strip of quads. */
    const char *qnames[4] = {"ymin", "xmax", "ymax", "xmin"};
    static cgsize_t quad[4 * (NX > NY ? NX : NY) * NZ * 4];
    for (int s = 0; s < 4; ++s)
    {
        p       = quad;
        int nq  = 0;
        int lim = (s % 2 == 0) ? NX : NY;
        for (int k = 0; k < NZ; ++k)
        {
            for (int m = 0; m < lim; ++m)
            {
                int i0, j0, i1, j1;
                switch (s)
                {
                    case 0:
                        i0 = m;
                        j0 = 0;
                        i1 = m + 1;
                        j1 = 0;
                        break;
                    case 1:
                        i0 = NX;
                        j0 = m;
                        i1 = NX;
                        j1 = m + 1;
                        break;
                    case 2:
                        i0 = m + 1;
                        j0 = NY;
                        i1 = m;
                        j1 = NY;
                        break;
                    default:
                        i0 = 0;
                        j0 = m + 1;
                        i1 = 0;
                        j1 = m;
                        break;
                }
                *p++ = PID(i0, j0, k);
                *p++ = PID(i1, j1, k);
                *p++ = PID(i1, j1, k + 1);
                *p++ = PID(i0, j0, k + 1);
                ++nq;
            }
        }
        chk(cg_section_write(fn, B, Z, qnames[s], QUAD_4, next, next + nq - 1,
                             0, quad, &S),
            "quad");
        next += nq;
    }

    chk(cg_close(fn), "close");
    printf("wrote %s: %d nodes, %d prisms (seed %lu)\n", argv[1], NNODE, NPRISM,
           seed);
    return 0;
}
