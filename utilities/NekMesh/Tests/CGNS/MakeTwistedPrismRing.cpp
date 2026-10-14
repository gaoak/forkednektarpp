///////////////////////////////////////////////////////////////////////////////
//
//  File: MakeTwistedPrismRing.cpp
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
//  Description: Generates twisted_prism_ring.cgns and
//  twisted_prism_ring_p2.cgns, a ring of prisms closed back onto itself with a
//  third of a turn in it.
//
////////////////////////////////////////////////////////////////////////////////
// A prism collapses onto one corner of its triangular cross-section -- that
// is where the Duffy transformation of the modified C^0 basis sends it -- and
// the two ends of the prism have to agree on which corner that is, as do two
// prisms meeting at a triangular face. Walking round this ring the choice is
// carried from one prism to the next, and the turn means it comes back one
// corner along from where it started. No numbering of the mesh can mend that,
// however it is done: the mesh simply has no representation as prisms.
//
// InputCGNS::ResetNodes therefore replaces the prism it cannot orient with
// the elements got by coning each of its faces to a new node at its centre --
// a tetrahedron on each of the two triangles, a pyramid on each of the three
// quadrilaterals. Splitting it into three tetrahedra instead would be
// smaller, but a tetrahedron cannot present the quadrilateral faces it shares
// with its neighbours; coning keeps all five faces exactly as they were, so
// the neighbours do not notice and the mesh stays conformal.
//
// The ring is written at first and at second order. The second-order one is
// the sharper test: above first order the reader takes an element's nodes as
// it finds them rather than rebuilding it from its faces, and a pyramid is
// never reoriented, so the new elements have to be wound correctly when they
// are made rather than put right afterwards.
//
// Built by utilities/NekMesh/CMakeLists.txt wherever CGNS is enabled, which
// also adds a test that reruns this and checks the result against the
// fixture committed beside it. To build it by hand instead:
//
//     c++ MakeTwistedPrismRing.cpp -I<cgns>/include -L<cgns>/lib -lcgns -lm \
//         -o MakeTwistedPrismRing
//     ./MakeTwistedPrismRing twisted_prism_ring.cgns 1
//     ./MakeTwistedPrismRing twisted_prism_ring_p2.cgns 2
//
#include <cgnslib.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#define N 12
#define NV (3 * N)
#define NNODE (9 * N)

/* vertex (k,j); mid of section k's triangle edge j; mid of element k's
   extrusion edge at corner j */
#define V(k, j) ((k) * 3 + (j))
#define TM(k, j) (NV + (k) * 3 + (j))
#define AM(k, j) (NV + 3 * N + (k) * 3 + (j))

static double X[NNODE], Y[NNODE], Z[NNODE];

/* M_PI is a POSIX extension rather than standard C++, and this is built
   without compiler extensions, so the constant is spelt out here. */
static constexpr double kPi = 3.14159265358979323846;

static void chk(int ier, const char *w)
{
    if (ier)
    {
        printf("FAILED: %s\n", w);
        cg_error_print();
        exit(1);
    }
}

static void mid(int m, int a, int b)
{
    X[m] = 0.5 * (X[a] + X[b]);
    Y[m] = 0.5 * (Y[a] + Y[b]);
    Z[m] = 0.5 * (Z[a] + Z[b]);
}

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        printf("usage: %s <out.cgns> <order 1|2>\n", argv[0]);
        return 1;
    }

    const int order = atoi(argv[2]);
    const double R = 3.0, r = 0.7;

    for (int k = 0; k < N; ++k)
    {
        const double th = 2.0 * kPi * k / N;
        const double cx = R * std::cos(th), cy = R * std::sin(th);
        const double ux = std::cos(th), uy = std::sin(th);

        for (int j = 0; j < 3; ++j)
        {
            /* the cross-section turns through a third over the whole ring */
            const double a = 2.0 * kPi * j / 3.0 + (2.0 * kPi / 3.0) * k / N;
            const int n    = V(k, j);
            X[n]           = cx + r * std::cos(a) * ux;
            Y[n]           = cy + r * std::cos(a) * uy;
            Z[n]           = r * std::sin(a);
        }
    }

    for (int k = 0; k < N; ++k)
    {
        const int nx = (k + 1) % N, sh = (k == N - 1) ? 1 : 0;
        for (int j = 0; j < 3; ++j)
        {
            mid(TM(k, j), V(k, j), V(k, (j + 1) % 3));
            mid(AM(k, j), V(k, j), V(nx, (j + sh) % 3));
        }
    }

    const int nnode = (order == 2) ? NNODE : NV;

    int fn, B, Z_, C, S;
    chk(cg_open(argv[1], CG_MODE_WRITE, &fn), "open");
    chk(cg_base_write(fn, "Base", 3, 3, &B), "base");

    cgsize_t size[3] = {nnode, N, 0};
    chk(cg_zone_write(fn, B, "fluid", size, Unstructured, &Z_), "zone");
    chk(cg_coord_write(fn, B, Z_, RealDouble, "CoordinateX", X, &C), "x");
    chk(cg_coord_write(fn, B, Z_, RealDouble, "CoordinateY", Y, &C), "y");
    chk(cg_coord_write(fn, B, Z_, RealDouble, "CoordinateZ", Z, &C), "z");

    /* The corners run the other way round from the order they were built in,
       so that each prism comes out the right way out. */
    const int np = (order == 2) ? 15 : 6;
    static cgsize_t p[N * 15];

    for (int k = 0; k < N; ++k)
    {
        const int nx = (k + 1) % N, sh = (k == N - 1) ? 1 : 0;
        cgsize_t *e = p + k * np;

        for (int j = 0; j < 3; ++j)
        {
            e[j]     = V(k, (3 - j) % 3) + 1;
            e[3 + j] = V(nx, ((3 - j) % 3 + sh) % 3) + 1;
        }

        if (order == 2)
        {
            e[6]  = TM(k, 2) + 1;
            e[7]  = TM(k, 1) + 1;
            e[8]  = TM(k, 0) + 1;
            e[9]  = AM(k, 0) + 1;
            e[10] = AM(k, 2) + 1;
            e[11] = AM(k, 1) + 1;
            e[12] = TM(nx, (2 + sh) % 3) + 1;
            e[13] = TM(nx, (1 + sh) % 3) + 1;
            e[14] = TM(nx, (0 + sh) % 3) + 1;
        }
    }

    chk(cg_section_write(fn, B, Z_, "volume", (order == 2) ? PENTA_15 : PENTA_6,
                         1, N, 0, p, &S),
        "penta");

    /* Every quadrilateral face is on the outside of the ring; the triangular
       ones are all interior, which is what makes the ring close. */
    const int nq = (order == 2) ? 8 : 4;
    static cgsize_t q[N * 3 * 8];
    cgsize_t *w = q;

    for (int k = 0; k < N; ++k)
    {
        const int nx = (k + 1) % N, sh = (k == N - 1) ? 1 : 0;
        for (int j = 0; j < 3; ++j)
        {
            const int j2 = (j + 1) % 3;
            *w++         = V(k, j) + 1;
            *w++         = V(k, j2) + 1;
            *w++         = V(nx, (j2 + sh) % 3) + 1;
            *w++         = V(nx, (j + sh) % 3) + 1;

            if (order == 2)
            {
                *w++ = TM(k, j) + 1;
                *w++ = AM(k, j2) + 1;
                *w++ = TM(nx, (j + sh) % 3) + 1;
                *w++ = AM(k, j) + 1;
            }
        }
    }

    chk(cg_section_write(fn, B, Z_, "skin", (order == 2) ? QUAD_8 : QUAD_4,
                         N + 1, N + 3 * N, 0, q, &S),
        "quad");
    chk(cg_close(fn), "close");

    printf("wrote %s: %d nodes, %d prisms, order %d\n", argv[1], nnode, N,
           order);
    (void)nq;
    return 0;
}
