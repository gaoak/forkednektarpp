///////////////////////////////////////////////////////////////////////////////
//
//  File: MakeTwoZoneMesh.cpp
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
//  Description: Generates two_zone.cgns and its single-zone equivalent
//  one_zone.cgns.
//
////////////////////////////////////////////////////////////////////////////////
// A box of hexahedra, written twice: once as one zone, and once split down the
// middle into two zones that abut across the cut. The split copy duplicates
// the vertices on the cut, one set per zone, writes the cut as a section in
// each, and declares the two copies to abut with an Abutting1to1
// GridConnectivity_t carrying a donor list -- which is how a mesher writes a
// volume and the boundary layer grown on it when they are given separate part
// IDs.
//
// The pair exists so that a test can assert the two give the same mesh. The
// single-zone file is the answer; the two-zone file is the question.
//
// Built by utilities/NekMesh/CMakeLists.txt wherever CGNS is enabled, which
// also adds a test that reruns this and checks the result against the
// fixture committed beside it. To build it by hand instead:
//
//     c++ MakeTwoZoneMesh.cpp -I<cgns>/include -L<cgns>/lib -lcgns \
//         -o MakeTwoZoneMesh
//     ./MakeTwoZoneMesh one_zone.cgns two_zone.cgns
//
#include <cgnslib.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

/*
 * Two cells thick in z, so that hexahedra stack and some of them meet a shared
 * face another element has already built. Reconstructing an element from its
 * faces used to leave those inside out, the whole upper layer at negative
 * Jacobian, so the stacking is part of what this fixture is for.
 */
#define NX 4
#define NY 2
#define NZ 2

static void chk(int ier, const char *what)
{
    if (ier)
    {
        printf("FAILED: %s\n", what);
        cg_error_print();
        exit(1);
    }
}

/* Vertex index within a box of (ix0..ix1) x (0..NY) x (0..NZ) cells. */
static int vid(int ix0, int ix1, int i, int j, int k)
{
    int nx = ix1 - ix0 + 1;
    return ((k * (NY + 1)) + j) * nx + (i - ix0);
}

/* The six faces of a hex, as corner offsets, outward normals consistent. */
static const int faceOf[6][4][3] = {
    {{0, 0, 0}, {0, 1, 0}, {0, 1, 1}, {0, 0, 1}}, /* x lo */
    {{1, 0, 0}, {1, 0, 1}, {1, 1, 1}, {1, 1, 0}}, /* x hi */
    {{0, 0, 0}, {0, 0, 1}, {1, 0, 1}, {1, 0, 0}}, /* y lo */
    {{0, 1, 0}, {1, 1, 0}, {1, 1, 1}, {0, 1, 1}}, /* y hi */
    {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}}, /* z lo */
    {{0, 0, 1}, {0, 1, 1}, {1, 1, 1}, {1, 0, 1}}, /* z hi */
};
static const char *faceName[6] = {"xmin", "xmax", "ymin",
                                  "ymax", "zmin", "zmax"};

/* Write one zone covering cells ix0..ix1-1. cutLo/cutHi name the cut planes
   that are interfaces rather than boundaries of the model. */
static void writeZone(int fn, int B, const char *zname, int ix0, int ix1,
                      int cutLo, int cutHi, cgsize_t *ifaceRange)
{
    int nx = ix1 - ix0, nv = (nx + 1) * (NY + 1) * (NZ + 1), Z, S, F;
    cgsize_t sizes[3] = {nv, nx * NY * NZ, 0};
    chk(cg_zone_write(fn, B, zname, sizes, CGNS_ENUMV(Unstructured), &Z),
        "zone");

    std::vector<double> x(nv), y(nv), z(nv);
    for (int k = 0; k <= NZ; ++k)
    {
        for (int j = 0; j <= NY; ++j)
        {
            for (int i = ix0; i <= ix1; ++i)
            {
                int v = vid(ix0, ix1, i, j, k);
                x[v]  = i;
                y[v]  = j;
                z[v]  = k;
            }
        }
    }
    chk(cg_coord_write(fn, B, Z, RealDouble, "CoordinateX", x.data(), &F), "x");
    chk(cg_coord_write(fn, B, Z, RealDouble, "CoordinateY", y.data(), &F), "y");
    chk(cg_coord_write(fn, B, Z, RealDouble, "CoordinateZ", z.data(), &F), "z");

    /* Boundary quads first, as exporters write them, volume after. */
    cgsize_t next = 1;
    for (int f = 0; f < 6; ++f)
    {
        std::vector<cgsize_t> q(4 * nx * NY * NZ);
        cgsize_t n = 0;
        for (int k = 0; k < NZ; ++k)
        {
            for (int j = 0; j < NY; ++j)
            {
                for (int i = ix0; i < ix1; ++i)
                {
                    if (f == 0 && i != ix0)
                    {
                        continue;
                    }
                    if (f == 1 && i != ix1 - 1)
                    {
                        continue;
                    }
                    if (f == 2 && j != 0)
                    {
                        continue;
                    }
                    if (f == 3 && j != NY - 1)
                    {
                        continue;
                    }
                    if (f == 4 && k != 0)
                    {
                        continue;
                    }
                    if (f == 5 && k != NZ - 1)
                    {
                        continue;
                    }
                    /* a cut plane is not a boundary of the model */
                    if (f == 0 && cutLo)
                    {
                        continue;
                    }
                    if (f == 1 && cutHi)
                    {
                        continue;
                    }
                    for (int c = 0; c < 4; ++c)
                    {
                        q[n++] = vid(ix0, ix1, i + faceOf[f][c][0],
                                     j + faceOf[f][c][1], k + faceOf[f][c][2]) +
                                 1;
                    }
                }
            }
        }
        if (!n)
        {
            continue;
        }
        cgsize_t cnt = n / 4;
        chk(cg_section_write(fn, B, Z, faceName[f], CGNS_ENUMV(QUAD_4), next,
                             next + cnt - 1, 0, q.data(), &S),
            "bnd section");

        /* A boundary condition over the section, pointing at a family that
           carries the bare part name, which is where an exporter puts it. The
           condition's own name is suffixed per zone, so the two zones' halves
           of one face only come together under the family. */
        std::vector<cgsize_t> pl(cnt);
        for (cgsize_t e = 0; e < cnt; ++e)
        {
            pl[e] = next + e;
        }
        char bcname[64];
        snprintf(bcname, sizeof(bcname), "%s_%d", faceName[f], Z);
        int BC;
        chk(cg_boco_write(fn, B, Z, bcname, CGNS_ENUMV(BCTypeUserDefined),
                          CGNS_ENUMV(PointList), cnt, pl.data(), &BC),
            "boco");
        chk(cg_boco_gridlocation_write(fn, B, Z, BC, CGNS_ENUMV(FaceCenter)),
            "gridloc");
        chk(cg_goto(fn, B, "Zone_t", Z, "ZoneBC_t", 1, "BC_t", BC, "end"),
            "goto bc");
        chk(cg_famname_write(faceName[f]), "famname");
        next += cnt;
    }

    /* The cut, written as its own section so a connectivity can name it. */
    ifaceRange[0] = ifaceRange[1] = 0;
    if (cutLo || cutHi)
    {
        int f = cutLo ? 0 : 1, i = cutLo ? ix0 : ix1 - 1;
        std::vector<cgsize_t> q(4 * NY * NZ);
        cgsize_t n = 0;
        for (int k = 0; k < NZ; ++k)
        {
            for (int j = 0; j < NY; ++j)
            {
                for (int c = 0; c < 4; ++c)
                {
                    q[n++] = vid(ix0, ix1, i + faceOf[f][c][0],
                                 j + faceOf[f][c][1], k + faceOf[f][c][2]) +
                             1;
                }
            }
        }
        chk(cg_section_write(fn, B, Z, "Interface", CGNS_ENUMV(QUAD_4), next,
                             next + NY * NZ - 1, 0, q.data(), &S),
            "iface section");
        ifaceRange[0] = next;
        ifaceRange[1] = next + NY * NZ - 1;
        next += NY * NZ;
    }

    std::vector<cgsize_t> h(8 * nx * NY * NZ);
    cgsize_t n = 0;
    for (int k = 0; k < NZ; ++k)
    {
        for (int j = 0; j < NY; ++j)
        {
            for (int i = ix0; i < ix1; ++i)
            {
                h[n++] = vid(ix0, ix1, i, j, k) + 1;
                h[n++] = vid(ix0, ix1, i + 1, j, k) + 1;
                h[n++] = vid(ix0, ix1, i + 1, j + 1, k) + 1;
                h[n++] = vid(ix0, ix1, i, j + 1, k) + 1;
                h[n++] = vid(ix0, ix1, i, j, k + 1) + 1;
                h[n++] = vid(ix0, ix1, i + 1, j, k + 1) + 1;
                h[n++] = vid(ix0, ix1, i + 1, j + 1, k + 1) + 1;
                h[n++] = vid(ix0, ix1, i, j + 1, k + 1) + 1;
            }
        }
    }
    chk(cg_section_write(fn, B, Z, "VolumeElements", CGNS_ENUMV(HEXA_8), next,
                         next + nx * NY * NZ - 1, 0, h.data(), &S),
        "vol section");
}

int main(int, char **argv)
{
    int fn, B;
    cgsize_t dummy[2], ifA[2], ifB[2];

    /* one zone */
    chk(cg_open(argv[1], CG_MODE_WRITE, &fn), "open one");
    chk(cg_base_write(fn, "Base", 3, 3, &B), "base one");
    for (int f = 0; f < 6; ++f)
    {
        int F;
        chk(cg_family_write(fn, B, faceName[f], &F), "family");
    }
    writeZone(fn, B, "fluid", 0, NX, 0, 0, dummy);
    chk(cg_close(fn), "close one");
    printf("%s: one zone, %d hexes\n", argv[1], NX * NY * NZ);

    /* two zones, cut at NX/2 */
    chk(cg_open(argv[2], CG_MODE_WRITE, &fn), "open two");
    chk(cg_base_write(fn, "Base", 3, 3, &B), "base two");
    for (int f = 0; f < 6; ++f)
    {
        int F;
        chk(cg_family_write(fn, B, faceName[f], &F), "family");
    }
    writeZone(fn, B, "lower", 0, NX / 2, 0, 1, ifA);
    writeZone(fn, B, "upper", NX / 2, NX, 1, 0, ifB);

    /* Declare the two copies of the cut to abut, face for face. */
    cgsize_t n = NY * NZ;
    std::vector<cgsize_t> a(n), b(n);
    for (cgsize_t i = 0; i < n; ++i)
    {
        a[i] = ifA[0] + i;
        b[i] = ifB[0] + i;
    }
    int I;
    chk(cg_conn_write(fn, B, 1, "Conn_1_2", CGNS_ENUMV(FaceCenter),
                      CGNS_ENUMV(Abutting1to1), CGNS_ENUMV(PointList), n,
                      a.data(), "upper", CGNS_ENUMV(Unstructured),
                      CGNS_ENUMV(PointListDonor), CGNS_ENUMV(LongInteger), n,
                      b.data(), &I),
        "conn 1->2");
    chk(cg_conn_write(fn, B, 2, "Conn_2_1", CGNS_ENUMV(FaceCenter),
                      CGNS_ENUMV(Abutting1to1), CGNS_ENUMV(PointList), n,
                      b.data(), "lower", CGNS_ENUMV(Unstructured),
                      CGNS_ENUMV(PointListDonor), CGNS_ENUMV(LongInteger), n,
                      a.data(), &I),
        "conn 2->1");
    chk(cg_close(fn), "close two");
    printf("%s: two zones, %d + %d hexes, cut of %ld faces duplicated\n",
           argv[2], NX / 2 * NY * NZ, NX / 2 * NY * NZ, (long)n);
    return 0;
}
