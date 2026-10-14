///////////////////////////////////////////////////////////////////////////////
//
//  File: MakePeriodicConnMesh.cpp
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
//  Description: Generates periodic_conn.cgns from
//  linear_tet_prism_pyra_mesh.cgns.
//
////////////////////////////////////////////////////////////////////////////////
// Adds a pair of periodic GridConnectivity_t nodes, which is how a mesher
// states that two surfaces are periodic images of each other and which none of
// the other test meshes here contains. The faces are addressed at FaceCenter,
// i.e. by element number, as exporters write them, and it is reading them as
// anything else that this fixture is here to catch.
//
// The two sides are the outlet patch of the source mesh and an equal number of
// faces taken from its inlet. They are not true periodic images of one another
// and nothing here claims they are; CGNS requires only that an Abutting1to1
// pair name the same number of faces on each side. Taking fewer faces than the
// inlet holds is deliberate, so that the inlet keeps a remainder of its own:
// the three counts that result can only come out right if the declared sets
// were read as the elements they name.
//
// Built by utilities/NekMesh/CMakeLists.txt wherever CGNS is enabled, which
// also adds a test that reruns this and checks the result against the
// fixture committed beside it. To build it by hand instead:
//
//     c++ MakePeriodicConnMesh.cpp -I<cgns>/include -L<cgns>/lib -lcgns \
//         -o MakePeriodicConnMesh
//     cp linear_tet_prism_pyra_mesh.cgns periodic_conn.cgns
//     ./MakePeriodicConnMesh periodic_conn.cgns
//
#include <cgnslib.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static void chk(int ier, const char *what)
{
    if (ier)
    {
        printf("FAILED: %s\n", what);
        cg_error_print();
        exit(1);
    }
}

int main(int, char **argv)
{
    int fn, ns;
    char zname[33];
    cgsize_t sz[9];

    chk(cg_open(argv[1], CG_MODE_MODIFY, &fn), "open");
    chk(cg_zone_read(fn, 1, 1, zname, sz), "zone read");
    chk(cg_nsections(fn, 1, 1, &ns), "nsections");

    /* Take the two sides from the boundary conditions that already name them.
     */
    int nbc;
    chk(cg_nbocos(fn, 1, 1, &nbc), "nbocos");

    std::vector<cgsize_t> side[2];
    cgsize_t count[2] = {0, 0};
    for (int I = 1; I <= nbc; ++I)
    {
        char bn[33];
        CGNS_ENUMT(BCType_t) bt;
        CGNS_ENUMT(PointSetType_t) ps;
        cgsize_t np, nls;
        int ni[3], nd;
        CGNS_ENUMT(DataType_t) ndt;
        chk(cg_boco_info(fn, 1, 1, I, bn, &bt, &ps, &np, ni, &nls, &ndt, &nd),
            "boco info");
        int w = strstr(bn, "Inlet") ? 0 : (strstr(bn, "Outlet") ? 1 : -1);
        if (w < 0)
        {
            continue;
        }
        side[w].resize(np);
        count[w] = np;
        chk(cg_boco_read(fn, 1, 1, I, side[w].data(), nullptr), "boco read");
        printf("side %d from '%s': %ld faces\n", w, bn, (long)np);
    }
    if (side[0].empty() || side[1].empty())
    {
        printf("inlet/outlet not found\n");
        return 1;
    }

    /* Abutting1to1 pairs face for face, so both sides are the same length. */
    cgsize_t *a = side[0].data(), *b = side[1].data();
    cgsize_t n = count[0] < count[1] ? count[0] : count[1];
    printf("zone '%s': periodic pair declared over %ld + %ld faces, leaving "
           "%ld of the inlet and %ld of the outlet to their conditions\n",
           zname, (long)n, (long)n, (long)(count[0] - n), (long)(count[1] - n));

    float centre[3] = {0, 0, 0}, angle[3] = {0, 0, 0};
    float fwd[3] = {0, 0, 1}, rev[3] = {0, 0, -1};

    int J;
    chk(cg_conn_write(fn, 1, 1, "periodic_a", CGNS_ENUMV(FaceCenter),
                      CGNS_ENUMV(Abutting1to1), CGNS_ENUMV(PointList), n, a,
                      zname, CGNS_ENUMV(Unstructured),
                      CGNS_ENUMV(PointListDonor), CGNS_ENUMV(LongInteger), n, b,
                      &J),
        "conn a");
    chk(cg_conn_periodic_write(fn, 1, 1, J, centre, angle, fwd), "periodic a");

    chk(cg_conn_write(fn, 1, 1, "periodic_b", CGNS_ENUMV(FaceCenter),
                      CGNS_ENUMV(Abutting1to1), CGNS_ENUMV(PointList), n, b,
                      zname, CGNS_ENUMV(Unstructured),
                      CGNS_ENUMV(PointListDonor), CGNS_ENUMV(LongInteger), n, a,
                      &J),
        "conn b");
    chk(cg_conn_periodic_write(fn, 1, 1, J, centre, angle, rev), "periodic b");

    chk(cg_close(fn), "close");
    printf("done\n");
    return 0;
}
