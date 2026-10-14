///////////////////////////////////////////////////////////////////////////////
//
//  File: MakeMixedPidMesh.cpp
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
//  Description: Generates mixed_pid_sections.cgns from
//  linear_tet_prism_pyra_mesh.cgns.
//
////////////////////////////////////////////////////////////////////////////////
// Rewrites a mesh in the shape a per-part-ID CGNS export has, which is how
// ANSA and others write one and which the plain test meshes here do not
// exercise:
//
// - one Elements_t section per part ID, all of type MIXED, including the
// volume;
// - a Family_t per part carrying the bare part name;
// - a BC_t named "<part>_1" pointing at that family, addressing its faces
// as a FaceCenter PointList;
// - one surface part left with neither a boundary condition nor a family,
// as exporters leave their periodic patches.
//
// The geometry is entirely that of the input mesh; only the organisation
// changes. Committed so the fixture can be regenerated or extended rather
// than only inspected.
//
// Built by utilities/NekMesh/CMakeLists.txt wherever CGNS is enabled, which
// also adds a test that reruns this and checks the result against the
// fixture committed beside it. To build it by hand instead:
//
//     c++ MakeMixedPidMesh.cpp -I<cgns>/include -L<cgns>/lib -lcgns \
//         -o MakeMixedPidMesh
//     ./MakeMixedPidMesh linear_tet_prism_pyra_mesh.cgns
//     mixed_pid_sections.cgns
//
#include <cgnslib.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define MAXE 16

struct Elem
{
    int type;
    int n;
    cgsize_t v[27];
};

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
    int fn, nz, ns, nbc;
    cgsize_t sz[9];
    char zname[33];

    chk(cg_open(argv[1], CG_MODE_READ, &fn), "open in");
    cg_nzones(fn, 1, &nz);
    chk(cg_zone_read(fn, 1, 1, zname, sz), "zone read");
    cgsize_t nvert = sz[0];

    std::vector<double> x(nvert), y(nvert), z(nvert);
    cgsize_t rmin = 1, rmax = nvert;
    chk(cg_coord_read(fn, 1, 1, "CoordinateX", RealDouble, &rmin, &rmax,
                      x.data()),
        "x");
    chk(cg_coord_read(fn, 1, 1, "CoordinateY", RealDouble, &rmin, &rmax,
                      y.data()),
        "y");
    chk(cg_coord_read(fn, 1, 1, "CoordinateZ", RealDouble, &rmin, &rmax,
                      z.data()),
        "z");

    /* Read every section into a flat element list indexed by CGNS element
       number, so that a boundary condition's point list can be followed. */
    cg_nsections(fn, 1, 1, &ns);
    cgsize_t maxid = 0;
    for (int S = 1; S <= ns; ++S)
    {
        char n[33];
        CGNS_ENUMT(ElementType_t) et;
        cgsize_t st, en;
        int nb, pf;
        cg_section_read(fn, 1, 1, S, n, &et, &st, &en, &nb, &pf);
        if (en > maxid)
        {
            maxid = en;
        }
    }
    std::vector<Elem> all(maxid + 1);

    for (int S = 1; S <= ns; ++S)
    {
        char n[33];
        CGNS_ENUMT(ElementType_t) et;
        cgsize_t st, en;
        int nb, pf, npe;
        cg_section_read(fn, 1, 1, S, n, &et, &st, &en, &nb, &pf);
        cgsize_t eds;
        cg_ElementDataSize(fn, 1, 1, S, &eds);
        std::vector<cgsize_t> c(eds);
        cgsize_t pd;
        chk(cg_elements_read(fn, 1, 1, S, c.data(), &pd), "elements read");
        cg_npe(et, &npe);
        for (cgsize_t e = st; e <= en; ++e)
        {
            all[e].type = (int)et;
            all[e].n    = npe;
            memcpy(all[e].v, &c[(e - st) * npe], npe * sizeof(cgsize_t));
        }
        printf("  read section '%s' %s %ld..%ld\n", n, ElementTypeName[et],
               (long)st, (long)en);
    }

    /* Surface parts come from the boundary conditions of the source. */
    cg_nbocos(fn, 1, 1, &nbc);
    std::vector<cgsize_t> bclist[MAXE];
    cgsize_t bcn[MAXE];
    for (int I = 1; I <= nbc; ++I)
    {
        char n[33];
        CGNS_ENUMT(BCType_t) bt;
        CGNS_ENUMT(PointSetType_t) ps;
        cgsize_t np, nls;
        int ni[3], nd;
        CGNS_ENUMT(DataType_t) ndt;
        cg_boco_info(fn, 1, 1, I, n, &bt, &ps, &np, ni, &nls, &ndt, &nd);
        bclist[I - 1].resize(np);
        bcn[I - 1] = np;
        chk(cg_boco_read(fn, 1, 1, I, bclist[I - 1].data(), nullptr),
            "boco read");
        printf("  read bc '%s' %ld faces\n", n, (long)np);
    }
    cg_close(fn);

    /* Parts, in the order they will be written. Source BC order on this mesh
       is Wall, Inlet, Outlet; the wall is split so that one part ends up with
       no boundary condition. */
    const char *pname[5] = {"inlet", "outlet", "periodic1", "wall", nullptr};
    int phasbc[4]        = {1, 1, 0, 1};
    cgsize_t *psrc[4];
    cgsize_t pcnt[4];
    psrc[0] = bclist[1].data();
    pcnt[0] = bcn[1]; /* inlet  104 */
    psrc[1] = bclist[2].data();
    pcnt[1] = bcn[2]; /* outlet 100 */
    psrc[2] = bclist[0].data();
    pcnt[2] = 100; /* wall[0..100) */
    psrc[3] = bclist[0].data() + 100;
    pcnt[3] = bcn[0] - 100; /* wall[100..]  */

    chk(cg_open(argv[2], CG_MODE_WRITE, &fn), "open out");
    int B, Z, F, S, BC;
    chk(cg_base_write(fn, "Base", 3, 3, &B), "base");
    for (int p = 0; p < 4; ++p)
    {
        if (phasbc[p])
        {
            chk(cg_family_write(fn, B, pname[p], &F), "family");
        }
    }

    cgsize_t nvol = 0;
    for (cgsize_t e = 1; e <= maxid; ++e)
    {
        int t = all[e].type;
        if (t != CGNS_ENUMV(TRI_3) && t != CGNS_ENUMV(QUAD_4))
        {
            ++nvol;
        }
    }
    cgsize_t zs[3] = {nvert, nvol, 0};
    chk(cg_zone_write(fn, B, "fluid", zs, CGNS_ENUMV(Unstructured), &Z),
        "zone");
    chk(cg_coord_write(fn, B, Z, RealDouble, "CoordinateX", x.data(), &F),
        "wx");
    chk(cg_coord_write(fn, B, Z, RealDouble, "CoordinateY", y.data(), &F),
        "wy");
    chk(cg_coord_write(fn, B, Z, RealDouble, "CoordinateZ", z.data(), &F),
        "wz");

    /* One MIXED section per part, then one for the volume. */
    cgsize_t next = 1;
    cgsize_t prange[4][2];
    std::vector<cgsize_t> conn(32 * (maxid + 1));
    std::vector<cgsize_t> offs(maxid + 2);

    for (int p = 0; p < 4; ++p)
    {
        cgsize_t k = 0;
        for (cgsize_t i = 0; i < pcnt[p]; ++i)
        {
            Elem *el  = &all[psrc[p][i]];
            offs[i]   = k;
            conn[k++] = el->type;
            for (int j = 0; j < el->n; ++j)
            {
                conn[k++] = el->v[j];
            }
        }
        offs[pcnt[p]] = k;
        prange[p][0]  = next;
        prange[p][1]  = next + pcnt[p] - 1;
        chk(cg_poly_section_write(fn, B, Z, pname[p], CGNS_ENUMV(MIXED),
                                  prange[p][0], prange[p][1], 0, conn.data(),
                                  offs.data(), &S),
            "poly section");
        printf("  wrote section '%s' MIXED %ld..%ld\n", pname[p],
               (long)prange[p][0], (long)prange[p][1]);
        next += pcnt[p];
    }

    cgsize_t k = 0, nv = 0;
    for (cgsize_t e = 1; e <= maxid; ++e)
    {
        int t = all[e].type;
        if (t == CGNS_ENUMV(TRI_3) || t == CGNS_ENUMV(QUAD_4))
        {
            continue;
        }
        offs[nv++] = k;
        conn[k++]  = t;
        for (int j = 0; j < all[e].n; ++j)
        {
            conn[k++] = all[e].v[j];
        }
    }
    offs[nv] = k;
    chk(cg_poly_section_write(fn, B, Z, "VolumeElements", CGNS_ENUMV(MIXED),
                              next, next + nv - 1, 0, conn.data(), offs.data(),
                              &S),
        "vol");
    printf("  wrote section 'VolumeElements' MIXED %ld..%ld\n", (long)next,
           (long)(next + nv - 1));

    /* Boundary conditions over the new element numbering. */
    for (int p = 0; p < 4; ++p)
    {
        if (!phasbc[p])
        {
            continue;
        }
        std::vector<cgsize_t> pl(pcnt[p]);
        for (cgsize_t i = 0; i < pcnt[p]; ++i)
        {
            pl[i] = prange[p][0] + i;
        }
        char bcname[64];
        snprintf(bcname, sizeof(bcname), "%s_1", pname[p]);
        chk(cg_boco_write(fn, B, Z, bcname, CGNS_ENUMV(BCTypeUserDefined),
                          CGNS_ENUMV(PointList), pcnt[p], pl.data(), &BC),
            "boco");
        chk(cg_boco_gridlocation_write(fn, B, Z, BC, CGNS_ENUMV(FaceCenter)),
            "gridloc");
        chk(cg_goto(fn, B, "Zone_t", Z, "ZoneBC_t", 1, "BC_t", BC, "end"),
            "goto bc");
        chk(cg_famname_write(pname[p]), "famname");
        printf("  wrote bc '%s' %ld faces, family '%s'\n", bcname,
               (long)pcnt[p], pname[p]);
    }

    cg_close(fn);
    printf("done\n");
    return 0;
}
