///////////////////////////////////////////////////////////////////////////////
//
//  File: MakeTwoImpossiblePyramids.cpp
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
//  Description: Generates two_impossible_pyramids.cgns from
//  ImpossiblePyramids_linear.cgns.
//
////////////////////////////////////////////////////////////////////////////////
// Writes two translated copies of that mesh into one zone, so that the file
// contains two pyramids the reader cannot number as it would like rather than
// one. The copies are far enough apart to share nothing, so each is shielded
// exactly as it is on its own and the mesh stays sound.
//
// Every pyramid fixture here contains a single such pyramid, which is why it
// went unnoticed that ResetNodes left its loop over pyramids altogether once
// one of them had been shielded, instead of going on to the next: the second
// pyramid and all after it were left without an apex id, and an apex that is
// not the highest-numbered node of its pyramid is not the collapsed point the
// geometry requires.
//
// Built by utilities/NekMesh/CMakeLists.txt wherever CGNS is enabled, which
// also adds a test that reruns this and checks the result against the
// fixture committed beside it. To build it by hand instead:
//
//     c++ MakeTwoImpossiblePyramids.cpp -I<cgns>/include -L<cgns>/lib -lcgns \
//         -o MakeTwoImpossiblePyramids
//     ./MakeTwoImpossiblePyramids ImpossiblePyramids_linear.cgns \
//         two_impossible_pyramids.cgns
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

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        printf("usage: %s <in.cgns> <out.cgns>\n", argv[0]);
        return 1;
    }

    int in;
    chk(cg_open(argv[1], CG_MODE_READ, &in), "open in");

    char zname[33];
    cgsize_t size[9];
    chk(cg_zone_read(in, 1, 1, zname, size), "zone_read");

    const cgsize_t nnode = size[0], ncell = size[1];

    int ncoords;
    chk(cg_ncoords(in, 1, 1, &ncoords), "ncoords");

    std::vector<double> x[3];
    char cname[3][33];
    for (int C = 1; C <= ncoords; ++C)
    {
        DataType_t dt;
        chk(cg_coord_info(in, 1, 1, C, &dt, cname[C - 1]), "coord_info");
        x[C - 1].resize(nnode * 2);
        cgsize_t one = 1;
        chk(cg_coord_read(in, 1, 1, cname[C - 1], RealDouble, &one, &nnode,
                          x[C - 1].data()),
            "coord_read");
    }

    /* The second copy is shifted clear of the first along x. */
    double xmin = x[0][0], xmax = x[0][0];
    for (cgsize_t i = 1; i < nnode; ++i)
    {
        xmin = x[0][i] < xmin ? x[0][i] : xmin;
        xmax = x[0][i] > xmax ? x[0][i] : xmax;
    }
    const double shift = 2.0 * (xmax - xmin) + 1.0;

    for (int c = 0; c < ncoords; ++c)
    {
        for (cgsize_t i = 0; i < nnode; ++i)
        {
            x[c][nnode + i] = x[c][i] + (c == 0 ? shift : 0.0);
        }
    }

    int out, B, Z, C, S;
    chk(cg_open(argv[2], CG_MODE_WRITE, &out), "open out");
    chk(cg_base_write(out, "Base", 3, 3, &B), "base_write");

    cgsize_t osize[3] = {nnode * 2, ncell * 2, 0};
    chk(cg_zone_write(out, B, zname, osize, Unstructured, &Z), "zone_write");

    for (int c = 0; c < ncoords; ++c)
    {
        chk(cg_coord_write(out, B, Z, RealDouble, cname[c], x[c].data(), &C),
            "coord_write");
    }

    int nsections;
    chk(cg_nsections(in, 1, 1, &nsections), "nsections");

    cgsize_t next = 1;
    for (int Sin = 1; Sin <= nsections; ++Sin)
    {
        char sname[33];
        ElementType_t type;
        cgsize_t start, end;
        int nbndry, parent;
        chk(cg_section_read(in, 1, 1, Sin, sname, &type, &start, &end, &nbndry,
                            &parent),
            "section_read");

        cgsize_t ndata;
        chk(cg_ElementDataSize(in, 1, 1, Sin, &ndata), "ElementDataSize");

        const cgsize_t nelem = end - start + 1;
        std::vector<cgsize_t> conn(ndata * 2);

        if (type == MIXED)
        {
            /* A MIXED section is read with an offset per element, since its
               elements are not all the same size. */
            std::vector<cgsize_t> off(nelem * 2 + 1);
            cgsize_t pd;
            chk(cg_poly_elements_read(in, 1, 1, Sin, conn.data(), off.data(),
                                      &pd),
                "poly_elements_read");

            cgsize_t i = 0, j = ndata;
            while (i < ndata)
            {
                int npe;
                chk(cg_npe((ElementType_t)conn[i], &npe), "npe");
                conn[j++] = conn[i++];
                for (int k = 0; k < npe; ++k)
                {
                    conn[j++] = conn[i++] + nnode;
                }
            }

            for (cgsize_t e = 1; e <= nelem; ++e)
            {
                off[nelem + e] = off[nelem] + off[e];
            }

            chk(cg_poly_section_write(out, B, Z, sname, type, next,
                                      next + nelem * 2 - 1, nbndry, conn.data(),
                                      off.data(), &S),
                "poly_section_write");
        }
        else
        {
            cgsize_t pd;
            chk(cg_elements_read(in, 1, 1, Sin, conn.data(), &pd),
                "elements_read");

            for (cgsize_t i = 0; i < ndata; ++i)
            {
                conn[ndata + i] = conn[i] + nnode;
            }

            chk(cg_section_write(out, B, Z, sname, type, next,
                                 next + nelem * 2 - 1, nbndry, conn.data(), &S),
                "section_write");
        }

        next += nelem * 2;
    }

    chk(cg_close(in), "close in");
    chk(cg_close(out), "close out");
    printf("wrote %s: %lld nodes, %lld cells\n", argv[2],
           (long long)(nnode * 2), (long long)(ncell * 2));
    return 0;
}
