///////////////////////////////////////////////////////////////////////////////
//
// File: CheckXmlFile.cpp
//
// For more information, please see: http://www.nektar.info
//
// The MIT License
//
// Copyright (c) 2006 Division of Applied Mathematics, Brown University (USA),
// Department of Aeronautics, Imperial College London (UK), and Scientific
// Computing and Imaging Institute, University of Utah (USA).
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
//
// Description:
//
///////////////////////////////////////////////////////////////////////////////

#include <MultiRegions/ExpList.h>
#include <SpatialDomains/MeshGraphIO.h>

using namespace Nektar;

bool CheckTetRotation(Array<OneD, NekDouble> &xc, Array<OneD, NekDouble> &yc,
                      Array<OneD, NekDouble> &xz, SpatialDomains::TetGeom *tet,
                      std::map<int, int> &vid2cnt);

int main(int argc, char *argv[])
{
    Array<OneD, NekDouble> fce;
    Array<OneD, NekDouble> xc0, xc1, xc2;

    if (argc != 2)
    {
        std::cerr << "Usage: CheckXmlFile  meshfile.xml" << std::endl;
        return 1;
    }

    LibUtilities::SessionReaderSharedPtr vSession =
        LibUtilities::SessionReader::CreateInstance(argc, argv);

    //----------------------------------------------
    // Read in mesh from input file
    std::string meshfile(argv[argc - 1]);
    SpatialDomains::MeshGraphSharedPtr mesh =
        SpatialDomains::MeshGraphIO::Read(vSession);
    //----------------------------------------------

    //----------------------------------------------
    // Define Expansion
    int expdim = mesh->GetMeshDimension();

    switch (expdim)
    {
        case 1:
            ASSERTL0(false, "1D not set up");
            break;
        case 2:
            ASSERTL0(false, "2D not set up");
            break;
        case 3:
        {
            int cnt = 0, nverts = mesh->GetNvertices();
            Array<OneD, NekDouble> xc(nverts), yc(nverts), zc(nverts);
            std::map<int, int> vid2cnt;

            for (auto [id, vert] :
                 mesh->GetGeomMap<SpatialDomains::PointGeom>())
            {
                // Check element rotation
                vert->GetCoords(xc[cnt], yc[cnt], zc[cnt]);
                vid2cnt[vert->GetGlobalID()] = cnt++;
            }

            // Check for any duplicated vertex coordinates.
            for (int i = 0; i < nverts; ++i)
            {
                for (int j = i + 1; j < nverts; ++j)
                {
                    if ((xc[i] - xc[j]) * (xc[i] - xc[j]) +
                            (yc[i] - yc[j]) * (yc[i] - yc[j]) +
                            (zc[i] - zc[j]) * (zc[i] - zc[j]) <
                        1e-10)
                    {
                        std::cerr << "ERROR: Duplicate vertices: " << i << " "
                                  << j << std::endl;
                    }
                }
            }

            bool NoRotateIssues      = true;
            bool NoOrientationIssues = true;
            for (auto [id, tet] : mesh->GetGeomMap<SpatialDomains::TetGeom>())
            {
                // check rotation and dump
                NoOrientationIssues =
                    CheckTetRotation(xc, yc, zc, tet, vid2cnt);

                // Check face rotation
                if (tet->GetFace(0)->GetVid(2) != tet->GetVid(2))
                {
                    std::cerr << "ERROR: Face " << tet->GetFid(0) << " (vert "
                              << tet->GetFace(0)->GetVid(2)
                              << ") is not aligned with base vertex of Tet "
                              << tet->GetGlobalID() << " (vert "
                              << tet->GetVid(2) << ")" << std::endl;
                    NoRotateIssues = false;
                }

                for (int i = 1; i < 4; ++i)
                {
                    if (tet->GetFace(i)->GetVid(2) != tet->GetVid(3))
                    {
                        std::cerr << "ERROR: Face " << tet->GetFid(i)
                                  << " is not aligned with top Vertex of Tet "
                                  << tet->GetGlobalID() << std::endl;
                        NoRotateIssues = false;
                    }
                }
            }
            if (NoOrientationIssues)
            {
                std::cout << "All Tet have correct ordering for anticlockwise "
                             "rotation"
                          << std::endl;
            }

            if (NoRotateIssues)
            {
                std::cout << "All Tet faces are correctly aligned" << std::endl;
            }

            for ([[maybe_unused]] auto [id, pyr] :
                 mesh->GetGeomMap<SpatialDomains::PyrGeom>())
            {
                // Put pyramid checks in here
            }

            NoRotateIssues      = true;
            NoOrientationIssues = true;
            for (auto [id, pri] : mesh->GetGeomMap<SpatialDomains::PrismGeom>())
            {
                // Check face rotation
                if (pri->GetFace(1)->GetVid(2) != pri->GetVid(4))
                {
                    std::cerr
                        << "ERROR: Face " << pri->GetFid(1) << " (vert "
                        << pri->GetFace(1)->GetVid(2)
                        << ") not aligned to face 1 singular vert of Prism "
                        << pri->GetGlobalID() << " (vert " << pri->GetVid(4)
                        << ")" << std::endl;
                    NoRotateIssues = false;
                }

                // Check face rotation
                if (pri->GetFace(3)->GetVid(2) != pri->GetVid(5))
                {
                    std::cerr
                        << "ERROR: Face " << pri->GetFid(3) << " (vert "
                        << pri->GetFace(3)->GetVid(2)
                        << ") not aligned to face 3 singular vert of Prism "
                        << pri->GetGlobalID() << " (vert " << pri->GetVid(5)
                        << ")" << std::endl;
                    NoRotateIssues = false;
                }
            }

            if (NoRotateIssues)
            {
                std::cout << "All Prism Tri faces are correctly aligned"
                          << std::endl;
            }

            for ([[maybe_unused]] auto [id, hex] :
                 mesh->GetGeomMap<SpatialDomains::HexGeom>())
            {
                // Put hex checks in here
            }
        }

        break;
        default:
            ASSERTL0(false, "Expansion dimension not recognised");
            break;
    }

    //-----------------------------------------------

    return 0;
}

struct Ord
{
public:
    double x;
    double y;
    double z;
};

bool CheckTetRotation(Array<OneD, NekDouble> &xc, Array<OneD, NekDouble> &yc,
                      Array<OneD, NekDouble> &zc, SpatialDomains::TetGeom *tet,
                      std::map<int, int> &vid2cnt)
{
    bool RotationOK = true;
    std::array<Ord, 4> v;
    NekDouble abx, aby, abz;

    v[0].x = xc[vid2cnt[tet->GetVid(0)]];
    v[0].y = yc[vid2cnt[tet->GetVid(0)]];
    v[0].z = zc[vid2cnt[tet->GetVid(0)]];

    v[1].x = xc[vid2cnt[tet->GetVid(1)]];
    v[1].y = yc[vid2cnt[tet->GetVid(1)]];
    v[1].z = zc[vid2cnt[tet->GetVid(1)]];

    v[2].x = xc[vid2cnt[tet->GetVid(2)]];
    v[2].y = yc[vid2cnt[tet->GetVid(2)]];
    v[2].z = zc[vid2cnt[tet->GetVid(2)]];

    v[3].x = xc[vid2cnt[tet->GetVid(3)]];
    v[3].y = yc[vid2cnt[tet->GetVid(3)]];
    v[3].z = zc[vid2cnt[tet->GetVid(3)]];

    // cross product of edge 0 and 2
    abx = (v[1].y - v[0].y) * (v[2].z - v[0].z) -
          (v[1].z - v[0].z) * (v[2].y - v[0].y);
    aby = (v[1].z - v[0].z) * (v[2].x - v[0].x) -
          (v[1].x - v[0].x) * (v[2].z - v[0].z);
    abz = (v[1].x - v[0].x) * (v[2].y - v[0].y) -
          (v[1].y - v[0].y) * (v[2].x - v[0].x);

    // inner product of cross product with respect to edge 3 should be positive
    if (((v[3].x - v[0].x) * abx + (v[3].y - v[0].y) * aby +
         (v[3].z - v[0].z) * abz) < 0.0)
    {
        std::cerr << "ERROR: Element " << tet->GetGlobalID()
                  << "is NOT counter-clockwise\n"
                  << std::endl;
        RotationOK = false;
    }
    return RotationOK;
}
