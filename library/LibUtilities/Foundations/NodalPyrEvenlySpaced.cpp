///////////////////////////////////////////////////////////////////////////////
//
// File: NodalPyrEvenlySpaced.cpp
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
// Description: 3D Nodal Pyramid Evenly Spaced Point Definitions
//
///////////////////////////////////////////////////////////////////////////////

#include <LibUtilities/Foundations/NodalPyrEvenlySpaced.h>
#include <vector>

namespace Nektar::LibUtilities
{
bool NodalPyrEvenlySpaced::initPointsManager[] = {
    PointsManager().RegisterCreator(PointsKey(0, eNodalPyrEvenlySpaced),
                                    NodalPyrEvenlySpaced::Create)};

// Local vertex/edge/face numbering for the standard pyramid:
//
//   Base square (z == 0): v0 = (0,0,0), v1 = (npts-1,0,0),
//                          v2 = (npts-1,npts-1,0), v3 = (0,npts-1,0)
//   Apex:                 v4 = (0,0,npts-1)
//
//   Edges: e01, e12, e23, e30 (base), e04, e14, e24, e34 (to apex)
//   Faces: f0123 (base quad), f014, f124, f234, f034 (triangles)
//
// Unlike the prism -- whose cross-section only collapses in one direction as
// z increases, since it is a uniform extrusion in y -- the pyramid collapses
// in *both* x and y simultaneously as z increases, tapering to the single
// apex point at z == npts-1. This is why the loop bounds and edge/face
// predicates below involve both (x+z) and (y+z), rather than just x as in
// the prism case.
namespace
{
bool isVertex(size_t x, size_t y, size_t z, size_t npts)
{
    return (x == 0 && y == 0 && z == 0) ||
           (x == (npts - 1) && y == 0 && z == 0) ||
           (x == (npts - 1) && y == (npts - 1) && z == 0) ||
           (x == 0 && y == (npts - 1) && z == 0) ||
           (x == 0 && y == 0 && z == (npts - 1));
}

bool isEdge_01([[maybe_unused]] size_t x, size_t y, size_t z,
               [[maybe_unused]] size_t npts)
{
    return y == 0 && z == 0;
}

bool isEdge_12(size_t x, [[maybe_unused]] size_t y, size_t z, size_t npts)
{
    return x == (npts - 1) && z == 0;
}

bool isEdge_23([[maybe_unused]] size_t x, size_t y, size_t z, size_t npts)
{
    return y == (npts - 1) && z == 0;
}

bool isEdge_30(size_t x, [[maybe_unused]] size_t y, size_t z,
               [[maybe_unused]] size_t npts)
{
    return x == 0 && z == 0;
}

bool isEdge_04(size_t x, size_t y, [[maybe_unused]] size_t z,
               [[maybe_unused]] size_t npts)
{
    return x == 0 && y == 0;
}

bool isEdge_14(size_t x, size_t y, size_t z, size_t npts)
{
    return y == 0 && (x + z) == (npts - 1);
}

bool isEdge_24(size_t x, size_t y, size_t z, size_t npts)
{
    return (x + z) == (npts - 1) && (y + z) == (npts - 1);
}

bool isEdge_34(size_t x, size_t y, size_t z, size_t npts)
{
    return x == 0 && (y + z) == (npts - 1);
}

bool isEdge(size_t x, size_t y, size_t z, size_t npts)
{
    return isEdge_01(x, y, z, npts) || isEdge_12(x, y, z, npts) ||
           isEdge_23(x, y, z, npts) || isEdge_30(x, y, z, npts) ||
           isEdge_04(x, y, z, npts) || isEdge_14(x, y, z, npts) ||
           isEdge_24(x, y, z, npts) || isEdge_34(x, y, z, npts);
}

bool isFace_0123([[maybe_unused]] size_t x, [[maybe_unused]] size_t y, size_t z,
                 [[maybe_unused]] size_t npts)
{
    return z == 0;
}

bool isFace_014([[maybe_unused]] size_t x, size_t y, [[maybe_unused]] size_t z,
                [[maybe_unused]] size_t npts)
{
    return y == 0;
}

bool isFace_124(size_t x, [[maybe_unused]] size_t y, size_t z, size_t npts)
{
    return (x + z) == (npts - 1);
}

bool isFace_234([[maybe_unused]] size_t x, size_t y, size_t z, size_t npts)
{
    return (y + z) == (npts - 1);
}

bool isFace_034(size_t x, [[maybe_unused]] size_t y, [[maybe_unused]] size_t z,
                [[maybe_unused]] size_t npts)
{
    return x == 0;
}

bool isFace(size_t x, size_t y, size_t z, size_t npts)
{
    return isFace_0123(x, y, z, npts) || isFace_014(x, y, z, npts) ||
           isFace_124(x, y, z, npts) || isFace_234(x, y, z, npts) ||
           isFace_034(x, y, z, npts);
}
} // namespace

// Calculate evenly spaced number of points
void NodalPyrEvenlySpaced::v_CalculatePoints()
{
    // Allocate the storage for points
    PointsBaseType::v_CalculatePoints();

    // Populate m_points
    size_t npts     = GetNumPoints();
    NekDouble delta = 2.0 / (npts - 1.0);
    for (size_t z = 0, index = 0; z < npts; ++z)
    {
        for (size_t y = 0; y < npts - z; ++y)
        {
            for (size_t x = 0; x < npts - z; ++x, ++index)
            {
                NekDouble xi = -1.0 + x * delta;
                NekDouble yi = -1.0 + y * delta;
                NekDouble zi = -1.0 + z * delta;

                m_points[0][index] = xi;
                m_points[1][index] = yi;
                m_points[2][index] = zi;
            }
        }
    }

    NodalPointReorder3d();
    m_util = MemoryManager<NodalUtilPyr>::AllocateSharedPtr(
        npts - 1, m_points[0], m_points[1], m_points[2]);
}

void NodalPyrEvenlySpaced::NodalPointReorder3d()
{
    size_t npts = GetNumPoints();
    using std::vector;
    vector<int> vertex;
    vector<int> iEdge_01;             // interior edge 0
    vector<int> iEdge_12;             // interior edge 1
    vector<int> iEdge_23;             // interior edge 2
    vector<int> iEdge_30;             // interior edge 3
    vector<int> iEdge_04;             // interior edge 4
    vector<int> iEdge_14;             // interior edge 5
    vector<int> iEdge_24;             // interior edge 6
    vector<int> iEdge_34;             // interior edge 7
    vector<int> iFace_0123;           // interior face 0 (base quad)
    vector<int> iFace_014;            // interior face 1
    vector<int> iFace_124;            // interior face 2
    vector<int> iFace_234;            // interior face 3
    vector<int> iFace_034;            // interior face 4
    vector<int> interiorVolumePoints; // interior volume points
    vector<int> map;

    // Build the lattice pyramid left to right - bottom to top. Note that,
    // unlike the prism, both the x and y extents shrink together as z
    // increases, since the pyramid tapers to a single apex point rather than
    // an edge.
    for (size_t z = 0, index = 0; z < npts; ++z)
    {
        for (size_t y = 0; y < npts - z; ++y)
        {
            for (size_t x = 0; x < npts - z; ++x, ++index)
            {
                if (isVertex(x, y, z, npts))
                {
                    vertex.push_back(index);
                }
                else if (isEdge(x, y, z, npts))
                {
                    if (isEdge_01(x, y, z, npts))
                    {
                        iEdge_01.push_back(index);
                    }
                    else if (isEdge_12(x, y, z, npts))
                    {
                        iEdge_12.push_back(index);
                    }
                    else if (isEdge_23(x, y, z, npts))
                    {
                        iEdge_23.push_back(index);
                    }
                    else if (isEdge_30(x, y, z, npts))
                    {
                        iEdge_30.push_back(index);
                    }
                    else if (isEdge_04(x, y, z, npts))
                    {
                        iEdge_04.push_back(index);
                    }
                    else if (isEdge_14(x, y, z, npts))
                    {
                        iEdge_14.push_back(index);
                    }
                    else if (isEdge_24(x, y, z, npts))
                    {
                        iEdge_24.push_back(index);
                    }
                    else if (isEdge_34(x, y, z, npts))
                    {
                        iEdge_34.push_back(index);
                    }
                }
                else if (isFace(x, y, z, npts))
                {
                    if (isFace_0123(x, y, z, npts))
                    {
                        iFace_0123.push_back(index);
                    }
                    else if (isFace_014(x, y, z, npts))
                    {
                        iFace_014.push_back(index);
                    }
                    else if (isFace_124(x, y, z, npts))
                    {
                        iFace_124.push_back(index);
                    }
                    else if (isFace_234(x, y, z, npts))
                    {
                        iFace_234.push_back(index);
                    }
                    else if (isFace_034(x, y, z, npts))
                    {
                        iFace_034.push_back(index);
                    }
                }
                else
                {
                    interiorVolumePoints.push_back(index);
                }
            }
        }
    }

    for (size_t n = 0; n < vertex.size(); ++n)
    {
        map.push_back(vertex[n]);
    }

    for (size_t n = 0; n < iEdge_01.size(); ++n)
    {
        map.push_back(iEdge_01[n]);
    }

    for (size_t n = 0; n < iEdge_12.size(); ++n)
    {
        map.push_back(iEdge_12[n]);
    }

    for (size_t n = 0; n < iEdge_23.size(); ++n)
    {
        map.push_back(iEdge_23[n]);
    }

    for (size_t n = 0; n < iEdge_30.size(); ++n)
    {
        map.push_back(iEdge_30[n]);
    }

    for (size_t n = 0; n < iEdge_04.size(); ++n)
    {
        map.push_back(iEdge_04[n]);
    }

    for (size_t n = 0; n < iEdge_14.size(); ++n)
    {
        map.push_back(iEdge_14[n]);
    }

    for (size_t n = 0; n < iEdge_24.size(); ++n)
    {
        map.push_back(iEdge_24[n]);
    }

    for (size_t n = 0; n < iEdge_34.size(); ++n)
    {
        map.push_back(iEdge_34[n]);
    }

    for (size_t n = 0; n < iFace_0123.size(); ++n)
    {
        map.push_back(iFace_0123[n]);
    }

    for (size_t n = 0; n < iFace_014.size(); ++n)
    {
        map.push_back(iFace_014[n]);
    }

    for (size_t n = 0; n < iFace_124.size(); ++n)
    {
        map.push_back(iFace_124[n]);
    }

    for (size_t n = 0; n < iFace_234.size(); ++n)
    {
        map.push_back(iFace_234[n]);
    }

    for (size_t n = 0; n < iFace_034.size(); ++n)
    {
        map.push_back(iFace_034[n]);
    }

    for (size_t n = 0; n < interiorVolumePoints.size(); ++n)
    {
        map.push_back(interiorVolumePoints[n]);
    }

    Array<OneD, NekDouble> points[3];
    points[0] = Array<OneD, NekDouble>(GetTotNumPoints());
    points[1] = Array<OneD, NekDouble>(GetTotNumPoints());
    points[2] = Array<OneD, NekDouble>(GetTotNumPoints());

    for (size_t index = 0; index < map.size(); ++index)
    {
        points[0][index] = m_points[0][index];
        points[1][index] = m_points[1][index];
        points[2][index] = m_points[2][index];
    }

    for (size_t index = 0; index < map.size(); ++index)
    {
        m_points[0][index] = points[0][map[index]];
        m_points[1][index] = points[1][map[index]];
        m_points[2][index] = points[2][map[index]];
    }
}

void NodalPyrEvenlySpaced::v_CalculateWeights()
{
    // Allocate the storage for points
    PointsBaseType::v_CalculateWeights();

    typedef DataType T;

    // Solve the Vandermonde system of integrals for the weight vector
    NekVector<T> w = m_util->GetWeights();

    m_weights = Array<OneD, T>(w.GetRows(), w.GetPtr());
}

// ////////////////////////////////////////
//        CalculateInterpMatrix()
void NodalPyrEvenlySpaced::CalculateInterpMatrix(
    const Array<OneD, const NekDouble> &xia,
    const Array<OneD, const NekDouble> &yia,
    const Array<OneD, const NekDouble> &zia, Array<OneD, NekDouble> &interp)
{
    Array<OneD, Array<OneD, NekDouble>> xi(3);
    xi[0] = xia;
    xi[1] = yia;
    xi[2] = zia;

    std::shared_ptr<NekMatrix<NekDouble>> mat =
        m_util->GetInterpolationMatrix(xi);
    Vmath::Vcopy(mat->GetRows() * mat->GetColumns(), mat->GetRawPtr(), 1,
                 &interp[0], 1);
}

// ////////////////////////////////////////
//        CalculateDerivMatrix()
void NodalPyrEvenlySpaced::v_CalculateDerivMatrix()
{
    // Allocate the derivative matrix.
    PointsBaseType::v_CalculateDerivMatrix();

    m_derivmatrix[0] = m_util->GetDerivMatrix(0);
    m_derivmatrix[1] = m_util->GetDerivMatrix(1);
    m_derivmatrix[2] = m_util->GetDerivMatrix(2);
}

std::shared_ptr<PointsBaseType> NodalPyrEvenlySpaced::Create(
    const PointsKey &key)
{
    std::shared_ptr<PointsBaseType> returnval(
        MemoryManager<NodalPyrEvenlySpaced>::AllocateSharedPtr(key));

    returnval->Initialize();

    return returnval;
}
} // namespace Nektar::LibUtilities
