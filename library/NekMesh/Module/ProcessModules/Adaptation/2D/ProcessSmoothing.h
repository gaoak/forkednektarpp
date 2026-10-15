////////////////////////////////////////////////////////////////////////////////
//
//  File: ProcessSmoothing.h
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
//  Description: Smoothing the mesh (Linear for the moment Laplacian or
//  Centroidal)
//
////////////////////////////////////////////////////////////////////////////////

#ifndef UTILITIES_NEKMESH_PROCESSSMOOTHING
#define UTILITIES_NEKMESH_PROCESSSMOOTHING

#include <NekMesh/Module/Module.h>

namespace Nektar::NekMesh
{

/**
 * @brief This processing module calculates the Jacobian of elements
 * using %SpatialDomains::GeomFactors and the %Element::GetGeom
 * method. For now it simply prints a list of elements which have
 * negative Jacobian.
 */
class ProcessSmoothing : public NekMesh::ProcessModule
{
public:
    /// Creates an instance of this class
    static std::shared_ptr<Module> create(NekMesh::MeshSharedPtr m)
    {
        return MemoryManager<ProcessSmoothing>::AllocateSharedPtr(m);
    }
    static NekMesh::ModuleKey className;

    ProcessSmoothing(NekMesh::MeshSharedPtr m);
    ~ProcessSmoothing() override;

    /// Write mesh to output file.
    void Process() override;

    std::string GetModuleName() override
    {
        return "ProcessSmoothing";
    }

    bool IsNotValid(std::vector<SpatialDomains::Geometry *> &els);

private:
    const int pyramidV0[4] = {0, 1, 2, 3};
    const int pyramidV1[4] = {1, 2, 3, 0};
    const int pyramidV2[4] = {3, 0, 1, 2};
    const int pyramidV3[4] = {4, 4, 4, 4};

    const NekDouble prismU1[6] = {-1.0, 1.0, 1.0, -1.0, -1.0, -1.0};
    const NekDouble prismV1[6] = {-1.0, -1.0, 1.0, 1.0, -1.0, 1.0};
    const NekDouble prismW1[6] = {-1.0, -1.0, -1.0, -1.0, 1.0, 1.0};

    const NekDouble quadU[4] = {-1.0, 1.0, 1.0, -1.0};
    const NekDouble quadV[4] = {-1.0, -1.0, 1.0, 1.0};
};
} // namespace Nektar::NekMesh

#endif
