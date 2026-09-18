////////////////////////////////////////////////////////////////////////////////
//
//  File: ProcessSpherigon.h
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
//  Description: Refine boundary layer of elements.
//
////////////////////////////////////////////////////////////////////////////////

#ifndef UTILITIES_NEKMESH_PROCESSSPHERIGON
#define UTILITIES_NEKMESH_PROCESSSPHERIGON

#include <NekMesh/Module/InputModules/InputPly.h>
#include <NekMesh/Module/Module.h>
#include <NekMesh/Module/SurfaceHints.h>

namespace Nektar::NekMesh
{

class ProcessSpherigon : public NekMesh::ProcessModule
{
public:
    /// Creates an instance of this class
    static std::shared_ptr<Module> create(NekMesh::MeshSharedPtr m)
    {
        return MemoryManager<ProcessSpherigon>::AllocateSharedPtr(m);
    }
    static NekMesh::ModuleKey className;

    ProcessSpherigon(NekMesh::MeshSharedPtr m);
    ~ProcessSpherigon() override;

    /// Write mesh to output file.
    void Process() override;

    std::string GetModuleName() override
    {
        return "ProcessSpherigon";
    }

protected:
    /// Approximate the vertex normals of a surface by averaging the normals
    /// of the entities meeting at each vertex.
    void GenerateNormals(const std::vector<SpatialDomains::Geometry *> &el,
                         int spaceDim, NekMesh::VertexNormals &out);
    /// Transfer normals from the nearest vertex of a separately read surface.
    void FindNormalFromPlyFile(
        NekMesh::MeshSharedPtr &plymesh,
        const std::vector<SpatialDomains::PointGeom *> &surfverts);

    /// Vertex normals in use for this invocation, whether supplied by an
    /// input module, read from a ply file or approximated here.
    NekMesh::VertexNormals m_normals;
};

} // namespace Nektar::NekMesh

#endif
