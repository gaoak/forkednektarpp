////////////////////////////////////////////////////////////////////////////////
//
//  File: ProcessScalar.cpp
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
//  Description: Add scalar function curvature to a given surface.
//
////////////////////////////////////////////////////////////////////////////////

#include <LibUtilities/BasicUtils/ParseUtils.h>
#include <LibUtilities/Interpreter/Interpreter.h>
#include <NekMesh/MeshElements/Element.h>

#include "ProcessScalar.h"

#include <SpatialDomains/Curve.hpp>

using namespace std;

namespace Nektar::NekMesh
{

ModuleKey ProcessScalar::className = GetModuleFactory().RegisterCreatorFunction(
    ModuleKey(eProcessModule, "scalar"), ProcessScalar::create,
    "Impose a scalar function z=f(x,y) on a surface.");

ProcessScalar::ProcessScalar(MeshSharedPtr m) : ProcessModule(m)
{
    m_config["surf"] = ConfigOption(
        false, "-1", "Tag identifying surface/composite to process.");
    m_config["nq"] =
        ConfigOption(false, "-1", "Number of quadrature points to generate.");
    m_config["scalar"] = ConfigOption(false, "", "Expression to evaluate.");
}

ProcessScalar::~ProcessScalar()
{
}

void ProcessScalar::Process()
{
    string surf = m_config["surf"].as<string>();

    // Obtain vector of surface IDs from string.
    vector<unsigned int> surfs;
    ParseUtils::GenerateSeqVector(surf, surfs);
    sort(surfs.begin(), surfs.end());

    // If we're running in verbose mode print out a list of surfaces.
    m_log(VERBOSE) << "Extracting surface" << (surfs.size() > 1 ? "s" : "")
                   << " " << surf << endl;

    const int nq = m_config["nq"].as<int>();
    string expr  = m_config["scalar"].as<string>();

    LibUtilities::Interpreter rEval;
    int rExprId = rEval.DefineFunction("x y z", expr);

    auto scalar = [&](NekDouble x, NekDouble y) {
        NekDouble z = rEval.Evaluate(rExprId, x, y, 0.0, 0.0);
        return z < 1e-32 ? 0.0 : z;
    };

    const int meshDim = m_mesh->m_meshGraph->GetMeshDimension();

    // The boundary elements are the surface geometries themselves, so what
    // used to be reached through Element::GetFaceLink() is now the tagged
    // entity directly.
    for (auto &[face, tag] : m_mesh->m_elementTags[meshDim - 1])
    {
        if (!binary_search(surfs.begin(), surfs.end(),
                           static_cast<unsigned int>(tag)))
        {
            continue;
        }

        // Update vertices.
        for (int j = 0; j < face->GetNumVerts(); ++j)
        {
            SpatialDomains::PointGeom *n = face->GetVertex(j);
            n->UpdatePosition((*n)[0], (*n)[1], scalar((*n)[0], (*n)[1]));
        }

        // Put curvature into edges.
        for (int j = 0; j < face->GetNumEdges(); ++j)
        {
            auto *edge =
                static_cast<SpatialDomains::SegGeom *>(face->GetEdge(j));

            SpatialDomains::PointGeom *n1 = edge->GetVertex(0);
            SpatialDomains::PointGeom *n2 = edge->GetVertex(1);

            auto curve =
                ObjPoolManager<SpatialDomains::Curve>::AllocateUniquePtr(
                    edge->GetGlobalID(), LibUtilities::ePolyEvenlySpaced);

            curve->m_points.push_back(n1);
            for (int k = 1; k < nq - 1; ++k)
            {
                const NekDouble t = k / (nq - 1.0);
                const NekDouble x = (*n1)[0] + ((*n2)[0] - (*n1)[0]) * t;
                const NekDouble y = (*n1)[1] + ((*n2)[1] - (*n1)[1]) * t;

                auto pt = ObjPoolManager<SpatialDomains::PointGeom>::
                    AllocateUniquePtr(n1->GetCoordim(), 0, x, y, scalar(x, y));
                curve->m_points.push_back(pt.get());
                m_mesh->m_meshGraph->GetAllCurveNodes().push_back(
                    std::move(pt));
            }
            curve->m_points.push_back(n2);

            edge->SetCurve(curve.get());
            m_mesh->m_meshGraph->AddCurvedEdge(std::move(curve));
        }
    }
}
} // namespace Nektar::NekMesh
