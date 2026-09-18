////////////////////////////////////////////////////////////////////////////////
//
//  File: ProcessVarOpti.cpp
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
//  Description:
//
////////////////////////////////////////////////////////////////////////////////

#include <LibUtilities/Foundations/ManagerAccess.h>
#include <LibUtilities/Foundations/NodalUtil.h>

#include "ElUtil.h"
#include "NodeOpti.h"
#include "ProcessVarOpti.h"
#include <NekMesh/MeshElements/Element.h>

#include <StdRegions/StdPrismExp.h>
#include <StdRegions/StdQuadExp.h>
#include <StdRegions/StdTetExp.h>
#include <StdRegions/StdTriExp.h>

#include <LibUtilities/BasicUtils/Timer.h>
#include <LibUtilities/Foundations/NodalUtil.h>

#include <SpatialDomains/CADSystem/CADAssociation.h>
#include <boost/algorithm/string.hpp>

// Including Timer.h includes Windows.h, which causes GetJob to be set as a
// macro for some reason.
#if _WIN32
#undef GetJob
#endif

using namespace std;
using namespace Nektar::NekMesh;

namespace Nektar::NekMesh
{

ModuleKey ProcessVarOpti::className =
    GetModuleFactory().RegisterCreatorFunction(
        ModuleKey(eProcessModule, "varopti"), ProcessVarOpti::create,
        "Optimise mesh locations.");

ProcessVarOpti::ProcessVarOpti(MeshSharedPtr m) : ProcessModule(m)
{
    // clang-format off
    m_config["linearelastic"] =
        ConfigOption(true, "", "Optimise for linear elasticity");
    m_config["winslow"] =
        ConfigOption(true, "", "Optimise for winslow");
    m_config["roca"] =
        ConfigOption(true, "", "Optimise for roca method");
    m_config["hyperelastic"] =
        ConfigOption(true, "", "Optimise for hyper elasticity");
    m_config["numthreads"] =
        ConfigOption(false, "1", "Number of threads");
    m_config["restol"] =
        ConfigOption(false, "1e-6", "Tolerance criterion");
    m_config["maxiter"] =
        ConfigOption(false, "500", "Maximum number of iterations");
    m_config["subiter"] =
        ConfigOption(false, "0", "Number of iterations between updates for r-adaptation");
    m_config["nq"] =
        ConfigOption(false, "-1", "Order of mesh");
    m_config["region"] =
        ConfigOption(false, "0.0", "create regions based on target");
    m_config["resfile"] =
        ConfigOption(false, "", "writes residual values to file");
    m_config["histfile"] =
        ConfigOption(false, "", "histogram of scaled jac");
    m_config["overint"] =
        ConfigOption(false, "6", "over integration order");
    m_config["analytics"] =
        ConfigOption(false, "", "basic analytics module");
    m_config["scalingfile"] =
        ConfigOption(false, "", "Read scaling field from file for r-adaptation");
    m_config["radaptscale"] =
        ConfigOption(false, "0", "Maps scaling for curve-based r-adaption");
    m_config["radaptrad"] =
        ConfigOption(false, "0", "Radius of influence for curve-based r-adaption");
    m_config["radaptcurves"] =
        ConfigOption(false, "", "Array of curves to use for adaption");
    // clang-format on
}

ProcessVarOpti::~ProcessVarOpti()
{
}

void ProcessVarOpti::Process()
{
    m_log(VERBOSE) << "Optimising mesh quality." << endl;

    const int meshDim = m_mesh->m_meshGraph->GetMeshDimension();

    if (m_config["linearelastic"].beenSet)
    {
        m_opti = eLinEl;
    }
    else if (m_config["winslow"].beenSet)
    {
        m_opti = eWins;
    }
    else if (m_config["roca"].beenSet)
    {
        m_opti = eRoca;
    }
    else if (m_config["hyperelastic"].beenSet)
    {
        m_opti = eHypEl;
    }
    else
    {
        m_log(FATAL) << "You should specify a type of optimisation, e.g. "
                     << "'hyperelastic'." << endl;
    }

    const int maxIter      = m_config["maxiter"].as<int>();
    const NekDouble restol = m_config["restol"].as<NekDouble>();
    int subIter            = m_config["subiter"].as<int>();
    if (subIter == 0)
    {
        subIter = maxIter;
    }

    if (m_config["nq"].beenSet)
    {
        m_nummode = m_config["nq"].as<int>();
    }

    if (!m_nummode)
    {
        for (auto &[id, edge] :
             m_mesh->m_meshGraph->GetGeomMap<SpatialDomains::SegGeom>())
        {
            SpatialDomains::Curve *c = edge->GetCurve();

            if (!c)
            {
                continue;
            }

            if (c->m_points.size() > 0)
            {
                m_nummode = c->m_points.size();
                break;
            }
        }
    }

    if (!m_nummode)
    {
        m_log(FATAL) << "Failed to find order of mesh: set nq directly as a "
                        "module option"
                     << std::endl;
    }

    // Safety feature: limit over-integration order for high-order triangles
    // over order 5.
    int intOrder = m_config["overint"].as<int>();
    intOrder     = m_nummode + intOrder <= 11 ? intOrder : 11 - m_nummode;

    m_log(VERBOSE) << "  - Identified mesh order as: " << m_nummode - 1 << endl;

    if (meshDim == 2 && m_mesh->m_meshGraph->GetSpaceDimension() == 3)
    {
        m_log(FATAL) << "Cannot presently optimise manifold meshes (2D embedded"
                     << " in 3D)." << endl;
    }

    m_res      = std::shared_ptr<Residual>(new Residual);
    m_res->val = 1.0;
    m_mesh->MakeOrder(m_nummode - 1, LibUtilities::eGaussLobattoLegendre,
                      m_log);

    if (m_config["analytics"].beenSet)
    {
        Analytics();
        return;
    }

    map<LibUtilities::ShapeType, DerivUtilSharedPtr> derivUtils =
        BuildDerivUtil(intOrder);

    GetElementMap(intOrder, derivUtils);

    m_res->startInv = 0;
    m_res->worstJac = numeric_limits<double>::max();
    for (int i = 0; i < m_dataSet.size(); i++)
    {
        m_dataSet[i]->Evaluate();
        m_dataSet[i]->InitialMinJac();
    }

    vector<ElUtilSharedPtr> elLock;

    if (m_config["region"].beenSet)
    {
        elLock = GetLockedElements(m_config["region"].as<NekDouble>());
    }

    vector<vector<SpatialDomains::PointGeom *>> freenodes =
        GetColouredNodes(elLock);
    vector<vector<NodeOptiSharedPtr>> optiNodes;

    // Null when the mesh has no CAD attached, in which case every node is
    // free. Fetched once: asking the graph for it creates it.
    SpatialDomains::CADAssociationSharedPtr cad =
        m_mesh->m_meshGraph->HasCAD() ? m_mesh->m_meshGraph->GetCADAssociation()
                                      : nullptr;

    // turn the free nodes into optimisable objects with all required data
    std::unordered_set<SpatialDomains::PointGeom *> check;
    for (int i = 0; i < freenodes.size(); i++)
    {
        vector<NodeOptiSharedPtr> ns;
        for (int j = 0; j < freenodes[i].size(); j++)
        {
            auto it = m_nodeElMap.find(freenodes[i][j]);
            ASSERTL0(it != m_nodeElMap.end(), "could not find");

            // Nodes sitting on the CAD are constrained to it: one on a CAD
            // curve may only slide along the curve, one on a CAD surface only
            // within the surface, and one coinciding with a CAD vertex is
            // frozen (optiKind 0, skipped below). Everything else moves
            // freely in the full space.
            int optiKind = m_mesh->m_meshGraph->GetSpaceDimension();

            if (cad != nullptr &&
                cad->Has(freenodes[i][j], SpatialDomains::CADType::eCurve))
            {
                // A CAD vertex is where two curves meet; moving it would pull
                // the topology apart, so it does not get optimised at all.
                optiKind = cad->GetVert(freenodes[i][j]) ? 0 : optiKind + 10;
            }
            else if (cad != nullptr &&
                     cad->Has(freenodes[i][j], SpatialDomains::CADType::eSurf))
            {
                optiKind += 20;
            }
            else
            {
                optiKind += 10 * meshDim;
            }

            auto c = check.find(freenodes[i][j]);
            ASSERTL0(c == check.end(), "duplicate node");
            check.insert(freenodes[i][j]);

            if (optiKind)
            {
                ns.push_back(GetNodeOptiFactory().CreateInstance(
                    optiKind, freenodes[i][j], it->second, m_res, derivUtils,
                    m_opti, m_mesh->m_meshGraph.get()));
            }
        }
        optiNodes.push_back(ns);
    }

    int nset = optiNodes.size();
    int p    = 0;
    int mn   = numeric_limits<int>::max();
    int mx   = 0;
    for (int i = 0; i < nset; i++)
    {
        p += optiNodes[i].size();
        mn = min(mn, int(optiNodes[i].size()));
        mx = max(mx, int(optiNodes[i].size()));
    }

    if (m_config["histfile"].beenSet)
    {
        ofstream histFile;
        string name = m_config["histfile"].as<string>() + "_start.txt";
        histFile.open(name.c_str());

        for (int i = 0; i < m_dataSet.size(); i++)
        {
            histFile << m_dataSet[i]->GetScaledJac() << endl;
        }
        histFile.close();
    }

    m_log(VERBOSE) << "Mesh statistics prior to optimisation:" << endl;
    m_log(VERBOSE) << "  - # elements         : "
                   << m_mesh->m_elementTags[meshDim].size() - elLock.size()
                   << endl;
    m_log(VERBOSE) << "  - # invalid elements : " << m_res->startInv << endl;
    m_log(VERBOSE) << "  - Worst Jacobian     : " << scientific
                   << m_res->worstJac << endl;
    m_log(VERBOSE) << "  - # free nodes       : " << m_res->n << endl;
    m_log(VERBOSE) << "  - # DoF              : " << m_res->nDoF << endl;
    m_log(VERBOSE) << "  - # color sets       : " << nset << endl;
    m_log(VERBOSE) << "  - Avg set colors     : " << scientific << p / nset
                   << endl;
    m_log(VERBOSE) << "  - Min set            : " << mn << endl;
    m_log(VERBOSE) << "  - Max set            : " << mx << endl;
    m_log(VERBOSE) << "  - Residual tolerance : " << scientific << restol
                   << endl;

    int nThreads = m_config["numthreads"].as<int>();

    if (m_mesh->m_meshGraph->HasCAD())
    {
        if (boost::equals(m_mesh->m_meshGraph->GetCAD()->GetEngine(), "cfi"))
        {
            m_log(WARNING) << "CFI is not thread-safe; forcing to "
                           << "'numthreads=1'." << endl;
            nThreads = 1;
        }
    }

    int ctr = 0;
    Thread::ThreadMaster tms;
    tms.SetThreadingType("ThreadManagerStd");
    Thread::ThreadManagerSharedPtr tm =
        tms.CreateInstance(Thread::ThreadMaster::SessionJob, nThreads);

    LibUtilities::Timer t;
    t.Start();

    ofstream resFile;
    if (m_config["resfile"].beenSet)
    {
        resFile.open(m_config["resfile"].as<string>().c_str());
    }

    for (int i = 0; i < optiNodes.size(); i++)
    {
        vector<Thread::ThreadJob *> jobs(optiNodes[i].size());
        for (int j = 0; j < optiNodes[i].size(); j++)
        {
            optiNodes[i][j]->CalcMinJac();
        }
    }

    m_log(VERBOSE) << "Beginning iterations..." << endl;

    while (m_res->val > restol && ctr < maxIter)
    {
        ctr++;
        m_res->val       = 0.0;
        m_res->func      = 0.0;
        m_res->nReset[0] = 0;
        m_res->nReset[1] = 0;
        m_res->nReset[2] = 0;
        m_res->alphaI    = 0;
        for (int i = 0; i < optiNodes.size(); i++)
        {
            vector<Thread::ThreadJob *> jobs(optiNodes[i].size());
            for (int j = 0; j < optiNodes[i].size(); j++)
            {
                jobs[j] = optiNodes[i][j]->GetJob();
            }

            tm->SetNumWorkers(0);
            tm->QueueJobs(jobs);
            tm->SetNumWorkers(nThreads);
            tm->Wait();
        }

        m_res->startInv = 0;
        m_res->worstJac = numeric_limits<double>::max();

        bool updateCAD = (m_radaptCAD && (ctr % subIter) == 0);
        bool updateFile =
            ((m_config["scalingfile"].beenSet) && (ctr % subIter) == 0);

        vector<Thread::ThreadJob *> elJobs(m_dataSet.size());
        for (int i = 0; i < m_dataSet.size(); i++)
        {
            if (updateCAD)
            {
                elJobs[i] = m_dataSet[i]->GetAdaptJob(
                    m_adaptCurves, m_config["radaptscale"].as<NekDouble>(),
                    m_config["radaptrad"].as<NekDouble>(),
                    m_mesh->m_meshGraph.get());
            }
            else
            {
                elJobs[i] = m_dataSet[i]->GetJob(updateFile);
            }
        }

        tm->SetNumWorkers(0);
        tm->QueueJobs(elJobs);
        tm->SetNumWorkers(nThreads);
        tm->Wait();

        if (m_config["resfile"].beenSet)
        {
            resFile << m_res->val << " " << m_res->worstJac << " "
                    << m_res->func << endl;
        }

        m_log(VERBOSE) << "  - Iteration: " << ctr
                       << "\tResidual: " << m_res->val
                       << "\tMin Jac: " << m_res->worstJac
                       << "\tInvalid: " << m_res->startInv
                       << "\tReset nodes: " << m_res->nReset[0] << "/"
                       << m_res->nReset[1] << "/" << m_res->nReset[2]
                       << "\tFunctional: " << m_res->func << endl;

        if (ctr >= maxIter)
        {
            break;
        }

        if (updateFile || updateCAD)
        {
            m_log(VERBOSE) << "    => Mapping updated!" << endl;
        }
    }

    if (m_config["histfile"].beenSet)
    {
        ofstream histFile;
        string name = m_config["histfile"].as<string>() + "_end.txt";
        histFile.open(name.c_str());

        for (int i = 0; i < m_dataSet.size(); i++)
        {
            histFile << m_dataSet[i]->GetScaledJac() << endl;
        }
        histFile.close();
    }
    if (m_config["resfile"].beenSet)
    {
        resFile.close();
    }

    t.Stop();

    // RemoveLinearCurvature();

    m_log(VERBOSE) << "Optimisation complete!" << endl;
    m_log(VERBOSE) << "  - Time to compute: " << t.TimePerTest(1) << endl;
    m_log(VERBOSE) << "  - Invalid at end : " << m_res->startInv << endl;
    m_log(VERBOSE) << "  - Worst at end   : " << m_res->worstJac << endl;
}

class NodalUtilTriMonomial : public LibUtilities::NodalUtilTriangle
{
public:
    NodalUtilTriMonomial(int degree, Array<OneD, NekDouble> r,
                         Array<OneD, NekDouble> s)
        : NodalUtilTriangle(degree, r, s)
    {
    }

    ~NodalUtilTriMonomial() override
    {
    }

protected:
    NekVector<NekDouble> v_OrthoBasis(const size_t mode) override
    {
        // Monomial basis.
        std::pair<int, int> modes = m_ordering[mode];
        NekVector<NekDouble> ret(m_numPoints);

        for (size_t i = 0; i < m_numPoints; ++i)
        {
            ret(i) =
                pow(m_xi[0][i], modes.first) * pow(m_xi[1][i], modes.second);
        }

        return ret;
    }

    NekVector<NekDouble> v_OrthoBasisDeriv(
        [[maybe_unused]] const size_t dir,
        [[maybe_unused]] const size_t mode) override
    {
        NEKERROR(ErrorUtil::efatal, "OrthoBasisDeriv: not supported");
        return NekVector<NekDouble>();
    }

    std::shared_ptr<NodalUtil> v_CreateUtil(
        Array<OneD, Array<OneD, NekDouble>> &xi) override
    {
        return MemoryManager<NodalUtilTriMonomial>::AllocateSharedPtr(
            m_degree, xi[0], xi[1]);
    }
};

void ProcessVarOpti::Analytics()
{
    const int meshDim = m_mesh->m_meshGraph->GetMeshDimension();

    // Grab the first element from the list
    ASSERTL0(!m_mesh->m_elementTags[meshDim].empty(),
             "No elements of the mesh dimension to analyse");
    SpatialDomains::Geometry *elmt =
        m_mesh->m_elementTags[meshDim].begin()->first;

    // Get curved nodes
    std::vector<SpatialDomains::PointGeom *> nodes = GetCurvedNodes(elmt);

    // We're going to investigate only the first node (corner node)
    SpatialDomains::PointGeom *node = nodes[4];

    // Loop over overintegration orders
    const int nPoints       = 200;
    const int overInt       = 40;
    const NekDouble originX = -1.0;
    const NekDouble originY = -1.0;
    const NekDouble length  = 2.0;
    const NekDouble dx      = length / (nPoints - 1);

    m_log(VERBOSE) << "# overint = " << overInt << endl;
    m_log(VERBOSE) << "# Columns: x, y, over-integration orders (0 -> "
                   << overInt - 1 << "), "
                   << " min(scaledJac)" << endl;

    // Loop over square defined by (originX, originY), length
    for (int k = 0; k < nPoints; ++k)
    {
        (*node)[1] = originY + k * dx;
        for (int j = 0; j < nPoints; ++j)
        {
            (*node)[0] = originX + j * dx;
            m_log(VERBOSE) << (*node)[0] << " " << (*node)[1] << " ";

            NekDouble minJacNew;

            for (int i = 0; i < overInt; ++i)
            {
                // Clear any existing node to element mapping.
                m_dataSet.clear();
                m_nodeElMap.clear();

                // Build deriv utils and element map.
                map<LibUtilities::ShapeType, DerivUtilSharedPtr> derivUtils =
                    BuildDerivUtil(i);

                // Reconstruct element map
                GetElementMap(i, derivUtils);

                for (int j = 0; j < m_dataSet.size(); j++)
                {
                    m_dataSet[j]->Evaluate();
                    m_dataSet[j]->InitialMinJac();
                }

                // Create NodeOpti object.
                NodeOptiSharedPtr nodeOpti =
                    GetNodeOptiFactory().CreateInstance(
                        m_mesh->m_meshGraph->GetSpaceDimension() * 11, node,
                        m_nodeElMap.find(node)->second, m_res, derivUtils,
                        m_opti, m_mesh->m_meshGraph.get());

                minJacNew = 0.0;

                // Evaluate functional.
                nodeOpti->CalcMinJac();
                m_log(VERBOSE) << nodeOpti->GetFunctional<2>(minJacNew) << " ";
            }

            m_log(VERBOSE) << minJacNew << endl;
        }
    }
}
} // namespace Nektar::NekMesh
