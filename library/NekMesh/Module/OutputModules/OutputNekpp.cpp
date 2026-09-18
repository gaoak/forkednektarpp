///////////////////////////////////////////////////////////////////////////////
//
//  File: OutputNekpp.cpp
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
//  Description: Nektar++ file format output.
//
///////////////////////////////////////////////////////////////////////////////

#include <fstream>
#include <set>
#include <string>
#include <thread>

#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/algorithm/string/split.hpp>

#include <LibUtilities/BasicUtils/CppCommandLine.hpp>
#include <LibUtilities/BasicUtils/Filesystem.hpp>
#include <SpatialDomains/MeshGraphIOXml.h>
#include <tinyxml.h>

#include <SpatialDomains/PointGeom.h>

#include "OutputNekpp.h"

using namespace std;

using namespace Nektar::SpatialDomains;

namespace Nektar::NekMesh
{
ModuleKey OutputNekpp::className1 = GetModuleFactory().RegisterCreatorFunction(
    ModuleKey(eOutputModule, "xml"), OutputNekpp::create,
    "Writes a Nektar++ xml file.");

ModuleKey OutputNekpp::className2 = GetModuleFactory().RegisterCreatorFunction(
    ModuleKey(eOutputModule, "nekg"), OutputNekpp::create,
    "Writes a Nektar++ file with hdf5.");

OutputNekpp::OutputNekpp(MeshSharedPtr m) : OutputModule(m)
{
    m_config["chkbndcomp"] = ConfigOption(
        true, "0", "Put all undefined in a compoaite with id=9999");
    m_config["test"] = ConfigOption(
        true, "0", "Attempt to load resulting mesh and create meshgraph.");
    m_config["stats"] =
        ConfigOption(true, "0", "Print out basic mesh statistics.");
    m_config["uncompress"] = ConfigOption(true, "0", "Uncompress xml sections");
    m_config["order"] = ConfigOption(false, "-1", "Enforce a polynomial order");
    m_config["testcond"] = ConfigOption(false, "", "Test a condition.");
    m_config["varopti"] =
        ConfigOption(true, "0", "Run the variational optimser");
    m_config["orient"] = ConfigOption(true, "0", "Reorder Prisms and Tets");
}

OutputNekpp::~OutputNekpp()
{
}

template <typename T>
void TestElmts(SpatialDomains::GeomMapView<T> &geomMap,
               [[maybe_unused]] SpatialDomains::MeshGraphSharedPtr &graph,
               LibUtilities::Interpreter &strEval, int exprId, Logger &log)
{
    for (auto [id, geom] : geomMap)
    {
        geom->Setup();
        geom->FillGeom();

        if (exprId != -1)
        {
            int nq  = geom->GetXmap()->GetTotPoints();
            int dim = geom->GetCoordim();

            Array<OneD, Array<OneD, NekDouble>> coords(3);

            for (int i = 0; i < 3; ++i)
            {
                coords[i] = Array<OneD, NekDouble>(nq, 0.0);
            }

            for (int i = 0; i < dim; ++i)
            {
                geom->GetXmap()->BwdTrans(geom->GetCoeffs(i), coords[i]);
            }

            for (int i = 0; i < nq; ++i)
            {
                NekDouble output = strEval.Evaluate(
                    exprId, coords[0][i], coords[1][i], coords[2][i], 0.0);

                if (output != 1.0)
                {
                    log(FATAL) << "Output mesh failed coordinate test" << endl;
                }
            }

            // Also evaluate at mid-point to test for deformed vs. regular
            // elements.
            Array<OneD, NekDouble> eta(dim, 0.0), evalPt(3, 0.0);
            for (int i = 0; i < dim; ++i)
            {
                evalPt[i] = geom->GetXmap()->PhysEvaluate(eta, coords[i]);
            }

            NekDouble output =
                strEval.Evaluate(exprId, evalPt[0], evalPt[1], evalPt[2], 0.0);

            if (output != 1.0)
            {
                log(FATAL) << "Output mesh failed coordinate midpoint test"
                           << endl;
            }
        }
    }
}

void OutputNekpp::Process()
{
    string filename = m_config["outfile"].as<string>();

    m_log(VERBOSE) << "Writing Nektar++ file '" << filename << "'" << endl;

    // Check whether file exists.
    if (!CheckOverwrite(filename))
    {
        return;
    }

    int order = m_config["order"].as<int>();

    if (order != -1)
    {
        m_mesh->MakeOrder(order, LibUtilities::ePolyEvenlySpaced, m_log);
    }

    // Useful when doing r-adaptation
    if (m_config["varopti"].beenSet)
    {
        unsigned int np        = std::thread::hardware_concurrency();
        ModuleSharedPtr module = GetModuleFactory().CreateInstance(
            ModuleKey(eProcessModule, "varopti"), m_mesh);
        module->RegisterConfig("hyperelastic", "");
        module->RegisterConfig("numthreads", std::to_string(np));

        try
        {
            module->SetDefaults();
            module->Process();
        }
        catch (runtime_error &e)
        {
            m_log(WARNING) << "Variational optimisation has failed with "
                           << "message:" << endl;
            m_log(WARNING) << e.what() << endl;
            m_log(WARNING) << "The mesh will be written as is, it may be "
                           << "invalid" << endl;
            return;
        }
    }

    if (m_config["stats"].beenSet)
    {
        m_mesh->PrintStats(m_log);
    }

    if (m_config["orient"].beenSet)
    {
        PerMap empty;
        ReorderPrisms(empty);
    }

    // Default to compressed XML output.
    std::string type = "XmlCompressed";

    // Compress output and append .gz extension
    if (fs::path(filename).extension() == ".xml" &&
        m_config["uncompress"].beenSet)
    {
        type = "Xml";
    }
    else if (fs::path(filename).extension() == ".nekg")
    {
        type = "HDF5";
    }

    SpatialDomains::MeshGraphIOSharedPtr graphIO =
        SpatialDomains::GetMeshGraphIOFactory().CreateInstance(type);
    graphIO->SetMeshGraph(m_mesh->m_meshGraph);

    graphIO->WriteGeometry(filename, true, m_mesh->m_metadata);

    // For testing let's attempt to load the mesh and create a meshgraph.
    if (m_config["test"].beenSet)
    {
        vector<string> filenames(1);

        if (type == "HDF5")
        {
            // The HDF5 writer puts the geometry in the .nekg file and a
            // session that references it alongside in a .xml, derived from
            // the output name the same way. The session reader needs the
            // latter.
            vector<string> tmp;
            boost::split(tmp, filename, boost::is_any_of("."));
            filenames[0] = tmp[0] + ".xml";
        }
        else
        {
            filenames[0] = filename;
        }

        LibUtilities::Interpreter strEval;
        int exprId       = -1;
        string condition = m_config["testcond"].as<string>();
        if (condition.size() > 0)
        {
            exprId = strEval.DefineFunction("x y z", condition);
        }

        // Fake command line argument for SessionReader construction
        LibUtilities::CppCommandLine cmd({"NekMesh"});
        LibUtilities::SessionReaderSharedPtr vSession =
            LibUtilities::SessionReader::CreateInstance(
                cmd.GetArgc(), cmd.GetArgv(), filenames, m_mesh->m_comm);
        SpatialDomains::MeshGraphSharedPtr graph =
            SpatialDomains::MeshGraphIO::Read(vSession);

        TestElmts(graph->GetGeomMap<SpatialDomains::SegGeom>(), graph, strEval,
                  exprId, m_log);
        TestElmts(graph->GetGeomMap<SpatialDomains::TriGeom>(), graph, strEval,
                  exprId, m_log);
        TestElmts(graph->GetGeomMap<SpatialDomains::QuadGeom>(), graph, strEval,
                  exprId, m_log);
        TestElmts(graph->GetGeomMap<SpatialDomains::TetGeom>(), graph, strEval,
                  exprId, m_log);
        TestElmts(graph->GetGeomMap<SpatialDomains::PrismGeom>(), graph,
                  strEval, exprId, m_log);
        TestElmts(graph->GetGeomMap<SpatialDomains::PyrGeom>(), graph, strEval,
                  exprId, m_log);
        TestElmts(graph->GetGeomMap<SpatialDomains::HexGeom>(), graph, strEval,
                  exprId, m_log);
    }
}
} // namespace Nektar::NekMesh
