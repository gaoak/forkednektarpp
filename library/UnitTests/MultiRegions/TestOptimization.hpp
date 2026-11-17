///////////////////////////////////////////////////////////////////////////////
//
// File: TestOptimization.hpp
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

#include "TestOp.hpp"
#include <MultiRegions/Common/Operator.hpp>
#include <MultiRegions/ElmtOps/ElmtOp.hpp>

using namespace Nektar;
using namespace Nektar::LibUtilities;
using namespace Nektar::MultiRegions;

class TestOptimization
{
public:
    TestOptimization()
    {
        meshName = "run/optimization.xml";
    }

    ~TestOptimization()
    {
        BOOST_TEST_MESSAGE("teardown fixture");
    }

    void Configure(const std::string &execSpace)
    {
        BOOST_TEST_MESSAGE("Creating input and output fields");

        // Construct a fake command-line argument array to be fed to
        // Session::Reader::CreateInstance. The first element stands for
        // the name of the executable which, in our case, doesn't matter.
        int argc    = 3;
        char **argv = new char *[argc];
        argv[0]     = strdup("exe_name");
        argv[1]     = strdup(meshName.data());
        argv[2]     = strdup(("--opExecSpace=" + execSpace).c_str());

        session = LibUtilities::SessionReader::CreateInstance(argc, argv);
        session->InitSession();

        for (int i = 0; i < argc; ++i)
        {
            free(argv[i]);
        }
        delete[] argv;
    }

    void Configure(const std::string &execSpace, const std::string &impl)
    {
        BOOST_TEST_MESSAGE("Creating input and output fields");

        // Construct a fake command-line argument array to be fed to
        // Session::Reader::CreateInstance. The first element stands for
        // the name of the executable which, in our case, doesn't matter.
        int argc    = 4;
        char **argv = new char *[argc];
        argv[0]     = strdup("exe_name");
        argv[1]     = meshName.data();
        argv[2]     = strdup(("--opExecSpace=" + execSpace).c_str());
        argv[3]     = strdup(("--opImpl=" + impl).c_str());

        session = LibUtilities::SessionReader::CreateInstance(argc, argv);
        session->InitSession();

        delete[] argv;
    }

    /// The mesh with every element at @p nmodes orthogonal modes, so that a
    /// test sets the expansion order the default implementation follows.
    /// The orthogonal basis is the one that admits a single mode (order 0).
    MultiRegions::ExpListSharedPtr ExpansionList(const int nmodes)
    {
        return ExpansionList(nmodes, nmodes);
    }

    /// The mesh with its first half of elements (composite C[0]) at
    /// @p nmodesFirst orthogonal modes and its second half at
    /// @p nmodesSecond, so that a test can give the two blocks different
    /// orders.
    MultiRegions::ExpListSharedPtr ExpansionList(const int nmodesFirst,
                                                 const int nmodesSecond)
    {
        const SpatialDomains::MeshGraphSharedPtr graph =
            SpatialDomains::MeshGraphIO::Read(session);
        for (const auto &expInfo : graph->GetExpansionInfo("u"))
        {
            expInfo.second->m_basisKeyVector =
                SpatialDomains::MeshGraph::DefineBasisKeyFromExpansionType(
                    expInfo.second->m_geomPtr, SpatialDomains::eOrthogonal,
                    expInfo.first < 5 ? nmodesFirst : nmodesSecond);
        }
        return MemoryManager<MultiRegions::ExpList>::AllocateSharedPtr(
            session, graph, true, "u");
    }

protected:
    std::string meshName = "";
    LibUtilities::SessionReaderSharedPtr session;
};
