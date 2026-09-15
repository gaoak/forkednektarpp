///////////////////////////////////////////////////////////////////////////////
//
// File: TestParallelReduce.hpp
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
// Description: Test fixture for the parallel_reduce loop-execution
// backends.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <MultiRegions/ExpList.h>
#include <SpatialDomains/MeshGraphIO.h>

#include <LibUtilities/BasicUtils/Field/Field.hpp>

#include <UnitTests/TestBoostSetup.hpp>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <string>

#if defined(_MSC_VER)
#undef max
#undef min
#endif

using namespace Nektar;
using namespace Nektar::LibUtilities;
using namespace Nektar::MultiRegions;

struct GlobalConfiguration
{
    GlobalConfiguration()
    {
        [[maybe_unused]] int argc =
            boost::unit_test::framework::master_test_suite().argc;
        [[maybe_unused]] char **argv =
            boost::unit_test::framework::master_test_suite().argv;

#ifdef NEKTAR_USE_MPI
        MPI_Init(&argc, &argv);
#endif
    }

    ~GlobalConfiguration()
    {
#ifdef NEKTAR_USE_MPI
        MPI_Finalize();
#endif
    }
};

#if defined(BOOST_TEST_NO_MAIN)

bool init_function()
{
    return true;
}

int main(int argc, char *argv[])
{
    GlobalConfiguration gc;

    return boost::unit_test::unit_test_main(&init_function, argc, argv);
}

#else
BOOST_TEST_GLOBAL_CONFIGURATION(GlobalConfiguration);
#endif

template <typename TData> class ParallelReduceField
{
public:
    ParallelReduceField()
    {
        m_meshName = "run/segment.xml";
    }

    ~ParallelReduceField()
    {
        BOOST_TEST_MESSAGE("teardown this->fixture");

        if (this->m_in)
        {
            delete this->m_in;
        }
    }

    void Configure()
    {
        BOOST_TEST_MESSAGE("Creating input and output fields");

        // Construct a fake command-line argument array to be fed to
        // Session::Reader::CreateInstance. The first element stands for
        // the name of the executable which, in our case, doesn't matter.
        int argc    = 2;
        char **argv = new char *[argc];
        argv[0]     = strdup("exe_name");
        argv[1]     = strdup(m_meshName.data());

        this->m_session = SessionReader::CreateInstance(argc, argv);
        auto graph      = SpatialDomains::MeshGraphIO::Read(this->m_session);

        for (int i = 0; i < argc; ++i)
        {
            free(argv[i]);
        }
        delete[] argv;

        this->m_expList = MemoryManager<ExpList>::AllocateSharedPtr(
            this->m_session, graph, true, "u", Collections::eNoCollection);

        // One component per session variable, so the reducers are exercised
        // over a field wider than a single component.
        const auto &components = this->m_session->GetVariables();
        auto blocksAttr =
            GetBlockAttributes<TData, FieldState::Phys>(m_expList);
        auto f_in =
            Field<TData, FieldState::Phys>("f_in", blocksAttr, components, 1);
        m_in = new Field<TData, FieldState::Phys>(std::move(f_in));
    }

    void SetTestCase()
    {
        // GetCoords() and Equation::Evaluate() take double arrays, whatever
        // TData is; the values are narrowed when copied into the field.
        Array<OneD, double> x(this->m_expList->GetTotPoints());
        Array<OneD, double> y(this->m_expList->GetTotPoints());
        Array<OneD, double> z(this->m_expList->GetTotPoints());
        Array<OneD, double> fce(this->m_expList->GetTotPoints());
        this->m_expList->GetCoords(x, y, z);

        for (unsigned int nc = 0; nc < this->m_in->GetNumComponents(); ++nc)
        {
            auto func = this->m_session->GetFunction("Forcing", nc);
            func->Evaluate(x, y, z, fce);

            auto ptr = fce.data();
            for (unsigned int blk = 0; blk < this->m_in->GetBlocks().size();
                 ++blk)
            {
                auto &block = this->m_in->GetBlocks()[blk];
                auto inptr =
                    block.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
                auto size = block.GetNumElements() * block.GetNumData();
                std::copy(ptr, ptr + size, inptr + nc * block.CompSize());
                ptr += size;
            }
        }
    }

    TData sum()
    {
        TData out = 0;
        for (unsigned int blk = 0; blk < this->m_in->GetBlocks().size(); ++blk)
        {
            auto x = this->m_in->GetBlocks()[blk]
                         .template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
            auto size = this->m_in->GetBlocks()[blk].GetNumElements() *
                        this->m_in->GetBlocks()[blk].GetNumData();
            auto stride = this->m_in->GetBlocks()[blk].CompSize();

            for (unsigned int nc = 0; nc < this->m_in->GetNumComponents();
                 ++nc, x += stride)
            {
                out = std::accumulate(x, x + size, out);
            }
        }
        return out;
    }

    TData max()
    {
        TData out = std::numeric_limits<TData>::lowest();
        for (unsigned int blk = 0; blk < this->m_in->GetBlocks().size(); ++blk)
        {
            auto x = this->m_in->GetBlocks()[blk]
                         .template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
            auto size = this->m_in->GetBlocks()[blk].GetNumElements() *
                        this->m_in->GetBlocks()[blk].GetNumData();
            auto stride = this->m_in->GetBlocks()[blk].CompSize();

            for (unsigned int nc = 0; nc < this->m_in->GetNumComponents();
                 ++nc, x += stride)
            {
                out = std::max(out, *(std::max_element(x, x + size)));
            }
        }
        return out;
    }

    TData min()
    {
        TData out = std::numeric_limits<TData>::max();
        for (unsigned int blk = 0; blk < this->m_in->GetBlocks().size(); ++blk)
        {
            auto x = this->m_in->GetBlocks()[blk]
                         .template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
            auto size = this->m_in->GetBlocks()[blk].GetNumElements() *
                        this->m_in->GetBlocks()[blk].GetNumData();
            auto stride = this->m_in->GetBlocks()[blk].CompSize();

            for (unsigned int nc = 0; nc < this->m_in->GetNumComponents();
                 ++nc, x += stride)
            {
                out = std::min(out, *(std::min_element(x, x + size)));
            }
        }
        return out;
    }

protected:
    std::string m_meshName               = "";
    Field<TData, FieldState::Phys> *m_in = nullptr;
    ExpListSharedPtr m_expList;
    SessionReaderSharedPtr m_session;
};

#include <UnitTests/TestBoostTeardown.hpp>
