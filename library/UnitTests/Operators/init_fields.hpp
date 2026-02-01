///////////////////////////////////////////////////////////////////////////////
//
// File: init_fields.hpp
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

#pragma once

#if defined(NEKTAR_ENABLE_MAGMA)
#include "magma_v2.h"
#endif

#include <MultiRegions/ContField.h>
#include <MultiRegions/ContField3DHomogeneous1D.h>
#include <MultiRegions/ContField3DHomogeneous2D.h>
#include <MultiRegions/DisContField.h>
#include <MultiRegions/DisContField3DHomogeneous1D.h>
#include <MultiRegions/DisContField3DHomogeneous2D.h>
#include <MultiRegions/ExpList.h>
#include <MultiRegions/ExpList3DHomogeneous1D.h>
#include <MultiRegions/ExpList3DHomogeneous2D.h>
#include <SpatialDomains/MeshGraphIO.h>

#include <Operators/Field/Field.hpp>
#include <Operators/Utils/UtilsKernels.hpp>

// Currently the BOOST_TEST_DYN_LINK is local only to this unit
// test. It is undefined at the bottom of the file.
#if defined(OPERATORS_BOOST_TEST_DYN_LINK)
#if !defined(BOOST_TEST_DYN_LINK)
#define LOCALLY_DEFINED_BOOST_TEST_DYN_LINK
#define BOOST_TEST_DYN_LINK
#endif
#endif

// Currently the BOOST_TEST_NO_MAIN is local only to this unit
// test. It is undefined at the bottom of the file.
#if defined(OPERATORS_BOOST_TEST_NO_MAIN)
#if !defined(BOOST_TEST_NO_MAIN)
#define LOCALLY_DEFINED_BOOST_TEST_NO_MAIN
#define BOOST_TEST_NO_MAIN
#endif
#endif

#if defined(BOOST_TEST_DYN_LINK) || defined(BOOST_TEST_NO_MAIN)
#define BOOST_TEST_ALTERNATIVE_INIT_API
#endif

#if defined(BOOST_TEST_DYN_LINK)
#include <boost/test/unit_test.hpp>
#else
#include <boost/test/included/unit_test.hpp>
#endif

#include <boost/test/unit_test_log.hpp>

#include <string>
#include <type_traits>
#include <vector>

#if defined(_MSC_VER)
#undef max
#undef min
#endif

using namespace Nektar::Operators;
using namespace Nektar::LibUtilities;
using namespace Nektar;

struct GlobalConfiguration
{
    GlobalConfiguration()
    {
        [[maybe_unused]] int argc =
            boost::unit_test::framework::master_test_suite().argc;
        [[maybe_unused]] char **argv =
            boost::unit_test::framework::master_test_suite().argv;

#ifdef NEKTAR_ENABLE_MAGMA
        magma_init();
#endif
#ifdef NEKTAR_USE_MPI
        MPI_Init(&argc, &argv);
#endif
    }

    ~GlobalConfiguration()
    {
#ifdef NEKTAR_USE_MPI
        MPI_Finalize();
#endif
#ifdef NEKTAR_ENABLE_MAGMA
        magma_finalize();
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

/**
 * @struct InitFields
 *
 * A test fixture responsible for making available input and output
 * Field objects.  The structure constructor (destructor) is called
 * before (after) each call to BOOST_FIXTURE_TEST_CASE(<test name>,
 * InitFields) macro.
 *
 * See "Single test case fixture" on the Boost.Test documentation for more
 * details:
 * https://www.boost.org/doc/libs/1_82_0/libs/test/doc/html/boost_test/tests_organization/fixtures/case.html
 */

template <typename TData, FieldState stateIn = FieldState::Coeff,
          FieldState stateOut = FieldState::Phys,
          typename TExpList   = MultiRegions::ExpList>
class InitFields
{
public:
    InitFields()
    {
    }

    virtual ~InitFields()
    {
        BOOST_TEST_MESSAGE("teardown fixture");

        if (this->fixt_in)
        {
            delete this->fixt_in;
        }
        if (this->fixt_out)
        {
            delete this->fixt_out;
        }
        if (this->fixt_expected)
        {
            delete this->fixt_expected;
        }
    }

    void Configure(void)
    {
        this->SetSession();
        this->SetExpList();
        this->fixt_explist->SetDataWarehouse();
        this->SetFixture(1);
    }

    void Configure3DH1(const unsigned int nhomo = 1)
    {
        this->SetSession();
        this->SetExpList3DH1(nhomo);
        this->fixt_explist->SetDataWarehouse();
        this->SetFixture(nhomo);
    }

    void Configure3DH2(const unsigned int nhomoY = 1,
                       const unsigned int nhomoZ = 1)
    {
        this->SetSession();
        this->SetExpList3DH2(nhomoY, nhomoZ);
        this->fixt_explist->SetDataWarehouse();
        this->SetFixture(nhomoY * nhomoZ);
    }

    void SetSession(void)
    {
        std::string execStr(
            boost::unit_test::framework::master_test_suite().argv[1]);
        std::string implStr(
            boost::unit_test::framework::master_test_suite().argv[2]);

        BOOST_TEST_MESSAGE("Creating input and output fields");

        // Construct a fake command-line argument array to be fed to
        // Session::Reader::CreateInstance. The first element stands for
        // the name of the executable which, in our case, doesn't matter.
        int argc    = 4;
        char **argv = new char *[argc];
        argv[0]     = strdup("exe_name");
        argv[1]     = meshName.data();
        argv[2]     = strdup(("--opExecSpace=" + execStr).c_str());
        argv[3]     = strdup(("--opImpl=" + implStr).c_str());

        this->session = LibUtilities::SessionReader::CreateInstance(argc, argv);

        delete[] argv;

        if (this->session->GetComm())
        {
            auto rank        = this->session->GetComm()->GetRank();
            auto num_device  = nekGetDeviceCount();
            auto device_rank = rank % num_device;
            nekSetDevice(device_rank);
        }
    }

    void SetExpList(void)
    {
        auto graph = SpatialDomains::MeshGraphIO::Read(this->session);
        if constexpr (std::is_same_v<TExpList, MultiRegions::ContField>)
        {
            this->fixt_explist =
                MemoryManager<MultiRegions::ContField>::AllocateSharedPtr(
                    this->session, graph, "u", true, false,
                    Collections::eNoCollection);
        }
        else if constexpr (std::is_same_v<TExpList, MultiRegions::DisContField>)
        {
            this->fixt_explist =
                MemoryManager<MultiRegions::DisContField>::AllocateSharedPtr(
                    this->session, graph, "u", true, true,
                    Collections::eNoCollection);
        }
        else if constexpr (std::is_same_v<TExpList, MultiRegions::ExpList>)
        {
            this->fixt_explist =
                MemoryManager<MultiRegions::ExpList>::AllocateSharedPtr(
                    this->session, graph, true, "u",
                    Collections::eNoCollection);
        }
    }

    void SetExpList3DH1(const unsigned int nhomo = 1)
    {
        auto graph = SpatialDomains::MeshGraphIO::Read(this->session);
        if constexpr (std::is_same_v<TExpList, MultiRegions::ContField>)
        {
            const Nektar::LibUtilities::PointsKey pkey(
                nhomo, Nektar::LibUtilities::eFourierEvenlySpaced);
            const Nektar::LibUtilities::BasisKey bkey(
                Nektar::LibUtilities::eFourier, nhomo, pkey);
            this->fixt_explist =
                MemoryManager<MultiRegions::ContField3DHomogeneous1D>::
                    AllocateSharedPtr(this->session, bkey, 1.0, false, false,
                                      graph, "u", false,
                                      Collections::eNoCollection);
        }
        else if constexpr (std::is_same_v<TExpList, MultiRegions::DisContField>)
        {
            const Nektar::LibUtilities::PointsKey pkey(
                nhomo, Nektar::LibUtilities::eFourierEvenlySpaced);
            const Nektar::LibUtilities::BasisKey bkey(
                Nektar::LibUtilities::eFourier, nhomo, pkey);
            this->fixt_explist =
                MemoryManager<MultiRegions::DisContField3DHomogeneous1D>::
                    AllocateSharedPtr(this->session, bkey, 1.0, false, false,
                                      graph, "u", Collections::eNoCollection);
        }
        else if constexpr (std::is_same_v<TExpList, MultiRegions::ExpList>)
        {
            const Nektar::LibUtilities::PointsKey pkey(
                nhomo, Nektar::LibUtilities::eFourierEvenlySpaced);
            const Nektar::LibUtilities::BasisKey bkey(
                Nektar::LibUtilities::eFourier, nhomo, pkey);
            this->fixt_explist =
                MemoryManager<MultiRegions::ExpList3DHomogeneous1D>::
                    AllocateSharedPtr(this->session, bkey, 1.0, false, false,
                                      graph, "u", Collections::eNoCollection);
        }
        this->fixt_explist->SetWaveSpace(true);
    }

    void SetExpList3DH2(const unsigned int nhomoY = 1,
                        const unsigned int nhomoZ = 1)
    {
        auto graph = SpatialDomains::MeshGraphIO::Read(this->session);
        if constexpr (std::is_same_v<TExpList, MultiRegions::ContField>)
        {
            const Nektar::LibUtilities::PointsKey pkey1(
                nhomoY, Nektar::LibUtilities::eFourierEvenlySpaced);
            const Nektar::LibUtilities::BasisKey bkey1(
                Nektar::LibUtilities::eFourier, nhomoY, pkey1);
            const Nektar::LibUtilities::PointsKey pkey2(
                nhomoZ, Nektar::LibUtilities::eFourierEvenlySpaced);
            const Nektar::LibUtilities::BasisKey bkey2(
                Nektar::LibUtilities::eFourier, nhomoZ, pkey2);
            this->fixt_explist =
                MemoryManager<MultiRegions::ContField3DHomogeneous2D>::
                    AllocateSharedPtr(this->session, bkey1, bkey2, 1.0, 1.0,
                                      false, false, graph, "u", false,
                                      Collections::eNoCollection);
        }
        else if constexpr (std::is_same_v<TExpList, MultiRegions::DisContField>)
        {
            const Nektar::LibUtilities::PointsKey pkey1(
                nhomoY, Nektar::LibUtilities::eFourierEvenlySpaced);
            const Nektar::LibUtilities::BasisKey bkey1(
                Nektar::LibUtilities::eFourier, nhomoY, pkey1);
            const Nektar::LibUtilities::PointsKey pkey2(
                nhomoZ, Nektar::LibUtilities::eFourierEvenlySpaced);
            const Nektar::LibUtilities::BasisKey bkey2(
                Nektar::LibUtilities::eFourier, nhomoZ, pkey2);
            this->fixt_explist =
                MemoryManager<MultiRegions::DisContField3DHomogeneous2D>::
                    AllocateSharedPtr(this->session, bkey1, bkey2, 1.0, 1.0,
                                      false, false, graph, "u",
                                      Collections::eNoCollection);
        }
        else if constexpr (std::is_same_v<TExpList, MultiRegions::ExpList>)
        {
            const Nektar::LibUtilities::PointsKey pkey1(
                nhomoY, Nektar::LibUtilities::eFourierEvenlySpaced);
            const Nektar::LibUtilities::BasisKey bkey1(
                Nektar::LibUtilities::eFourier, nhomoY, pkey1);
            const Nektar::LibUtilities::PointsKey pkey2(
                nhomoZ, Nektar::LibUtilities::eFourierEvenlySpaced);
            const Nektar::LibUtilities::BasisKey bkey2(
                Nektar::LibUtilities::eFourier, nhomoZ, pkey2);
            this->fixt_explist =
                MemoryManager<MultiRegions::ExpList3DHomogeneous2D>::
                    AllocateSharedPtr(this->session, bkey1, bkey2, 1.0, 1.0,
                                      false, false, graph,
                                      Collections::eNoCollection);
        }
        this->fixt_explist->SetWaveSpace(true);
    }

    virtual void SetFixture(const unsigned int nhomo)
    {
        auto nin  = this->session->GetVariables().size();
        auto nout = this->session->GetVariables().size();
        auto inblockAttr =
            GetBlockAttributes<TData, stateIn>(this->fixt_explist);
        auto outblockAttr =
            GetBlockAttributes<TData, stateOut>(this->fixt_explist);

        auto f_in  = Field<TData, stateIn>("f_in", inblockAttr, nin, nhomo);
        auto f_out = Field<TData, stateOut>("f_out", outblockAttr, nout, nhomo);
        auto f_expected =
            Field<TData, stateOut>("f_expected", outblockAttr, nout, nhomo);
        this->fixt_in       = new Field<TData, stateIn>(std::move(f_in));
        this->fixt_out      = new Field<TData, stateOut>(std::move(f_out));
        this->fixt_expected = new Field<TData, stateOut>(std::move(f_expected));
    }

    /**
     * @brief Compare this field to another field, with absolute
     * tolerance tol. Two fields must have same storage shape
     * and same components.
     *
     * @return bool
     */
    bool Compare(TData tol)
    {
        auto rank = this->session->GetComm()->GetRank();

        if (this->fixt_expected->GetNumComponents() !=
            this->fixt_out->GetNumComponents())
        {
            std::cout << "Mismatch of number of components." << std::endl;
            return false;
        }

        if (this->fixt_expected->GetNumHomoModes() !=
            this->fixt_out->GetNumHomoModes())
        {
            std::cout << "Mismatch of number of homogeneous modes."
                      << std::endl;
            return false;
        }

        if (this->fixt_expected->GetBlocks().size() !=
            this->fixt_out->GetBlocks().size())
        {
            std::cout << "Mismatch of block size." << std::endl;
            return false;
        }

        ReshapeToScalar(*this->fixt_out);
        ReshapeToScalar(*this->fixt_expected);

        bool isMatch = true;

        if (rank == 0)
        {
            printf("#elm #pts output               expected            "
                   "difference\n");
        }
        for (unsigned int blk = 0; blk < this->fixt_out->GetBlocks().size();
             ++blk)
        {
            const TData *outptr =
                this->fixt_out->GetBlocks()[blk]
                    .template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
            const TData *expptr =
                this->fixt_expected->GetBlocks()[blk]
                    .template GetPtr<NektarSpaces::HostSpace, ReadOnly>();

            if ((this->fixt_out->GetBlocks()[blk].GetNumElements() !=
                 this->fixt_expected->GetBlocks()[blk].GetNumElements()) ||
                (this->fixt_out->GetBlocks()[blk].GetNumData() !=
                 this->fixt_expected->GetBlocks()[blk].GetNumData()))
            {
                std::cout << "Mismatch of block structure." << std::endl;
                return false;
            }

            for (unsigned int n = 0; n < this->fixt_out->GetNumComponents() *
                                             this->fixt_out->GetNumHomoModes();
                 ++n)
            {
                size_t MisMatchcnt = 0, total = 0;

                for (size_t el = 0, cnt = 0;
                     el < this->fixt_out->GetBlocks()[blk].GetNumElements();
                     ++el)
                {
                    for (unsigned int pts = 0;
                         pts < this->fixt_out->GetBlocks()[blk].GetNumData();
                         ++pts, ++cnt)
                    {
                        if (std::isnan(outptr[cnt]) ||
                            std::isinf(outptr[cnt]) ||
                            std::abs(outptr[cnt] - expptr[cnt]) > tol)
                        {
                            printf("%04lu %04u %20.16f %20.16f %20.16f\n", el,
                                   pts, outptr[cnt], expptr[cnt],
                                   std::abs(outptr[cnt] - expptr[cnt]));
                            MisMatchcnt++;
                        }
                        total++;
                    }
                }

                outptr += this->fixt_out->GetBlocks()[blk].CompSize();
                expptr += this->fixt_expected->GetBlocks()[blk].CompSize();

                if (MisMatchcnt)
                {
                    std::cout << "Number of mismatches in component " << n
                              << " on block " << blk << " is " << MisMatchcnt
                              << " out of " << total << " on rank: " << rank
                              << std::endl;
                    isMatch = false;
                }
            }
        }

        if (isMatch)
        {
            return true;
        }
        else
        {
            return false;
        }
    }

    template <FieldState state> void ReshapeToScalar(Field<TData, state> &in)
    {
        for (unsigned int blk = 0; blk < in.GetBlocks().size(); ++blk)
        {
            auto &block = in.GetBlocks()[blk];
            TData *inptr =
                block.template GetPtr<NektarSpaces::HostSpace, ReadWrite>();
            size_t numElmtsPad =
                block.GetNumElements() + block.GetNumPaddingElements();
            for (unsigned int component = 0; component < in.GetNumComponents();
                 component++)
            {
                ReshapeStorage<NektarSpaces::Serial>(
                    1, block.GetInterleaveWidth(),
                    numElmtsPad * in.GetNumHomoModes(), block.GetNumData(),
                    inptr +
                        component * block.CompSize() * in.GetNumHomoModes());
            }

            block.template SetInterleaveWidth<TData>(1);
        }
    }

protected:
    std::string meshName                  = "";
    Field<TData, stateIn> *fixt_in        = nullptr;
    Field<TData, stateOut> *fixt_out      = nullptr;
    Field<TData, stateOut> *fixt_expected = nullptr;
    std::shared_ptr<MultiRegions::ExpList> fixt_explist;
    LibUtilities::SessionReaderSharedPtr session;
};

#if defined(LOCALLY_DEFINED_BOOST_TEST_DYN_LINK)
#undef BOOST_TEST_DYN_LINK
#endif

#if defined(LOCALLY_DEFINED_BOOST_TEST_NO_MAIN)
#undef BOOST_TEST_NO_MAIN
#endif
