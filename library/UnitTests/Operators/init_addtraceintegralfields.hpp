///////////////////////////////////////////////////////////////////////////////
//
// File: init_addtraceintegralfields.hpp
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

#include "init_fields.hpp"

#include "Operators/OperatorAddTraceIntegral.hpp"

using namespace Nektar::Operators;
using namespace Nektar::LibUtilities;
using namespace Nektar;

class AddTraceIntegralField
    : public InitFields<double, FieldState::Phys, FieldState::Coeff,
                        MultiRegions::DisContField>
{
public:
    AddTraceIntegralField()
        : InitFields<double, FieldState::Phys, FieldState::Coeff,
                     MultiRegions::DisContField>()
    {
    }

    /*
     *  Re-Initialise the input blocks based on the Trace-ExpList for this
     * operator Delete previouisly defined fixt_in (also for CUDA) and re-define
     * input based on TraceExpList
     */
    void ReConfigure(size_t nin = 1, [[maybe_unused]] size_t nout = 1)
    {
        const FieldState stateIn = FieldState::Phys;

        if (testModule.find("Kokkos") != std::string::npos ||
            testModule.find("KOKKOS") != std::string::npos)
        {
            if (fixt_in)
            {
                delete fixt_in;
            }
            if (fixt_kokkos_in)
            {
                delete fixt_kokkos_in;
            }
            auto blocks_in =
                GetBlockAttributes<double>(stateIn, fixt_explist->GetTrace());
            auto f_in = Field<double, stateIn>::template create<
                NektarSpaces::HostSpace>("f_in", blocks_in, nin,
                                         NektarSpaces::KOKKOS::alignment);
            auto fkokkos_in = Field<double, stateIn>::template create<
                NektarSpaces::DeviceSpace>("fkokkos_in", blocks_in, nin,
                                           NektarSpaces::KOKKOS::alignment);
            fixt_in        = new Field<double, stateIn>(std::move(f_in));
            fixt_kokkos_in = new Field<double, stateIn>(std::move(fkokkos_in));
        }
        else if (testModule.find("CUDA") != std::string::npos)
        {
            if (fixt_in)
            {
                delete fixt_in;
            }
            if (fixt_cuda_in)
            {
                delete fixt_cuda_in;
            }
            auto blocks_in =
                GetBlockAttributes<double>(stateIn, fixt_explist->GetTrace());
            auto f_in = Field<double, stateIn>::template create<
                NektarSpaces::HostSpace>("f_in", blocks_in, nin,
                                         NektarSpaces::CUDA::alignment);
            auto fcuda_in = Field<double, stateIn>::template create<
                NektarSpaces::DeviceSpace>("fcuda_in", blocks_in, nin,
                                           NektarSpaces::CUDA::alignment);
            fixt_in      = new Field<double, stateIn>(std::move(f_in));
            fixt_cuda_in = new Field<double, stateIn>(std::move(fcuda_in));
        }
        else if (testModule.find("SYCL") != std::string::npos)
        {
            if (fixt_in)
            {
                delete fixt_in;
            }
            if (fixt_sycl_in)
            {
                delete fixt_sycl_in;
            }
            auto blocks_in =
                GetBlockAttributes<double>(stateIn, fixt_explist->GetTrace());
            auto f_in = Field<double, stateIn>::template create<
                NektarSpaces::HostSpace>("f_in", blocks_in, nin,
                                         NektarSpaces::SYCL::alignment);
            auto fsycl_in = Field<double, stateIn>::template create<
                NektarSpaces::DeviceSpace>("fsycl_in", blocks_in, nin,
                                           NektarSpaces::SYCL::alignment);
            fixt_in      = new Field<double, stateIn>(std::move(f_in));
            fixt_sycl_in = new Field<double, stateIn>(std::move(fsycl_in));
        }
        else if (testModule.find("AVX") != std::string::npos)
        {
            if (fixt_in)
            {
                delete fixt_in;
            }
            auto blocks_in =
                GetBlockAttributes<double>(stateIn, fixt_explist->GetTrace());
            auto f_in = Field<double, stateIn>::template create<
                NektarSpaces::HostSpace>("f_in", blocks_in, nin,
                                         NektarSpaces::AVX::alignment);
            fixt_in = new Field<double, stateIn>(std::move(f_in));
        }
        else
        {
            if (fixt_in)
            {
                delete fixt_in;
            }
            auto blocks_in =
                GetBlockAttributes<double>(stateIn, fixt_explist->GetTrace());
            auto f_in = Field<double, stateIn>::template create<
                NektarSpaces::HostSpace>("f_in", blocks_in, nin,
                                         NektarSpaces::Serial::alignment);
            fixt_in = new Field<double, stateIn>(std::move(f_in));
        }
    }

    void SetTestCase(const std::vector<BlockAttributes> &blocks, double *outptr,
                     bool padding = true)

    {
        for (auto const &block : blocks)
        {
            size_t cnt = 0;
            for (size_t el = 0; el < block.num_elements; ++el)
            {
                for (size_t phys = 0; phys < block.num_pts; ++phys, ++cnt)
                {
                    outptr[cnt] = phys;
                }
            }
            outptr += (padding) ? block.block_size : cnt;
        }
    }

    void ExpectedSolution(const std::vector<BlockAttributes> &blocks,
                          double *outptr)
    {
        auto fixt_explist_trace = fixt_explist->GetTrace();
        Array<OneD, NekDouble> inTracephys(fixt_explist_trace->GetNpoints(),
                                           0.0);
        Array<OneD, NekDouble> outFieldcoeffs(fixt_explist->GetNcoeffs(), 0.0);

        // Set test case
        SetTestCase(fixt_in->GetBlocks(), inTracephys.get(), false);

        // Calculate expected result from Nektar++
        fixt_explist->AddTraceIntegral(inTracephys, outFieldcoeffs);

        // Copy expected result from Array to fixt_expected
        double *ptr = outFieldcoeffs.get();
        for (auto const &block : blocks)
        {
            size_t cnt = 0;
            for (size_t el = 0; el < block.num_elements; ++el)
            {
                for (size_t coeff = 0; coeff < block.num_pts; ++coeff, ++cnt)
                {
                    outptr[cnt] = (*ptr++);
                }
            }
            outptr += block.block_size;
        }
    }
};

#define TEST(type, filename)                                                   \
    class type : public AddTraceIntegralField                                  \
    {                                                                          \
    public:                                                                    \
        type()                                                                 \
        {                                                                      \
            meshName = filename;                                               \
        }                                                                      \
    };

TEST(Seg, "run/segment.xml")

TEST(SegSEM, "run/line_sem.xml")

TEST(Quad, "run/square.xml")

TEST(QuadVarP, "run/square_varp.xml")

TEST(QuadSEM, "run/square_sem.xml")

TEST(Tri, "run/tri.xml")

TEST(TriVarP, "run/tri_varp.xml")

TEST(SquareAllElements, "run/square_all_elements.xml")

TEST(Hex, "run/hex.xml")

TEST(HexVarP, "run/hex_varp.xml")

TEST(HexSEM, "run/hex_sem.xml")

TEST(Prism, "run/prism.xml")

TEST(PrismVarP, "run/prism_varp.xml")

TEST(Pyr, "run/pyr.xml")

TEST(PyrVarP, "run/pyr_varp.xml")

TEST(Tet, "run/tet.xml")

TEST(TetVarP, "run/tet_varp.xml")

TEST(CubePrismHex, "run/cube_prismhex.xml")

TEST(CubeAllElements, "run/cube_all_elements.xml")
