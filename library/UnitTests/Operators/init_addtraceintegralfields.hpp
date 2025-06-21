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

#include "Operators/AddTraceIntegral/AddTraceIntegralOp.hpp"

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
     * operator Delete previouisly defined fixt_in (also for Device) and
     * re-define input based on TraceExpList
     */
    void ReConfigure(unsigned int nin                   = 1,
                     [[maybe_unused]] unsigned int nout = 1)
    {
        fixt_explist->GetTrace()->SetDataWarehouse();

        const FieldState stateIn = FieldState::Phys;

        if (fixt_in)
        {
            delete fixt_in;
        }
        auto blocks_in =
            GetBlockAttributes<double>(stateIn, fixt_explist->GetTrace());
        auto f_in =
            Field<double, stateIn>::Create("f_in", blocks_in, nin, alignment);
        fixt_in = new Field<double, stateIn>(std::move(f_in));
    }

    void SetTestCase()
    {
        for (unsigned int blk = 0; blk < fixt_in->GetBlocks().size(); ++blk)
        {
            auto &block = fixt_in->GetBlocks()[blk];
            auto inptr =
                block.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
            for (size_t el = 0, cnt = 0; el < block.GetNumElements(); ++el)
            {
                for (unsigned int phys = 0; phys < block.GetNumData();
                     ++phys, ++cnt)
                {
                    inptr[cnt] = phys;
                }
            }
        }
        ExpectedSolution();
    }

    void RunTestCase()
    {
        auto op = AddTraceIntegralOp<double>::Create(fixt_explist);
        op->Apply(*fixt_in, *fixt_out);
    }

    void ExpectedSolution()
    {
        // Calculate expected result from Nektar++
        Array<OneD, double> inTracephys = fixt_in->ToArray();
        Array<OneD, double> outcoeffs(fixt_explist->GetNcoeffs(), 0.0);
        fixt_explist->AddTraceIntegral(inTracephys, outcoeffs);
        fixt_expected->CopyArray<NektarSpaces::HostSpace>(outcoeffs);
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
