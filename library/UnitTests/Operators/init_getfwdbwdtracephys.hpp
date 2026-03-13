///////////////////////////////////////////////////////////////////////////////
//
// File: init_getfwdbwdtracephys.hpp
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

#include "Operators/GetFwdBwdTracePhys/GetFwdBwdTracePhysOp.hpp"

using namespace Nektar::Operators;
using namespace Nektar::LibUtilities;
using namespace Nektar::MultiRegions;
using namespace Nektar;

template <typename TData>
class GetFwdBwdTracePhys
    : public InitFields<TData, FieldState::Phys, FieldState::Phys,
                        MultiRegions::DisContField>
{
public:
    GetFwdBwdTracePhys()
        : InitFields<TData, FieldState::Phys, FieldState::Phys,
                     MultiRegions::DisContField>()
    {
    }

    void Configure(void)
    {
        this->SetSession();
        this->SetExpList();
        this->fixt_explist->SetDataWarehouse();
        this->fixt_explist->GetTrace()->SetDataWarehouse();
        this->SetFixture(1);
    }

    void SetFixture(const unsigned int nhomo) override
    {
        auto nin  = this->session->GetVariables().size();
        auto nout = this->session->GetVariables().size();
        auto inblockAttr =
            GetBlockAttributes<TData, FieldState::Phys>(this->fixt_explist);
        auto outblockAttr = GetBlockAttributes<TData, FieldState::Phys>(
            this->fixt_explist->GetTrace());

        auto f_in =
            Field<TData, FieldState::Phys>("f_in", inblockAttr, nin, nhomo);
        auto f_out =
            Field<TData, FieldState::Phys>("f_out", outblockAttr, nout, nhomo);
        auto f_out_2 = Field<TData, FieldState::Phys>("f_out_2", outblockAttr,
                                                      nout, nhomo);
        auto f_expected = Field<TData, FieldState::Phys>(
            "f_expected", outblockAttr, nout, nhomo);
        this->fixt_in  = new Field<TData, FieldState::Phys>(std::move(f_in));
        this->fixt_out = new Field<TData, FieldState::Phys>(std::move(f_out));
        this->fixt_out_2 =
            new Field<TData, FieldState::Phys>(std::move(f_out_2));
        this->fixt_expected =
            new Field<TData, FieldState::Phys>(std::move(f_expected));
    }

    void SetTestCase()
    {
        // Set initial conditions.
        for (unsigned int blk = 0; blk < this->fixt_in->GetBlocks().size();
             ++blk)
        {
            auto &block = this->fixt_in->GetBlocks()[blk];
            auto inptr =
                block.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
            for (unsigned int n = 0; n < this->fixt_in->GetNumComponents() *
                                             this->fixt_in->GetNumHomoModes();
                 ++n)
            {
                for (size_t el = 0, cnt = 0; el < block.GetNumElements(); ++el)
                {
                    for (unsigned int phys = 0; phys < block.GetNumData();
                         ++phys, ++cnt)
                    {
                        inptr[cnt] = phys + n;
                    }
                }
                inptr += block.CompSize();
            }
        }

        // Compute expected solution.
        ExpectedSolution();
    }

    void RunTestCase()
    {
        auto TraceOp = GetFwdBwdTracePhysOp<TData>::Create(
            this->fixt_explist, this->session->GetVariables());
        TraceOp->Apply(*this->fixt_in, *this->fixt_out, *this->fixt_out_2);
    }

    void ExpectedSolution()
    {
        // Calculate expected result from Nektar++.
        const unsigned int numComp = this->fixt_in->GetNumComponents();
        const size_t nTracePointsTot =
            this->fixt_explist->GetTrace()->GetTotPoints();
        const size_t nphys = this->fixt_explist->GetTotPoints();

        Array<OneD, TData> inFwd = this->fixt_in->ToArray();

        Array<OneD, TData> Fwd(numComp * nTracePointsTot, 0.0);
        Array<OneD, TData> Bwd(numComp * nTracePointsTot, 0.0);
        Array<OneD, TData> tmpFwd;
        Array<OneD, TData> tmpBwd;

        for (unsigned int i = 0; i < numComp; ++i)
        {
            this->fixt_explist->ExpList::GetFwdBwdTracePhys(
                inFwd + i * nphys, tmpFwd = Fwd + i * nTracePointsTot,
                tmpBwd = Bwd + i * nTracePointsTot);
        }
        this->fixt_expected->template CopyArray<NektarSpaces::HostSpace>(Fwd);
    }
};

// clang-format off
#if defined(NEKTAR_ENABLE_SINGLE_PRECISION)
#define TESTFLOAT(type, filename)                                              \
    class type##float : public GetFwdBwdTracePhys<float>                       \
    {                                                                          \
    public:                                                                    \
        type##float()                                                          \
        {                                                                      \
            meshName = filename;                                               \
        }                                                                      \
    };
#else
#define TESTFLOAT(type, filename)
#endif
#if defined(NEKTAR_ENABLE_DOUBLE_PRECISION)
#define TESTDOUBLE(type, filename)                                             \
    class type : public GetFwdBwdTracePhys<double>                             \
    {                                                                          \
    public:                                                                    \
        type()                                                                 \
        {                                                                      \
            meshName = filename;                                               \
        }                                                                      \
    };
#else
#define TESTDOUBLE(type, filename)
#endif
#define TEST(type, filename)                                                   \
    TESTFLOAT(type, filename)                                                  \
    TESTDOUBLE(type, filename)
// clang-format on

TEST(Seg, "run/segment.xml")

TEST(SegSEM, "run/line_sem.xml")

TEST(Seg3D, "run/segment_3D.xml")

TEST(Quad, "run/square.xml")

TEST(Quad3D, "run/square_3D.xml")

TEST(QuadVarP, "run/square_varp.xml")

TEST(QuadSEM, "run/square_sem.xml")

TEST(Tri, "run/tri.xml")

TEST(Tri3D, "run/tri_3D.xml")

TEST(TriVarP, "run/tri_varp.xml")

TEST(TriNodal, "run/tri_nodal.xml")

TEST(SquareAllElements, "run/square_all_elements.xml")

TEST(Hex, "run/hex.xml")

TEST(HexVarP, "run/hex_varp.xml")

TEST(HexSEM, "run/hex_sem.xml")

TEST(Prism, "run/prism.xml")

TEST(PrismVarP, "run/prism_varp.xml")

TEST(PrismNodal, "run/prism_nodal.xml")

TEST(Pyr, "run/pyr.xml")

TEST(PyrVarP, "run/pyr_varp.xml")

TEST(Tet, "run/tet.xml")

TEST(TetVarP, "run/tet_varp.xml")

TEST(TetNodal, "run/tet_nodal.xml")

TEST(CubePrismHex, "run/cube_prismhex.xml")

TEST(CubeAllElements, "run/cube_all_elements.xml")
