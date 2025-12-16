///////////////////////////////////////////////////////////////////////////////
//
// File: init_assmbscatrfields.hpp
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

#include "Operators/AssmbScatr/AssmbScatrOp.hpp"
#include "Operators/AssmbScatr/AssmbScatrOpImpl.hpp"
#include "Operators/AssmbScatr/AssmbScatrZeroDirOp.hpp"

#include <LibUtilities/LinearAlgebra/NekLinSysIter.h>
#include <MultiRegions/ContField.h>
#include <MultiRegions/GlobalLinSysIterativeFull.h>

#include <Operators/Common/Spaces.hpp>
#include <Operators/Utils/UtilsKernels.hpp>

using namespace Nektar::Operators;
using namespace Nektar::Operators::detail;
using namespace Nektar::LibUtilities;
using namespace Nektar::MultiRegions;
using namespace Nektar;

template <typename TData>
class AssmbScatrField
    : public InitFields<TData, FieldState::Coeff, FieldState::Coeff, ContField>
{

public:
    AssmbScatrField()
        : InitFields<TData, FieldState::Coeff, FieldState::Coeff, ContField>()
    {
    }

    ~AssmbScatrField()
    {
    }

    void SetTestCase(bool ZeroDir = false)
    {
        // Set initial conditions.
        std::string execStr = Operator<TData>::GetOpExecSpace(this->session);

        for (unsigned int blk = 0; blk < this->fixt_in->GetBlocks().size();
             ++blk)
        {
            auto &block = this->fixt_in->GetBlocks()[blk];
            auto inptr =
                block.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
            for (unsigned int nc = 0; nc < this->fixt_in->GetNumComponents();
                 ++nc)
            {
                for (size_t el = 0, cnt = 0; el < block.GetNumElements(); ++el)
                {
                    for (unsigned int coeff = 0; coeff < block.GetNumData();
                         ++coeff, ++cnt)
                    {
                        inptr[cnt] = coeff + nc;
                    }
                }
                inptr += block.size();
            }
        }

        // Compute expected solution.
        ExpectedSolution(ZeroDir);

        // reshape this->fixt_in
        if (execStr == "AVX")
        {
            for (unsigned int blk = 0; blk < this->fixt_in->GetBlocks().size();
                 ++blk)
            {
                auto &block = this->fixt_in->GetBlocks()[blk];
                auto inptr =
                    block.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
                for (unsigned int nc = 0;
                     nc < this->fixt_in->GetNumComponents(); ++nc)
                {
                    // reshuffle data into simd_t width for AVX check
                    ReshapeStorage<NektarSpaces::Serial>(
                        NektarSpaces::vector_width<NektarSpaces::AVX,
                                                   TData>::value,
                        block.GetInterleaveWidth(),
                        block.GetNumElementsWithPadding(), block.GetNumData(),
                        inptr);
                    inptr += block.size();
                }

                block.template SetInterleaveWidth<TData>(
                    NektarSpaces::vector_width<NektarSpaces::AVX,
                                               TData>::value);
            }
        }
    }

    void RunTestCase()
    {
        auto op = AssmbScatrOp<double>::Create(this->fixt_explist);
        op->Apply(*this->fixt_in, *this->fixt_out);
    }

    void RunTestCaseZeroDir()
    {
        auto op = AssmbScatrZeroDirOp<double>::Create(this->fixt_explist);
        op->Apply(*this->fixt_in, *this->fixt_out);
    }

    void ExpectedSolution(bool ZeroDir = false)
    {
        std::vector<std::string> variables = this->session->GetVariables();
        std::vector<std::shared_ptr<MultiRegions::ContField>> contfields;
        auto graph = SpatialDomains::MeshGraphIO::Read(this->session);
        unsigned int compSize = this->fixt_in->GetNumComponents();

        for (auto &variable : variables)
        {
            contfields.push_back(
                MemoryManager<MultiRegions::ContField>::AllocateSharedPtr(
                    this->session, graph, variable, true, false,
                    Collections::eNoCollection));
        }

        // Calculate expected result from Nektar++.
        size_t ncoeffs               = this->fixt_explist->GetNcoeffs();
        Array<OneD, double> incoeffs = this->fixt_in->ToArray();
        Array<OneD, double> outcoeffs(compSize * ncoeffs);

        for (unsigned int i = 0; i < variables.size(); ++i)
        {
            Array<OneD, double> tmp;
            auto map = contfields[i]->GetLocalToGlobalMap();
            map->Assemble(incoeffs + i * ncoeffs,
                          tmp = outcoeffs + i * ncoeffs);
            if (ZeroDir)
            {
                Vmath::Zero(map->GetNumGlobalDirBndCoeffs(),
                            tmp = outcoeffs + i * ncoeffs, 1);
            }
            map->GlobalToLocal(outcoeffs + i * ncoeffs,
                               tmp = outcoeffs + i * ncoeffs);
        }
        this->fixt_expected->template CopyArray<NektarSpaces::HostSpace>(
            outcoeffs);
    }
};

// clang-format off
#if defined(NEKTAR_ENABLE_SINGLE_PRECISION)
#define TESTFLOAT(type, filename)                                              \
    class type##float : public AssmbScatrField<float>                          \
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
    class type : public AssmbScatrField<double>                                \
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
