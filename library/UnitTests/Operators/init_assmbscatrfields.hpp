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

class AssmbScatrField
    : public InitFields<double, FieldState::Coeff, FieldState::Coeff, ContField>
{

public:
    AssmbScatrField()
        : InitFields<double, FieldState::Coeff, FieldState::Coeff, ContField>()
    {
    }

    ~AssmbScatrField()
    {
    }

    void SetTestCase(bool ZeroDir = false)
    {
        // Set initial conditions.
        std::string execStr =
            session->GetCmdLineArgument<std::string>("opExecSpace");

        for (unsigned int blk = 0; blk < fixt_in->GetBlocks().size(); ++blk)
        {
            auto &block = fixt_in->GetBlocks()[blk];
            auto inptr =
                block.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
            for (unsigned int nc = 0; nc < fixt_in->GetNumComponents(); ++nc)
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

        // reshape fixt_in
        if (execStr == "AVX")
        {
            for (unsigned int blk = 0; blk < fixt_in->GetBlocks().size(); ++blk)
            {
                auto &block = fixt_in->GetBlocks()[blk];
                auto inptr =
                    block.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
                for (unsigned int nc = 0; nc < fixt_in->GetNumComponents();
                     ++nc)
                {
                    // reshuffle data into simd_t width for AVX check
                    ReshapeStorage<NektarSpaces::Serial>(
                        NektarSpaces::vector_width<double>::value,
                        block.GetInterleaveWidth(),
                        block.GetNumElementsWithPadding(), block.GetNumData(),
                        inptr);
                    inptr += block.size();
                }

                block.template SetInterleaveWidth<double>(
                    NektarSpaces::vector_width<double>::value);
            }
        }
    }

    template <typename ExecSpace> void RunTestCase()
    {
        auto op = AssmbScatrOp<double>::Create(fixt_explist[0]);
        // setup assembly map cg for multiple components.
        std::vector<MultiRegions::AssemblyMapCGSharedPtr> assemblyMap;
        for (auto &e : fixt_explist)
        {
            auto contfield =
                std::dynamic_pointer_cast<MultiRegions::ContField>(e);
            assemblyMap.push_back(contfield->GetLocalToGlobalMap());
        }
        op->SetAssemblyMap(assemblyMap);
        op->Apply(*fixt_in, *fixt_out);
    }

    template <typename ExecSpace> void RunTestCaseZeroDir()
    {
        auto op = AssmbScatrZeroDirOp<double>::Create(fixt_explist[0]);
        // setup assembly map cg for multiple components.
        std::vector<MultiRegions::AssemblyMapCGSharedPtr> assemblyMap;
        for (auto &e : fixt_explist)
        {
            auto contfield =
                std::dynamic_pointer_cast<MultiRegions::ContField>(e);
            assemblyMap.push_back(contfield->GetLocalToGlobalMap());
        }
        op->SetAssemblyMap(assemblyMap);
        op->Apply(*fixt_in, *fixt_out);
    }

    void ExpectedSolution(bool ZeroDir = false)
    {
        // Calculate expected result from Nektar++.
        int compSize                 = fixt_in->GetNumComponents();
        int ncoeffs                  = fixt_explist[0]->GetNcoeffs();
        Array<OneD, double> incoeffs = fixt_in->ToArray();
        Array<OneD, double> outcoeffs(compSize * ncoeffs);
        Array<OneD, NekDouble> tmp;

        for (int i = 0; i < compSize; ++i)
        {
            auto map = std::dynamic_pointer_cast<MultiRegions::ContField>(
                           fixt_explist[i])
                           ->GetLocalToGlobalMap();
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
        fixt_expected->CopyArray<NektarSpaces::HostSpace>(outcoeffs);
    }
};

#define TEST(type, filename)                                                   \
    class type : public AssmbScatrField                                        \
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
