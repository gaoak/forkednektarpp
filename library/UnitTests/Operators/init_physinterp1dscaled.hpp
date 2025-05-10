///////////////////////////////////////////////////////////////////////////////
//
// File: init_physinterp1dscaled.hpp
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

#include "Operators/ElmtOps/PhysInterp1DScaled/OperatorPhysInterp1DScaled.hpp"

using namespace Nektar::Operators;
using namespace Nektar::LibUtilities;
using namespace Nektar;

class PhysInterp1DScaledField
    : public InitFields<double, FieldState::Phys, FieldState::Phys>
{
public:
    PhysInterp1DScaledField()
        : InitFields<double, FieldState::Phys, FieldState::Phys>()
    {
    }

    void SetTestCase(double scale)
    {
        for (unsigned int blk = 0; blk < fixt_in->GetBlocks().size(); ++blk)
        {
            auto &block = fixt_in->GetBlocks()[blk];
            auto inptr =
                block.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();

            for (unsigned int nc = 0; nc < fixt_in->GetNumComponents(); ++nc)
            {
                for (size_t el = 0, cnt = 0; el < block.GetNumElements(); ++el)
                {
                    for (unsigned int phys = 0; phys < block.GetNumData();
                         ++phys, ++cnt)
                    {
                        inptr[cnt] = phys + nc;
                    }
                }
                inptr += block.size();
            }
        }
        ExpectedSolution(scale);
    }

    void RunTestCase(const double scale)
    {
        auto PhysInterp1DOp =
            OperatorPhysInterp1DScaled<double>::Create(fixt_explist);
        PhysInterp1DOp->SetScaleFactor(scale);
        PhysInterp1DOp->Apply(*fixt_in, *fixt_out);
    }

    void ExpectedSolution(double scale)
    {
        unsigned int compSize = fixt_in->GetNumComponents();
        size_t nphys          = fixt_explist->GetTotPoints();
        size_t nphys1D        = fixt_explist->Get1DScaledTotPoints(scale);

        // Calculate expected result from Nektar++
        Array<OneD, double> inphys = fixt_in->ToArray();
        Array<OneD, double> outphys(compSize * nphys1D), tmp;

        for (unsigned int i = 0; i < compSize; ++i)
        {
            fixt_explist->PhysInterp1DScaled(scale, inphys + i * nphys,
                                             tmp = outphys + i * nphys1D);
        }
        fixt_expected->CopyArray<NektarSpaces::HostSpace>(outphys);
    }
};

#define TEST(type, filename)                                                   \
    class type : public PhysInterp1DScaledField                                \
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
