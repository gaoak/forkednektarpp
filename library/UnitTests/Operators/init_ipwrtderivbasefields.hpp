///////////////////////////////////////////////////////////////////////////////
//
// File: init_ipwrtderivbasefields.hpp
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

#include "Operators/ElmtOps/IProductWRTDerivBase/IProductWRTDerivBaseOp.hpp"

using namespace Nektar::Operators;
using namespace Nektar::LibUtilities;
using namespace Nektar;

class IProductWRTDerivBaseField
    : public InitFields<double, FieldState::Phys, FieldState::Coeff>
{
public:
    IProductWRTDerivBaseField()
        : InitFields<double, FieldState::Phys, FieldState::Coeff>()
    {
    }

    void SetTestCase()
    {
        // expect coordim components for each input dimension
        unsigned int coordim  = fixt_explist->GetCoordim(0);
        unsigned int compSize = fixt_in->GetNumComponents() / coordim;

        for (unsigned int blk = 0; blk < fixt_in->GetBlocks().size(); ++blk)
        {
            auto &block = fixt_in->GetBlocks()[blk];
            double *inptr =
                block.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();

            for (unsigned int nc = 0; nc < compSize; ++nc)
            {
                for (unsigned int k = 0; k < coordim; k++)
                {
                    for (size_t el = 0, cnt = 0; el < block.GetNumElements();
                         ++el)
                    {
                        for (unsigned int phys = 0; phys < block.GetNumData();
                             ++phys, ++cnt)
                        {
                            inptr[cnt] = phys + k + nc;
                        }
                    }
                    inptr += block.size();
                }
            }
        }
        ExpectedSolution();
    }

    void RunTestCase()
    {
        auto op = IProductWRTDerivBaseOp<double>::Create(fixt_explist);
        op->Apply(*fixt_in, *fixt_out);
    }

    void ExpectedSolution()
    {
        // Calculate expected result from Nektar++
        size_t ncoeffs       = fixt_explist->GetNcoeffs();
        size_t nphys         = fixt_explist->GetTotPoints();
        unsigned int coordim = fixt_explist->GetCoordim(0);

        // expect coordim components for each input dimension
        unsigned int compSize = fixt_in->GetNumComponents() / coordim;

        Array<OneD, double> inphys = fixt_in->ToArray();
        Array<OneD, double> outcoeffs(ncoeffs * compSize, 0.0), tmp;
        Array<OneD, Array<OneD, double>> inphysarray(coordim);

        for (unsigned int i = 0; i < compSize; ++i)
        {
            inphysarray[0] = inphys + i * nphys * coordim;
            for (unsigned int j = 1; j < coordim; ++j)
            {
                inphysarray[j] = inphysarray[j - 1] + nphys;
            }
            fixt_explist->IProductWRTDerivBase(inphysarray,
                                               tmp = outcoeffs + i * ncoeffs);
        }
        fixt_expected->CopyArray<NektarSpaces::HostSpace>(outcoeffs);
    }
};

#define TEST(type, filename)                                                   \
    class type : public IProductWRTDerivBaseField                              \
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
