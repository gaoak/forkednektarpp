///////////////////////////////////////////////////////////////////////////////
//
// File: init_physderivfields.hpp
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

#include "Operators/ElmtOps/OperatorPhysDeriv.hpp"

using namespace Nektar::Operators;
using namespace Nektar::LibUtilities;
using namespace Nektar;

class PhysDerivField
    : public InitFields<double, FieldState::Phys, FieldState::Phys>
{
public:
    PhysDerivField() : InitFields<double, FieldState::Phys, FieldState::Phys>()
    {
    }

    void RunTestCase()
    {
        auto PhysDerivOp = OperatorPhysDeriv<double>::Create(fixt_explist);
        PhysDerivOp->apply(*fixt_in, *fixt_out);
    }
    void SetTestCase()
    {
        auto coordim   = fixt_explist->GetCoordim(0);
        auto totpoints = fixt_explist->GetTotPoints();
        Array<OneD, double> x(totpoints);
        Array<OneD, double> y(totpoints);
        Array<OneD, double> z(totpoints);
        fixt_explist->GetCoords(x, y, z);
        if (coordim == 1)
        {
            Vmath::Fill(totpoints, 1.0, y, 1);
            Vmath::Fill(totpoints, 1.0, z, 1);
        }
        else if (coordim == 2)
        {
            Vmath::Fill(totpoints, 1.0, z, 1);
        }

        unsigned int el = 0, exp_pts = 0;
        for (unsigned int blk = 0; blk < fixt_in->GetBlocks().size(); ++blk)
        {
            auto &block = fixt_in->GetBlocks()[blk];
            auto inptr =
                block.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();

            for (unsigned int nc = 0; nc < fixt_in->GetNumComponents(); ++nc)
            {
                for (unsigned int e = 0, cnt = 0; e < block.GetNumElements();
                     ++e)
                {
                    // set M[3] to the point in the zero direction
                    // otherwise to the points in the basis direction if
                    // that basis exists
                    unsigned int M[3];
                    M[0] = M[1] = M[2] =
                        fixt_explist->GetExp(el)->GetNumPoints(0);
                    for (unsigned int i = 1;
                         i < fixt_explist->GetExp(el)->GetNumBases(); ++i)
                    {
                        M[i] = fixt_explist->GetExp(el)->GetNumPoints(i);
                    }

                    for (unsigned int phys = 0, pts = exp_pts;
                         phys < block.GetNumData(); ++phys, ++pts, ++cnt)
                    {
                        double tmp = 0.0;
                        for (unsigned int i = 0; i < M[0] / 2; i++)
                        {
                            for (unsigned int j = 0; j < M[1] / 2; j++)
                            {
                                for (unsigned int k = 0; k < M[2] / 2; ++k)
                                {
                                    tmp += std::pow(x[pts], i) *
                                           std::pow(y[pts], j) *
                                           std::pow(z[pts], k);
                                }
                            }
                        }
                        inptr[cnt] = tmp;
                    }
                }
                inptr += block.size();
            }
            el += block.GetNumElements();
            exp_pts += block.GetNumData();
        }
        NektarSolution();
    }
    void NektarSolution()
    {
        // Calculate expected result from Nektar++
        int compSize               = fixt_in->GetNumComponents();
        int nphys                  = fixt_explist->GetTotPoints();
        int coordim                = fixt_explist->GetCoordim(0);
        Array<OneD, double> inphys = fixt_in->ToArray();
        Array<OneD, double> outphys(compSize * coordim * nphys);

        for (int i = 0; i < compSize; ++i)
        {
            Array<OneD, double> outphys0 = outphys + i * nphys * coordim;
            Array<OneD, double> outphys1 = outphys0 + nphys;
            Array<OneD, double> outphys2 = outphys1 + nphys;
            fixt_explist->PhysDeriv(inphys + i * nphys, outphys0, outphys1,
                                    outphys2);
        }
        fixt_expected->CopyArray<NektarSpaces::HostSpace>(outphys);
    }
};

#define TEST(type, filename)                                                   \
    class type : public PhysDerivField                                         \
    {                                                                          \
    public:                                                                    \
        type()                                                                 \
        {                                                                      \
            meshName = filename;                                               \
        }                                                                      \
    };

TEST(Seg, "run/segment.xml")

TEST(SegSEM, "run/line_sem.xml")

TEST(Seg3D, "run/segment_3D.xml")

TEST(Quad, "run/square.xml")

TEST(QuadVarP, "run/square_varp.xml")

TEST(QuadSEM, "run/square_sem.xml")

TEST(Tri, "run/tri.xml")

TEST(Tri3D, "run/tri_3D.xml")

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
