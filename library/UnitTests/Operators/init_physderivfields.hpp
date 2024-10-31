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

    template <typename ExecSpace, typename Impl> void RunTestCase()
    {
        PhysDeriv<>::template create<ExecSpace, Impl>(fixt_explist)
            ->apply(*fixt_in, *fixt_out);
    }
};

class PhysDerivField1D : public PhysDerivField
{
public:
    PhysDerivField1D()
    {
    }

    void SetTestCase()
    {
        size_t el = 0, pts = 0;
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

        double *inptr =
            fixt_in->template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
        for (const auto &block : fixt_in->GetBlocks())
        {
            for (size_t e = 0, cnt = 0; e < block.num_elements; ++e, ++el)
            {
                size_t M = fixt_explist->GetExp(el)->GetNumPoints(0);
                for (size_t phys = 0; phys < block.num_pts;
                     ++phys, ++pts, ++cnt)
                {
                    double tmp = 0.0;
                    for (size_t i = 0; i < M / 2; i++)
                    {
                        for (size_t j = 0; j < M / 2; j++)
                        {
                            for (size_t k = 0; k < M / 2; k++)
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
            inptr += block.block_size;
        }
        NektarSolution();
    }

    void NektarSolution()
    {
        // Calculate expected result from Nektar++
        Array<OneD, double> inphys = fixt_in->toArray();
        Array<OneD, double> outphys(fixt_explist->GetCoordim(0) *
                                    fixt_explist->GetTotPoints());
        Array<OneD, double> outphys0 = outphys;
        Array<OneD, double> outphys1 = outphys0 + fixt_explist->GetTotPoints();
        Array<OneD, double> outphys2 = outphys1 + fixt_explist->GetTotPoints();
        fixt_explist->PhysDeriv(inphys, outphys0, outphys1, outphys2);
        fixt_expected->copyArray<NektarSpaces::HostSpace>(outphys);
    }

    void ExpectedSolution()
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

        size_t el = 0, pts = 0;
        double *expptr =
            fixt_expected
                ->template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
        for (const auto &block : fixt_expected->GetBlocks())
        {
            for (size_t n = 0; n < fixt_explist->GetCoordim(0); n++)
            {
                for (size_t e = 0, cnt = 0; e < block.num_elements; ++e)
                {
                    size_t M = fixt_explist->GetExp(el)->GetNumPoints(0);
                    for (size_t phys = 0; phys < block.num_pts;
                         ++phys, ++pts, ++cnt)
                    {
                        size_t index = pts + e * block.num_pts + phys;
                        double tmp   = 0.0;
                        for (size_t i = 1; i < M / 2; i++)
                        {
                            for (size_t j = 1; j < M / 2; j++)
                            {
                                for (size_t k = 1; k < M / 2; k++)
                                {
                                    if (n == 0 && i >= 1)
                                    {
                                        tmp += i * std::pow(x[index], i - 1) *
                                               std::pow(y[index], j) *
                                               std::pow(z[index], k);
                                    }
                                    if (n == 1 && j >= 1)
                                    {
                                        tmp += j * std::pow(x[index], i) *
                                               std::pow(y[index], j - 1) *
                                               std::pow(z[index], k);
                                    }

                                    if (n == 2 && k >= 1)
                                    {
                                        tmp += k * std::pow(x[index], i) *
                                               std::pow(y[index], j) *
                                               std::pow(z[index], k - 1);
                                    }
                                }
                            }
                        }
                        expptr[cnt] = tmp;
                    }
                }
                expptr += block.block_size;
            }
            pts += block.num_elements * block.num_pts;
            el += block.num_elements;
        }
    }
};

class PhysDerivField2D : public PhysDerivField
{
public:
    PhysDerivField2D()
    {
    }

    void SetTestCase()
    {
        size_t el = 0, pts = 0;
        auto coordim   = fixt_explist->GetCoordim(0);
        auto totpoints = fixt_explist->GetTotPoints();
        Array<OneD, double> x(totpoints);
        Array<OneD, double> y(totpoints);
        Array<OneD, double> z(totpoints);
        fixt_explist->GetCoords(x, y, z);
        if (coordim == 2)
        {
            Vmath::Fill(totpoints, 1.0, z, 1);
        }
        double *inptr =
            fixt_in->template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
        for (const auto &block : fixt_in->GetBlocks())
        {
            for (size_t e = 0, cnt = 0; e < block.num_elements; ++e, ++el)
            {
                size_t M = fixt_explist->GetExp(el)->GetNumPoints(0);
                size_t N = fixt_explist->GetExp(el)->GetNumPoints(1);
                size_t P = (coordim == 2) ? 0 : N;

                for (size_t phys = 0; phys < block.num_pts;
                     ++phys, ++pts, ++cnt)
                {
                    double tmp = 0.0;
                    for (size_t i = 0; i < M / 2; i++)
                    {
                        for (size_t j = 0; j < N / 2; j++)
                        {
                            for (size_t k = 0; k < P / 2; ++k)
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
            inptr += block.block_size;
        }
        NektarSolution();
    }

    void NektarSolution()
    {
        // Calculate expected result from Nektar++
        Array<OneD, double> inphys = fixt_in->toArray();
        Array<OneD, double> outphys(fixt_explist->GetCoordim(0) *
                                    fixt_explist->GetTotPoints());
        Array<OneD, double> outphys0 = outphys;
        Array<OneD, double> outphys1 = outphys0 + fixt_explist->GetTotPoints();
        Array<OneD, double> outphys2 = outphys1 + fixt_explist->GetTotPoints();
        fixt_explist->PhysDeriv(inphys, outphys0, outphys1, outphys2);
        fixt_expected->copyArray<NektarSpaces::HostSpace>(outphys);
    }

    void ExpectedSolution()
    {
        auto coordim   = fixt_explist->GetCoordim(0);
        auto totpoints = fixt_explist->GetTotPoints();
        Array<OneD, double> x(totpoints);
        Array<OneD, double> y(totpoints);
        Array<OneD, double> z(totpoints);
        fixt_explist->GetCoords(x, y, z);

        if (coordim == 2)
        {
            Vmath::Fill(totpoints, 1.0, z, 1);
        }

        size_t el = 0, pts = 0;
        double *expptr =
            fixt_expected
                ->template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
        for (const auto &block : fixt_expected->GetBlocks())
        {
            for (size_t n = 0; n < fixt_explist->GetCoordim(0); n++)
            {
                for (size_t e = 0, cnt = 0; e < block.num_elements; ++e)
                {
                    size_t M = fixt_explist->GetExp(el)->GetNumPoints(0);
                    size_t N = fixt_explist->GetExp(el)->GetNumPoints(1);
                    size_t P = (coordim == 2) ? 0 : N;
                    for (size_t phys = 0; phys < block.num_pts;
                         ++phys, ++pts, ++cnt)
                    {
                        size_t index = pts + e * block.num_pts + phys;
                        double tmp   = 0.0;
                        for (size_t i = 0; i < M / 2; i++)
                        {
                            for (size_t j = 0; j < N / 2; j++)
                            {
                                for (size_t k = 0; k < P / 2; k++)
                                {
                                    if (n == 0 && i >= 1)
                                    {
                                        tmp += i * std::pow(x[index], i - 1) *
                                               std::pow(y[index], j) *
                                               std::pow(z[index], k);
                                    }
                                    if (n == 1 && j >= 1)
                                    {
                                        tmp += j * std::pow(x[index], i) *
                                               std::pow(y[index], j - 1) *
                                               std::pow(z[index], k);
                                    }

                                    if (n == 2 && k >= 1)
                                    {
                                        tmp += k * std::pow(x[index], i) *
                                               std::pow(y[index], j) *
                                               std::pow(z[index], k - 1);
                                    }
                                }
                            }
                        }
                        expptr[cnt] = tmp;
                    }
                }
                expptr += block.block_size;
            }
            pts += block.num_elements * block.num_pts;
            el += block.num_elements;
        }
    }
};

class PhysDerivField3D : public PhysDerivField
{
public:
    PhysDerivField3D()
    {
    }

    void SetTestCase()
    {
        size_t el = 0, pts = 0;
        Array<OneD, double> x(fixt_explist->GetTotPoints());
        Array<OneD, double> y(fixt_explist->GetTotPoints());
        Array<OneD, double> z(fixt_explist->GetTotPoints());
        fixt_explist->GetCoords(x, y, z);
        double *inptr =
            fixt_in->template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
        for (const auto &block : fixt_in->GetBlocks())
        {
            for (size_t e = 0, cnt = 0; e < block.num_elements; ++e, ++el)
            {
                size_t M = fixt_explist->GetExp(el)->GetNumPoints(0);
                size_t N = fixt_explist->GetExp(el)->GetNumPoints(1);
                size_t K = fixt_explist->GetExp(el)->GetNumPoints(2);
                for (size_t phys = 0; phys < block.num_pts;
                     ++phys, ++pts, ++cnt)
                {
                    double tmp = 0.0;
                    for (size_t i = 0; i < M / 2; i++)
                    {
                        for (size_t j = 0; j < N / 2; j++)
                        {
                            for (size_t k = 0; k < K / 2; k++)
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
            inptr += block.block_size;
        }
        NektarSolution();
    }

    void NektarSolution()
    {
        // Calculate expected result from Nektar++
        Array<OneD, double> inphys = fixt_in->toArray();
        Array<OneD, double> outphys(fixt_explist->GetCoordim(0) *
                                    fixt_explist->GetTotPoints());
        Array<OneD, double> outphys0 = outphys;
        Array<OneD, double> outphys1 = outphys0 + fixt_explist->GetTotPoints();
        Array<OneD, double> outphys2 = outphys1 + fixt_explist->GetTotPoints();
        fixt_explist->PhysDeriv(inphys, outphys0, outphys1, outphys2);
        fixt_expected->copyArray<NektarSpaces::HostSpace>(outphys);
    }

    void ExpectedSolution()
    {
        Array<OneD, double> x(fixt_explist->GetTotPoints());
        Array<OneD, double> y(fixt_explist->GetTotPoints());
        Array<OneD, double> z(fixt_explist->GetTotPoints());
        fixt_explist->GetCoords(x, y, z);
        size_t el = 0, pts = 0;
        double *expptr =
            fixt_expected
                ->template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
        for (const auto &block : fixt_expected->GetBlocks())
        {
            for (size_t n = 0; n < fixt_explist->GetCoordim(0); n++)
            {
                for (size_t e = 0, cnt = 0; e < block.num_elements; ++e)
                {
                    size_t M = fixt_explist->GetExp(el + e)->GetNumPoints(0);
                    size_t N = fixt_explist->GetExp(el + e)->GetNumPoints(1);
                    size_t K = fixt_explist->GetExp(el + e)->GetNumPoints(2);
                    for (size_t phys = 0; phys < block.num_pts; ++phys, ++cnt)
                    {
                        size_t index = pts + e * block.num_pts + phys;
                        double tmp   = 0.0;
                        for (size_t i = 0; i < M / 2; i++)
                        {
                            for (size_t j = 0; j < N / 2; j++)
                            {
                                for (size_t k = 0; k < K / 2; k++)
                                {
                                    if (n == 0 && i >= 1)
                                    {
                                        tmp += i * std::pow(x[index], i - 1) *
                                               std::pow(y[index], j) *
                                               std::pow(z[index], k);
                                    }
                                    if (n == 1 && j >= 1)
                                    {
                                        tmp += j * std::pow(x[index], i) *
                                               std::pow(y[index], j - 1) *
                                               std::pow(z[index], k);
                                    }
                                    if (n == 2 && k >= 1)
                                    {
                                        tmp += k * std::pow(x[index], i) *
                                               std::pow(y[index], j) *
                                               std::pow(z[index], k - 1);
                                    }
                                }
                            }
                        }
                        expptr[cnt] = tmp;
                    }
                }
                expptr += block.block_size;
            }
            pts += block.num_elements * block.num_pts;
            el += block.num_elements;
        }
    }
};

#define TEST1D(type, filename)                                                 \
    class type : public PhysDerivField1D                                       \
    {                                                                          \
    public:                                                                    \
        type()                                                                 \
        {                                                                      \
            meshName = filename;                                               \
        }                                                                      \
    };

TEST1D(Seg, "run/segment.xml")

TEST1D(SegSEM, "run/line_sem.xml")

TEST1D(Seg3D, "run/segment_3D.xml")

#define TEST2D(type, filename)                                                 \
    class type : public PhysDerivField2D                                       \
    {                                                                          \
    public:                                                                    \
        type()                                                                 \
        {                                                                      \
            meshName = filename;                                               \
        }                                                                      \
    };

TEST2D(Quad, "run/square.xml")

TEST2D(QuadVarP, "run/square_varp.xml")

TEST2D(QuadSEM, "run/square_sem.xml")

TEST2D(Tri, "run/tri.xml")

TEST2D(Tri3D, "run/tri_3D.xml")

TEST2D(TriVarP, "run/tri_varp.xml")

TEST2D(SquareAllElements, "run/square_all_elements.xml")

#define TEST3D(type, filename)                                                 \
    class type : public PhysDerivField3D                                       \
    {                                                                          \
    public:                                                                    \
        type()                                                                 \
        {                                                                      \
            meshName = filename;                                               \
        }                                                                      \
    };

TEST3D(Hex, "run/hex.xml")

TEST3D(HexVarP, "run/hex_varp.xml")

TEST3D(HexSEM, "run/hex_sem.xml")

TEST3D(Prism, "run/prism.xml")

TEST3D(PrismVarP, "run/prism_varp.xml")

TEST3D(Pyr, "run/pyr.xml")

TEST3D(PyrVarP, "run/pyr_varp.xml")

TEST3D(Tet, "run/tet.xml")

TEST3D(TetVarP, "run/tet_varp.xml")

TEST3D(CubePrismHex, "run/cube_prismhex.xml")

TEST3D(CubeAllElements, "run/cube_all_elements.xml")
