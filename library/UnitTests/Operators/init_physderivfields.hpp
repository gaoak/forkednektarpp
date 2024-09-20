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
};

class PhysDerivField1D : public PhysDerivField
{
public:
    PhysDerivField1D()
    {
    }

    void SetTestCase(const std::vector<BlockAttributes> &blocks, double *inptr,
                     bool padding = true)
    {
        size_t el = 0, pts = 0;
        Array<OneD, double> x(fixt_explist->GetTotPoints());
        fixt_explist->GetCoords(x);
        for (auto const &block : blocks)
        {
            size_t cnt = 0;
            for (size_t e = 0; e < block.num_elements; ++e, ++el)
            {
                size_t M = fixt_explist->GetExp(el)->GetNumPoints(0);
                for (size_t phys = 0; phys < block.num_pts;
                     ++phys, ++pts, ++cnt)
                {
                    double tmp = 0.0;
                    for (size_t i = 0; i < M / 2; i++)
                    {
                        tmp += std::pow(x[pts], i);
                    }
                    inptr[cnt] = tmp;
                }
            }
            inptr += (padding) ? block.block_size : cnt;
        }
    }

    void NektarSolution(const std::vector<BlockAttributes> &blocks,
                        double *inptr)
    {
        Array<OneD, NekDouble> inphys(fixt_explist->GetTotPoints());
        Array<OneD, NekDouble> outphys(fixt_explist->GetCoordim(0) *
                                       fixt_explist->GetTotPoints());
        Array<OneD, NekDouble> outphys0 = outphys;
        Array<OneD, NekDouble> outphys1 = NullNekDouble1DArray;
        Array<OneD, NekDouble> outphys2 = NullNekDouble1DArray;

        // Set test case
        SetTestCase(fixt_in->GetBlocks(), inphys.get(), false);

        // Calculate expected result from Nektar++
        fixt_explist->PhysDeriv(inphys, outphys0, outphys1, outphys2);

        // Copy expected result from Array to pointer
        double *ptr = outphys.get();
        for (auto const &block : blocks)
        {
            size_t cnt = 0;
            for (size_t el = 0; el < block.num_elements; ++el)
            {
                for (size_t phys = 0; phys < block.num_pts; ++phys, ++cnt)
                {
                    inptr[cnt] = (*ptr++);
                }
            }
            inptr += block.block_size;
        }
    }

    void ExpectedSolution(const std::vector<BlockAttributes> &blocks,
                          double *inptr)
    {
        size_t el = 0, pts = 0;
        Array<OneD, double> x(fixt_explist->GetTotPoints());
        fixt_explist->GetCoords(x);
        for (auto const &block : blocks)
        {
            size_t cnt = 0;
            for (size_t e = 0; e < block.num_elements; ++e, ++el)
            {
                size_t M = fixt_explist->GetExp(el)->GetNumPoints(0);
                for (size_t phys = 0; phys < block.num_pts;
                     ++phys, ++pts, ++cnt)
                {
                    double tmp = 0.0;
                    for (size_t i = 1; i < M / 2; i++)
                    {
                        tmp += i * std::pow(x[pts], i - 1);
                    }
                    inptr[cnt] = tmp;
                }
            }
            inptr += block.block_size;
        }
    }
};

class PhysDerivField2D : public PhysDerivField
{
public:
    PhysDerivField2D()
    {
    }

    void SetTestCase(const std::vector<BlockAttributes> &blocks, double *inptr,
                     bool padding = true)
    {
        size_t el = 0, pts = 0;
        Array<OneD, double> x(fixt_explist->GetTotPoints());
        Array<OneD, double> y(fixt_explist->GetTotPoints());
        fixt_explist->GetCoords(x, y);
        for (auto const &block : blocks)
        {
            size_t cnt = 0;
            for (size_t e = 0; e < block.num_elements; ++e, ++el)
            {
                size_t M = fixt_explist->GetExp(el)->GetNumPoints(0);
                size_t N = fixt_explist->GetExp(el)->GetNumPoints(1);
                for (size_t phys = 0; phys < block.num_pts;
                     ++phys, ++pts, ++cnt)
                {
                    double tmp = 0.0;
                    for (size_t i = 0; i < M / 2; i++)
                    {
                        for (size_t j = 0; j < N / 2; j++)
                        {
                            tmp += std::pow(x[pts], i) * std::pow(y[pts], j);
                        }
                    }
                    inptr[cnt] = tmp;
                }
            }
            inptr += (padding) ? block.block_size : cnt;
        }
    }

    void NektarSolution(const std::vector<BlockAttributes> &blocks,
                        double *inptr)
    {
        Array<OneD, NekDouble> inphys(fixt_explist->GetTotPoints());
        Array<OneD, NekDouble> outphys(fixt_explist->GetCoordim(0) *
                                       fixt_explist->GetTotPoints());
        Array<OneD, NekDouble> outphys0 = outphys;
        Array<OneD, NekDouble> outphys1 =
            outphys0 + fixt_explist->GetTotPoints();
        Array<OneD, NekDouble> outphys2 = NullNekDouble1DArray;

        // Set test case
        SetTestCase(fixt_in->GetBlocks(), inphys.get(), false);

        // Calculate expected result from Nektar++
        fixt_explist->PhysDeriv(inphys, outphys0, outphys1, outphys2);

        // Copy expected result from Array to fixt_expected
        double *ptr = outphys.get();
        for (size_t k = 0; k < fixt_explist->GetCoordim(0); k++)
        {
            for (auto const &block : blocks)
            {
                size_t cnt = 0;
                for (size_t el = 0; el < block.num_elements; ++el)
                {
                    for (size_t phys = 0; phys < block.num_pts; ++phys, ++cnt)
                    {
                        inptr[cnt] = (*ptr++);
                    }
                }
                inptr += block.block_size;
            }
        }
    }

    void ExpectedSolution(const std::vector<BlockAttributes> &blocks,
                          double *inptr)
    {
        Array<OneD, double> x(fixt_explist->GetTotPoints());
        Array<OneD, double> y(fixt_explist->GetTotPoints());
        fixt_explist->GetCoords(x, y);
        for (size_t n = 0; n < fixt_explist->GetCoordim(0); n++)
        {
            size_t el = 0, pts = 0;
            for (auto const &block : blocks)
            {
                size_t cnt = 0;
                for (size_t e = 0; e < block.num_elements; ++e, ++el)
                {
                    size_t M = fixt_explist->GetExp(el)->GetNumPoints(0);
                    size_t N = fixt_explist->GetExp(el)->GetNumPoints(1);
                    for (size_t phys = 0; phys < block.num_pts;
                         ++phys, ++pts, ++cnt)
                    {
                        double tmp = 0.0;
                        for (size_t i = 0; i < M / 2; i++)
                        {
                            for (size_t j = 0; j < N / 2; j++)
                            {
                                if (n == 0 && i >= 1)
                                {
                                    tmp += i * std::pow(x[pts], i - 1) *
                                           std::pow(y[pts], j);
                                }
                                if (n == 1 && j >= 1)
                                {
                                    tmp += j * std::pow(x[pts], i) *
                                           std::pow(y[pts], j - 1);
                                }
                            }
                        }
                        inptr[cnt] = tmp;
                    }
                }
                inptr += block.block_size;
            }
        }
    }
};

class PhysDerivField3D : public PhysDerivField
{
public:
    PhysDerivField3D()
    {
    }

    void SetTestCase(const std::vector<BlockAttributes> &blocks, double *inptr,
                     bool padding = true)
    {
        size_t el = 0, pts = 0;
        Array<OneD, double> x(fixt_explist->GetTotPoints());
        Array<OneD, double> y(fixt_explist->GetTotPoints());
        Array<OneD, double> z(fixt_explist->GetTotPoints());
        fixt_explist->GetCoords(x, y, z);
        for (auto const &block : blocks)
        {
            size_t cnt = 0;
            for (size_t e = 0; e < block.num_elements; ++e, ++el)
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
            inptr += (padding) ? block.block_size : cnt;
        }
    }

    void NektarSolution(const std::vector<BlockAttributes> &blocks,
                        double *inptr)
    {
        Array<OneD, NekDouble> inphys(fixt_explist->GetTotPoints());
        Array<OneD, NekDouble> outphys(fixt_explist->GetCoordim(0) *
                                       fixt_explist->GetTotPoints());
        Array<OneD, NekDouble> outphys0 = outphys;
        Array<OneD, NekDouble> outphys1 =
            outphys0 + fixt_explist->GetTotPoints();
        Array<OneD, NekDouble> outphys2 =
            outphys1 + fixt_explist->GetTotPoints();

        // Set test case
        SetTestCase(fixt_in->GetBlocks(), inphys.get(), false);

        // Calculate expected result from Nektar++
        fixt_explist->PhysDeriv(inphys, outphys0, outphys1, outphys2);

        // Copy expected result from Array to fixt_expected
        double *ptr = outphys.get();
        for (size_t k = 0; k < fixt_explist->GetCoordim(0); k++)
        {
            for (auto const &block : blocks)
            {
                size_t cnt = 0;
                for (size_t el = 0; el < block.num_elements; ++el)
                {
                    for (size_t phys = 0; phys < block.num_pts; ++phys, ++cnt)
                    {
                        inptr[cnt] = (*ptr++);
                    }
                }
                inptr += block.block_size;
            }
        }
    }

    void ExpectedSolution(const std::vector<BlockAttributes> &blocks,
                          double *inptr)
    {
        Array<OneD, double> x(fixt_explist->GetTotPoints());
        Array<OneD, double> y(fixt_explist->GetTotPoints());
        Array<OneD, double> z(fixt_explist->GetTotPoints());
        fixt_explist->GetCoords(x, y, z);
        for (size_t n = 0; n < fixt_explist->GetCoordim(0); n++)
        {
            size_t el = 0, pts = 0;
            for (auto const &block : blocks)
            {
                size_t cnt = 0;
                for (size_t e = 0; e < block.num_elements; ++e, ++el)
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
                                    if (n == 0 && i >= 1)
                                    {
                                        tmp += i * std::pow(x[pts], i - 1) *
                                               std::pow(y[pts], j) *
                                               std::pow(z[pts], k);
                                    }
                                    if (n == 1 && j >= 1)
                                    {
                                        tmp += j * std::pow(x[pts], i) *
                                               std::pow(y[pts], j - 1) *
                                               std::pow(z[pts], k);
                                    }
                                    if (n == 2 && k >= 1)
                                    {
                                        tmp += k * std::pow(x[pts], i) *
                                               std::pow(y[pts], j) *
                                               std::pow(z[pts], k - 1);
                                    }
                                }
                            }
                        }
                        inptr[cnt] = tmp;
                    }
                }
                inptr += block.block_size;
            }
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
