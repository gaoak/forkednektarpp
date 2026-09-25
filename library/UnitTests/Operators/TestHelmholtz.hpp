///////////////////////////////////////////////////////////////////////////////
//
// File: TestHelmholtz.hpp
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

#include <cmath>

#include <MultiRegions/ExpListHomogeneous1D.h>

#include "TestOp.hpp"

#include "Operators/ElmtOps/Helmholtz/HelmholtzOp.hpp"

using namespace Nektar;
using namespace Nektar::LibUtilities;
using namespace Nektar::Operators;

template <typename TData>
class TestHelmholtz : public TestOp<TData, FieldState::Coeff, FieldState::Coeff>
{
public:
    TestHelmholtz() = default;

    void SetTestCase()
    {
        // Set initial conditions.
        for (unsigned int blk = 0; blk < this->fixt_in->GetBlocks().size();
             ++blk)
        {
            auto &block = this->fixt_in->GetBlocks()[blk];
            auto inptr =
                block.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
            const unsigned int nhomo = this->fixt_in->GetNumHomoModes();

            for (unsigned int n = 0;
                 n < this->fixt_in->GetNumComponents() * nhomo; ++n)
            {
                // The planes of a component are consecutive, so n % nhomo is
                // the plane this slice sits on. Giving it a single Fourier
                // mode along z is what exercises the z terms at all: a
                // profile flat in z carries only the k = 0 mode, where they
                // vanish, and the test would pass without them.
                const TData zfac =
                    (nhomo == 1)
                        ? TData(1.0)
                        : static_cast<TData>(std::cos(
                              2.0 * M_PI * static_cast<double>(n % nhomo) /
                              static_cast<double>(nhomo)));

                for (size_t el = 0, cnt = 0; el < block.GetNumElements(); ++el)
                {
                    for (unsigned int coeff = 0; coeff < block.GetNumData();
                         ++coeff, ++cnt)
                    {
                        inptr[cnt] = coeff * zfac;
                    }
                }
                inptr += block.CompSize();
            }
        }

        // Get lambda from this->session or default to 10.0
        m_lambda = this->session->DefinesParameter("Lambda")
                       ? this->session->GetParameter("Lambda")
                       : 10.0;

        // Set up diffusion coefficient.
        const auto dim           = this->fixt_explist->GetShapeDimension();
        const auto coordDim      = this->fixt_explist->GetCoordim(0);
        const auto diffCoeffSize = coordDim * (coordDim + 1) / 2;
        m_diffCoeff.resize(diffCoeffSize);

        if (coordDim == 1)
        {
            m_diffCoeff[0] = 1.0; // D00
        }
        else if (coordDim == 2)
        {
            if (dim == 1)
            {
                m_diffCoeff[0] = 1.0; // D00
                m_diffCoeff[2] = 1.0; // D11
            }
            else
            {
                m_diffCoeff[0] = 2.0; // D00
                m_diffCoeff[2] = 3.0; // D11
            }
        }
        else
        {
            if (dim == 1)
            {
                m_diffCoeff[0] = 1.0; // D00
                m_diffCoeff[2] = 1.0; // D11
                m_diffCoeff[5] = 1.0; // D22
            }
            else
            {
                m_diffCoeff[0] = 2.0; // D00
                m_diffCoeff[2] = 3.0; // D11
                m_diffCoeff[5] = 4.0; // D22
            }
        }

        // Compute expected solution.
        ExpectedSolution();
    }

    void RunTestCase()
    {
        auto op = HelmholtzOp<TData>::Create(this->fixt_explist,
                                             this->session->GetVariables());
        op->SetLambda(m_lambda);
        op->SetDiffCoeff(m_diffCoeff);
        op->Apply(*this->fixt_in, *this->fixt_out);
    }

    void ExpectedSolution()
    {
        // Calculate expected result from Nektar++.
        const unsigned int numComp = this->fixt_in->GetNumComponents() *
                                     this->fixt_in->GetNumHomoModes();
        const size_t ncoeffs =
            this->fixt_explist->GetNcoeffs() / this->fixt_in->GetNumHomoModes();

        // Set up diffusion coefficient.
        StdRegions::FactorMap factors;
        if (this->fixt_explist->GetCoordim(0) == 1)
        {
            factors[StdRegions::eFactorLambda] = m_lambda;
        }
        else if (this->fixt_explist->GetCoordim(0) == 2)
        {
            factors[StdRegions::eFactorLambda]   = m_lambda;
            factors[StdRegions::eFactorCoeffD00] = m_diffCoeff[0];
            factors[StdRegions::eFactorCoeffD01] = m_diffCoeff[1];
            factors[StdRegions::eFactorCoeffD11] = m_diffCoeff[2];
        }
        else
        {
            factors[StdRegions::eFactorLambda]   = m_lambda;
            factors[StdRegions::eFactorCoeffD00] = m_diffCoeff[0];
            factors[StdRegions::eFactorCoeffD01] = m_diffCoeff[1];
            factors[StdRegions::eFactorCoeffD11] = m_diffCoeff[2];
            factors[StdRegions::eFactorCoeffD02] = m_diffCoeff[3];
            factors[StdRegions::eFactorCoeffD12] = m_diffCoeff[4];
            factors[StdRegions::eFactorCoeffD22] = m_diffCoeff[5];
        }

        Array<OneD, TData> incoeffs = this->fixt_in->ToArray();
        Array<OneD, TData> outcoeffs(numComp * ncoeffs);
        Array<OneD, TData> tmp;

        // The weak z-Laplacian of a 3DH1 field is (beta k)^2 M on each Fourier
        // mode, which rides the existing lambda: the reference transforms to
        // wave space, applies a plane-wise Helmholtz with lambda + (beta k)^2,
        // and transforms back. The operator reaches the same result by another
        // route, the mass matrix applied to minus the second z-derivative, so
        // this is a genuine cross-check.
        auto homoExpList =
            std::dynamic_pointer_cast<MultiRegions::ExpListHomogeneous1D>(
                this->fixt_explist);
        const unsigned int nhomo = this->fixt_in->GetNumHomoModes();
        const bool wave          = homoExpList && nhomo > 1;

        Array<OneD, NekDouble> lambdaK(nhomo, 0.0);
        if (wave)
        {
            const NekDouble beta = 2.0 * M_PI / homoExpList->GetHomoLen();
            for (unsigned int p = 0; p < nhomo; ++p)
            {
                const NekDouble betaK =
                    beta *
                    homoExpList->m_transposition->GetK(static_cast<int>(p));
                lambdaK[p] = m_lambda + betaK * betaK;
            }

            HomogeneousTrans(homoExpList, ncoeffs, incoeffs, true);
        }

        for (unsigned int i = 0; i < numComp; ++i)
        {
            // The planes of a component are consecutive, so i % nhomo picks
            // the Fourier mode this slice carries.
            if (wave)
            {
                factors[StdRegions::eFactorLambda] = lambdaK[i % nhomo];
            }

            size_t e      = 0;
            size_t offset = i * ncoeffs;
            for (const auto &block : this->fixt_expected->GetBlocks())
            {
                auto nmTot = this->fixt_explist->GetExp(e)->GetNcoeffs();
                for (size_t el = 0; el < block.GetNumElements(); ++el)
                {
                    StdRegions::StdMatrixKey mkey(
                        StdRegions::eHelmholtz,
                        this->fixt_explist->GetExp(e)->DetShapeType(),
                        *(this->fixt_explist->GetExp(e)), factors);
                    this->fixt_explist->GetExp(e)->GeneralMatrixOp(
                        incoeffs + offset, tmp = outcoeffs + offset, mkey);
                    e++;
                    offset += nmTot;
                }
            }
        }

        if (wave)
        {
            HomogeneousTrans(homoExpList, ncoeffs, outcoeffs, false);
        }

        this->fixt_expected->template CopyArray<NektarSpaces::HostSpace>(
            outcoeffs);
    }

    /// Transform @p coeffs between physical z and wave space, one component at
    /// a time; Homogeneous1DTrans works in NekDouble whatever TData is.
    void HomogeneousTrans(
        const std::shared_ptr<MultiRegions::ExpListHomogeneous1D> &homoExpList,
        const size_t ncoeffs, Array<OneD, TData> &coeffs, const bool forwards)
    {
        const unsigned int nhomo = this->fixt_in->GetNumHomoModes();
        const size_t nPerComp    = ncoeffs * nhomo;

        Array<OneD, NekDouble> in(nPerComp);
        Array<OneD, NekDouble> out(nPerComp);

        for (unsigned int c = 0; c < this->fixt_in->GetNumComponents(); ++c)
        {
            const size_t base = c * nPerComp;
            for (size_t j = 0; j < nPerComp; ++j)
            {
                in[j] = static_cast<NekDouble>(coeffs[base + j]);
            }

            homoExpList->Homogeneous1DTrans(static_cast<int>(nPerComp), in, out,
                                            forwards);

            for (size_t j = 0; j < nPerComp; ++j)
            {
                coeffs[base + j] = static_cast<TData>(out[j]);
            }
        }
    }

protected:
    TData m_lambda;
    std::vector<TData> m_diffCoeff;
};

// clang-format off
#if defined(NEKTAR_ENABLE_SINGLE_PRECISION)
#define TESTFLOAT(type, filename)                                              \
    class type##float : public TestHelmholtz<float>                            \
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
    class type : public TestHelmholtz<double>                                  \
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

template <typename TData> class TestHelmholtzFFT : public TestHelmholtz<TData>
{
public:
    TestHelmholtzFFT() = default;
};

// clang-format off
#if defined(NEKTAR_ENABLE_SINGLE_PRECISION)
#define TEST_FFTFLOAT(type, filename)                                          \
    class type##float : public TestHelmholtzFFT<float>                         \
    {                                                                          \
    public:                                                                    \
        type##float()                                                          \
        {                                                                      \
            meshName = filename;                                               \
        }                                                                      \
    };
#else
#define TEST_FFTFLOAT(type, filename)
#endif
#if defined(NEKTAR_ENABLE_DOUBLE_PRECISION)
#define TEST_FFTDOUBLE(type, filename)                                         \
    class type : public TestHelmholtzFFT<double>                               \
    {                                                                          \
    public:                                                                    \
        type()                                                                 \
        {                                                                      \
            meshName = filename;                                               \
        }                                                                      \
    };
#else
#define TEST_FFTDOUBLE(type, filename)
#endif
#define TEST_FFT(type, filename)                                               \
    TEST_FFTFLOAT(type, filename)                                              \
    TEST_FFTDOUBLE(type, filename)
// clang-format on

TEST_FFT(QuadFFT, "run/square.xml")
TEST_FFT(TriFFT, "run/tri.xml")
TEST_FFT(SquareAllElementsFFT, "run/square_all_elements.xml")
