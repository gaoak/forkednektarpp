///////////////////////////////////////////////////////////////////////////////
//
// File: TestCurlCurl.hpp
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

#include "TestOp.hpp"

#include <cmath>

#include <MultiRegions/ExpListHomogeneous1D.h>

#include <MultiRegions/ElmtOps/CurlCurl/CurlCurlOp.hpp>

using namespace Nektar;
using namespace Nektar::LibUtilities;
using namespace Nektar::MultiRegions;

template <typename TData>
class TestCurlCurl : public TestOp<TData, FieldState::Phys, FieldState::Phys>
{
public:
    TestCurlCurl() = default;

    void SetFixture(const unsigned int nhomo) override
    {
        auto nin  = NumCurlComponents();
        auto nout = nin;

        auto inblockAttr =
            GetBlockAttributes<TData, FieldState::Phys>(this->fixt_explist);
        auto outblockAttr =
            GetBlockAttributes<TData, FieldState::Phys>(this->fixt_explist);

        auto f_in =
            Field<TData, FieldState::Phys>("f_in", inblockAttr, nin, nhomo);
        auto f_out =
            Field<TData, FieldState::Phys>("f_out", outblockAttr, nout, nhomo);
        auto f_expected = Field<TData, FieldState::Phys>(
            "f_expected", outblockAttr, nout, nhomo);
        this->fixt_in  = new Field<TData, FieldState::Phys>(std::move(f_in));
        this->fixt_out = new Field<TData, FieldState::Phys>(std::move(f_out));
        this->fixt_expected =
            new Field<TData, FieldState::Phys>(std::move(f_expected));
    }

    void RunTestCase()
    {
        auto op = CurlCurlOp<TData>::Create(this->fixt_explist,
                                            this->session->GetVariables());
        op->Apply(*this->fixt_in, *this->fixt_out);
    }

    void SetTestCase()
    {
        // Set initial conditions.
        auto dim       = this->fixt_explist->GetCoordim(0);
        auto totpoints = this->fixt_explist->GetTotPoints();
        Array<OneD, TData> x(totpoints);
        Array<OneD, TData> y(totpoints);
        Array<OneD, TData> z(totpoints);
        this->fixt_explist->GetCoords(x, y, z);
        // A 3DH1 expansion has a real third coordinate although its planes
        // are two-dimensional, and the initial condition has to vary along
        // it for the z terms to be exercised at all.
        const bool isHomogeneous = this->fixt_in->GetNumHomoModes() > 1;
        if (dim == 2 && !isHomogeneous)
        {
            Vmath::Fill(totpoints, 1.0, z, 1);
        }

        size_t el = 0;
        for (unsigned int blk = 0; blk < this->fixt_in->GetBlocks().size();
             ++blk)
        {
            auto &block = this->fixt_in->GetBlocks()[blk];
            auto inptr =
                block.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();

            for (unsigned int n = 0; n < this->fixt_in->GetNumComponents(); ++n)
            {
                size_t pts = 0;
                for (unsigned int m = 0; m < this->fixt_in->GetNumHomoModes();
                     ++m)
                {
                    for (size_t e = 0, cnt = 0; e < block.GetNumElements(); ++e)
                    {
                        // set M[3] to the point in the zero direction
                        // otherwise to the points in the basis direction if
                        // that basis exists
                        unsigned int M[3];
                        M[0] = M[1] = M[2] =
                            this->fixt_explist->GetExp(el)->GetNumPoints(0);
                        for (unsigned int i = 1;
                             i < this->fixt_explist->GetExp(el)->GetNumBases();
                             ++i)
                        {
                            M[i] =
                                this->fixt_explist->GetExp(el)->GetNumPoints(i);
                        }

                        for (unsigned int phys = 0; phys < block.GetNumData();
                             ++phys, ++pts, ++cnt)
                        {
                            TData tmp = 0.0;
                            for (unsigned int i = 0; i < M[0] / 2; i++)
                            {
                                for (unsigned int j = 0; j < M[1] / 2; j++)
                                {
                                    for (unsigned int k = 0; k < M[2] / 2; ++k)
                                    {
                                        // The homogeneous direction carries a
                                        // Fourier basis over a unit period, so
                                        // it is driven by a mode it can
                                        // represent. A monomial in z is not
                                        // periodic: its Fourier derivative
                                        // rings, which would make this a test
                                        // of how alike the two sides' Gibbs
                                        // noise is rather than of the
                                        // operator.
                                        const TData zfac =
                                            isHomogeneous
                                                ? std::cos(2.0 * M_PI * k *
                                                           z[pts])
                                                : std::pow(z[pts], k);
                                        tmp += std::pow(x[pts], i) *
                                                   std::pow(y[pts], j) * zfac +
                                               n + m + blk;
                                    }
                                }
                            }
                            inptr[cnt] = tmp;
                        }
                    }
                    inptr += block.CompSize();
                }
            }
            el += block.GetNumElements();
        }

        // Compute expected solution.
        ExpectedSolution();
    }

    void ExpectedSolution()
    {
        // Calculate expected result from Nektar++
        const unsigned int numComp = this->fixt_in->GetNumComponents();
        const unsigned int dim     = NumCurlComponents();
        const size_t nphys         = this->fixt_explist->GetTotPoints();
        Array<OneD, TData> inphys  = this->fixt_in->ToArray();
        Array<OneD, TData> outphys(nphys * numComp, 0.0);

        // Build 2D arrays for input to CurlCurl.
        Array<OneD, Array<OneD, TData>> inphysmd(numComp), outphysmd(dim);
        for (unsigned int d = 0; d < numComp; ++d)
        {
            inphysmd[d] = inphys + d * nphys;
        }
        for (unsigned int d = 0; d < dim; ++d)
        {
            outphysmd[d] = outphys + d * nphys;
        }

        ASSERTL0(
            numComp == dim,
            "Need the same number of components and dimension for this test.");

        // Calculate \nabla \times \nabla \times u. On 3DH1 the input is in
        // physical space while SetExpList3DH1 leaves WaveSpace=true, so it
        // is turned off around the call: otherwise CurlCurl takes the planes
        // for Fourier coefficients and skips the transforms its
        // z-derivatives need.
        const bool is3DH1 =
            std::dynamic_pointer_cast<MultiRegions::ExpListHomogeneous1D>(
                this->fixt_explist) != nullptr &&
            this->fixt_in->GetNumHomoModes() > 1;

        if (is3DH1)
        {
            this->fixt_explist->SetWaveSpace(false);
        }

        this->fixt_explist->CurlCurl(inphysmd, outphysmd);

        if (is3DH1)
        {
            this->fixt_explist->SetWaveSpace(true);
        }

        this->fixt_expected->template CopyArray<NektarSpaces::HostSpace>(
            outphys);
    }

protected:
    /// How many components a curl takes here. It is the coordinate dimension
    /// of an ordinary expansion; a 3DH1 curl is three-dimensional although
    /// its planes are not, so the homogeneous fixture says three instead.
    virtual unsigned int NumCurlComponents() const
    {
        return this->fixt_explist->GetCoordim(0);
    }
};

// clang-format off
#if defined(NEKTAR_ENABLE_SINGLE_PRECISION)
#define TESTFLOAT(type, filename)                                              \
    class type##float : public TestCurlCurl<float>                             \
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
    class type : public TestCurlCurl<double>                                   \
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

// Curl Curl operator is not defined for 1D
// TEST(Seg, "run/segment.xml")
//
// TEST(SegSEM, "run/line_sem.xml")

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

template <typename TData> class TestCurlCurlFFT : public TestCurlCurl<TData>
{
public:
    TestCurlCurlFFT() = default;

protected:
    /// A curl on a 3DH1 expansion is three-dimensional: the planes carry two
    /// coordinates and the homogeneous direction the third.
    unsigned int NumCurlComponents() const override
    {
        return 3u;
    }
};

// clang-format off
#if defined(NEKTAR_ENABLE_SINGLE_PRECISION)
#define TEST_FFTFLOAT(type, filename)                                          \
    class type##float : public TestCurlCurlFFT<float>                          \
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
    class type : public TestCurlCurlFFT<double>                                \
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
