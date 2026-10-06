///////////////////////////////////////////////////////////////////////////////
//
// File: TestIProductWRTDerivBase.hpp
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

#include <MultiRegions/ExpListHomogeneous1D.h>

#include "TestOp.hpp"

#include <MultiRegions/ElmtOps/IProductWRTBase/IProductWRTBaseOp.hpp>
#include <MultiRegions/ElmtOps/IProductWRTDerivBase/IProductWRTDerivBaseOp.hpp>

using namespace Nektar;
using namespace Nektar::LibUtilities;
using namespace Nektar::MultiRegions;

template <typename TData>
class TestIProductWRTDerivBase
    : public TestOp<TData, FieldState::Phys, FieldState::Coeff>
{
public:
    TestIProductWRTDerivBase() = default;

private:
    /// Direction components the field carries: three on a multi-plane 3DH1
    /// expansion, where the homogeneous direction adds one the planes
    /// themselves do not count, and the plane's own dimension otherwise.
    unsigned int m_coordDim;
    Field<TData, FieldState::Phys> *m_f_phys = nullptr;

public:
    void SetFixture(const unsigned int nhomo) override
    {
        // A 3DH1 expansion carries a third direction along the homogeneous
        // direction, which the planes themselves do not count.
        const bool is3DH1 =
            std::dynamic_pointer_cast<MultiRegions::ExpListHomogeneous1D>(
                this->fixt_explist) != nullptr;
        m_coordDim =
            (is3DH1 && nhomo > 1)
                ? 3u
                : static_cast<unsigned int>(this->fixt_explist->GetCoordim(0));

        auto nin  = this->session->GetVariables().size() * m_coordDim;
        auto nout = this->session->GetVariables().size();
        auto inblockAttr =
            GetBlockAttributes<TData, FieldState::Phys>(this->fixt_explist);
        auto outblockAttr =
            GetBlockAttributes<TData, FieldState::Coeff>(this->fixt_explist);

        auto f_in =
            Field<TData, FieldState::Phys>("f_in", inblockAttr, nin, nhomo);
        auto f_out =
            Field<TData, FieldState::Coeff>("f_out", outblockAttr, nout, nhomo);
        auto f_expected = Field<TData, FieldState::Coeff>(
            "f_expected", outblockAttr, nout, nhomo);
        this->fixt_in  = new Field<TData, FieldState::Phys>(std::move(f_in));
        this->fixt_out = new Field<TData, FieldState::Coeff>(std::move(f_out));
        this->fixt_expected =
            new Field<TData, FieldState::Coeff>(std::move(f_expected));

        auto f_phys =
            Field<TData, FieldState::Phys>("f_phys", inblockAttr, nout, nhomo);
        m_f_phys = new Field<TData, FieldState::Phys>(std::move(f_phys));
    }

    void SetTestCase()
    {
        // Set initial conditions.
        for (unsigned int blk = 0; blk < this->fixt_in->GetBlocks().size();
             ++blk)
        {
            auto &block = this->fixt_in->GetBlocks()[blk];
            TData *inptr =
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
        auto op = IProductWRTDerivBaseOp<FieldState::Coeff, TData>::Create(
            this->fixt_explist, this->session->GetVariables());
        op->Apply(*this->fixt_in, *this->fixt_out);
    }

    void RunTestCaseApplyPhys()
    {
        auto op = IProductWRTDerivBaseOp<FieldState::Phys, TData>::Create(
            this->fixt_explist, this->session->GetVariables());
        auto BT = IProductWRTBaseOp<TData>::Create(
            this->fixt_explist, this->session->GetVariables());
        BT->SetIntegration(false);

        op->Apply(*this->fixt_in, *m_f_phys);
        BT->Apply(*m_f_phys, *this->fixt_out);
    }

    void ExpectedSolution()
    {
        // Calculate expected result from Nektar++
        const size_t ncoeffs       = this->fixt_explist->GetNcoeffs();
        const size_t nphys         = this->fixt_explist->GetTotPoints();
        const unsigned int numComp = this->fixt_out->GetNumComponents();
        const unsigned int nhomo   = this->fixt_in->GetNumHomoModes();

        // A 3DH1 expansion carries a third direction, and the z part of the
        // weak derivative needs the input in physical space, while
        // SetExpList3DH1 leaves WaveSpace true.
        const bool is3DH1 =
            std::dynamic_pointer_cast<MultiRegions::ExpListHomogeneous1D>(
                this->fixt_explist) != nullptr;
        const bool wave = is3DH1 && nhomo > 1;

        Array<OneD, TData> inphys = this->fixt_in->ToArray();
        Array<OneD, TData> outcoeffs(ncoeffs * numComp, 0.0), tmp;
        Array<OneD, Array<OneD, TData>> inphysarray(m_coordDim);

        for (unsigned int i = 0; i < numComp; ++i)
        {
            inphysarray[0] = inphys + i * nphys * m_coordDim;
            for (unsigned int j = 1; j < m_coordDim; ++j)
            {
                inphysarray[j] = inphysarray[j - 1] + nphys;
            }
            if (wave)
            {
                // Legacy splits the two spaces differently from the
                // redesign. v_PhysDeriv takes x and y plane by plane and
                // only transforms for z, which is what the redesign does
                // throughout, but v_IProductWRTDerivBase forward transforms
                // every direction because its result is defined in wave
                // space. The eFourier basis carries no Nyquist mode, so that
                // round trip is a projection rather than the identity and
                // would drop xy content the redesign keeps. The directions
                // are therefore taken one at a time, each in its own space.
                Array<OneD, TData> acc(ncoeffs, 0.0), dir(ncoeffs, 0.0),
                    dirz(ncoeffs, 0.0);

                // x and y: plane local, no transform either way.
                this->fixt_explist->SetWaveSpace(true);
                this->fixt_explist->IProductWRTDerivBase(0, inphysarray[0],
                                                         acc);
                this->fixt_explist->IProductWRTDerivBase(1, inphysarray[1],
                                                         dir);
                Vmath::Vadd(ncoeffs, dir, 1, acc, 1, acc, 1);

                // z: forward transform, weight by the wavenumber, and come
                // back, exactly as DerivZOp does.
                this->fixt_explist->SetWaveSpace(false);
                this->fixt_explist->IProductWRTDerivBase(2, inphysarray[2],
                                                         dir);
                this->fixt_explist->SetWaveSpace(true);
                this->fixt_explist->HomogeneousBwdTrans(ncoeffs, dir, dirz);
                Vmath::Vadd(ncoeffs, dirz, 1, acc, 1, acc, 1);

                Vmath::Vcopy(ncoeffs, acc, 1, tmp = outcoeffs + i * ncoeffs, 1);
            }
            else
            {
                this->fixt_explist->IProductWRTDerivBase(
                    inphysarray, tmp = outcoeffs + i * ncoeffs);
            }
        }

        this->fixt_expected->template CopyArray<NektarSpaces::HostSpace>(
            outcoeffs);
    }
};

// clang-format off
#if defined(NEKTAR_ENABLE_SINGLE_PRECISION)
#define TESTFLOAT(type, filename)                                              \
    class type##float : public TestIProductWRTDerivBase<float>                 \
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
    class type : public TestIProductWRTDerivBase<double>                       \
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

template <typename TData>
class TestIProductWRTDerivBaseFFT : public TestIProductWRTDerivBase<TData>
{
public:
    TestIProductWRTDerivBaseFFT() = default;
};

// clang-format off
#if defined(NEKTAR_ENABLE_SINGLE_PRECISION)
#define TEST_FFTFLOAT(type, filename)                                          \
    class type##float : public TestIProductWRTDerivBaseFFT<float>              \
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
    class type : public TestIProductWRTDerivBaseFFT<double>                    \
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
