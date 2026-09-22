///////////////////////////////////////////////////////////////////////////////
//
// File: Deriv2ZOpSerialAVX.hpp
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
// Description: Host (FFTW) backend for the homogeneous second z-derivative.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <LibUtilities/BasicUtils/ErrorUtil.hpp>

namespace Nektar::Operators::detail
{

/// \brief Host FFTW backend for minus the second z-derivative of a 3DH1
/// field, shared by the operators whose z-coupling is the weak Laplacian.
///
/// The weak z-Laplacian of \f$u=\sum_k \hat u_k(x,y)e^{i\beta k z}\f$ is
/// \f$(\beta k)^2 M_{xy}\hat u_k\f$, which is the xy mass matrix applied to
/// \f$-\partial_z^2 u\f$. Launch() writes that second derivative, leaving its
/// caller to apply the mass matrix and add the result to the xy part.
///
/// The multiplier is \f$(\beta k)^2\f$, real, so a mode stays in its own
/// plane rather than moving to its conjugate partner as it does for the first
/// derivative. Nektar's eFourier basis holds the constant in plane 0 and a
/// structural zero in plane 1, and GetK() reports 0 for both, so both fall
/// out of the multiply on their own.
///
/// Launch() gathers each component's unpadded per-plane data from all blocks
/// into one contiguous array, applies the forward transform, wavenumber
/// multiply and inverse transform over the full field, then scatters the
/// result. The gather is needed because Homogeneous1DTrans' transposition
/// object is built for the full-field count and cannot operate on per-block
/// subsets; it works on coefficients as readily as on quadrature points, as
/// ExpListHomogeneous1D::v_FwdTrans relies on.
template <typename ExecSpace, typename TData>
class Deriv2ZOpImpl<
    ExecSpace, TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Serial> ||
                     std::is_same_v<ExecSpace, NektarSpaces::AVX>>>
{
public:
    /// \param expansionList  Cast to ExpListHomogeneous1D for the FFT and
    ///                       transposition objects.
    Deriv2ZOpImpl(const MultiRegions::ExpListSharedPtr &expansionList)
    {
        m_homoExpList =
            std::dynamic_pointer_cast<MultiRegions::ExpListHomogeneous1D>(
                expansionList);

        ASSERTL0(m_homoExpList, "The homogeneous second z-derivative needs an "
                                "ExpListHomogeneous1D");

        m_beta = 2.0 * M_PI / m_homoExpList->GetHomoLen();
    }

    // Non-copyable and non-movable.
    Deriv2ZOpImpl(const Deriv2ZOpImpl &)            = delete;
    Deriv2ZOpImpl &operator=(const Deriv2ZOpImpl &) = delete;
    Deriv2ZOpImpl(Deriv2ZOpImpl &&)                 = delete;
    Deriv2ZOpImpl &operator=(Deriv2ZOpImpl &&)      = delete;

    /// Write out = -d2(in)/dz2, component for component.
    void Launch(LibUtilities::Field<TData, FieldState::Coeff> &in,
                LibUtilities::Field<TData, FieldState::Coeff> &out)
    {
        const unsigned int nhomo = in.GetNumHomoModes();
        const unsigned int nComp = in.GetNumComponents();

        // Unpadded entries per plane, summed over the blocks of `in`, and the
        // total across all planes.
        size_t planePts = 0;
        for (auto &inblock : in.GetBlocks())
        {
            planePts += inblock.GetNumElements() * inblock.GetNumData();
        }
        const size_t nTotal = planePts * nhomo;

        Array<OneD, double> gathered(nTotal);
        Array<OneD, double> coef(nTotal, 0.0);
        Array<OneD, double> waveCoef(nTotal, 0.0);
        Array<OneD, double> d2zFlat(nTotal);

        for (unsigned int n = 0; n < nComp; ++n)
        {
            // Gather: collect unpadded per-plane data from all blocks into
            // gathered[], laid out as [plane0_all, plane1_all, ...]. Block b
            // contributes realPts = GetNumElements * GetNumData entries per
            // plane, placed at gathered[p*planePts + spatialOffset].
            size_t spatialOffset = 0;
            for (auto &inblock : in.GetBlocks())
            {
                const size_t realPts =
                    inblock.GetNumElements() * inblock.GetNumData();
                const size_t compStride = inblock.CompSize();
                const TData *phiPtr =
                    inblock
                        .template GetPtr<NektarSpaces::HostSpace, ReadOnly>() +
                    n * compStride * nhomo;

                for (unsigned int p = 0; p < nhomo; ++p)
                {
                    const TData *src = phiPtr + p * compStride;
                    std::copy(src, src + realPts,
                              gathered.data() + p * planePts + spatialOffset);
                }

                spatialOffset += realPts;
            }

            // Forward FFT: physical -> spectral (full field, XY-major layout).
            m_homoExpList->Homogeneous1DTrans(static_cast<int>(nTotal),
                                              gathered, coef, true);

            // Wavenumber multiply by (beta k)^2. The multiplier is real, so
            // a mode stays in its own plane rather than moving to its
            // partner; the constant and the basis' zero slot both report
            // k = 0 and so fall out on their own.
            std::memset(waveCoef.data(), 0, nTotal * sizeof(double));
            for (unsigned int i = 0; i < nhomo; ++i)
            {
                const double betaK =
                    m_beta *
                    m_homoExpList->m_transposition->GetK(static_cast<int>(i));
                const double factor = betaK * betaK;
                const size_t offset = i * planePts;
                for (size_t j = 0; j < planePts; ++j)
                {
                    waveCoef[offset + j] = factor * coef[offset + j];
                }
            }

            // Backward FFT: spectral -> physical -d2phi/dz2.
            m_homoExpList->Homogeneous1DTrans(static_cast<int>(nTotal),
                                              waveCoef, d2zFlat, false);

            // Scatter: overwrite the output component. Homogeneous1DTrans
            // works in NekDouble, which is double whatever TData is, so the
            // copy narrows and cannot go through the Math kernels, whose
            // operands share one type.
            spatialOffset = 0;
            for (auto &outblock : out.GetBlocks())
            {
                const size_t realPts =
                    outblock.GetNumElements() * outblock.GetNumData();
                const size_t compStride = outblock.CompSize();
                TData *d2zPtr =
                    outblock
                        .template GetPtr<NektarSpaces::HostSpace, WriteOnly>() +
                    n * compStride * nhomo;

                for (unsigned int p = 0; p < nhomo; ++p)
                {
                    const double *src =
                        d2zFlat.data() + p * planePts + spatialOffset;
                    TData *dst = d2zPtr + p * compStride;
                    for (size_t j = 0; j < realPts; ++j)
                    {
                        dst[j] = static_cast<TData>(src[j]);
                    }
                }

                spatialOffset += realPts;
            }
        }
    }

private:
    std::shared_ptr<MultiRegions::ExpListHomogeneous1D> m_homoExpList;
    double m_beta = 0.0;
};

} // namespace Nektar::Operators::detail
