///////////////////////////////////////////////////////////////////////////////
//
// File: DerivZOpSerialAVX.hpp
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
// Description: Host (FFTW) backend for the homogeneous z-derivative.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <LibUtilities/BasicUtils/ErrorUtil.hpp>

namespace Nektar::Operators::detail
{

/// \brief Host FFTW backend for the homogeneous z-derivative.
///
/// Launch() gathers each component's unpadded per-plane data from all blocks
/// into one contiguous array, applies the forward transform, wavenumber
/// multiply and inverse transform over the full field, then scatters the
/// result back. The gather is needed because Homogeneous1DTrans' transposition
/// object is built for the full-field point count (nhomo × planePts) and
/// cannot operate on per-block subsets.
///
/// DerivZOpImpl.hpp says what LAYOUT, DERIVORDER and APPEND select. Only the
/// wavenumber multiply depends on DERIVORDER; the gather, the transforms and
/// the scatter are the same either way.
template <typename ExecSpace, typename TData, DerivZLayout LAYOUT,
          DerivZOrder DERIVORDER, bool APPEND>
class DerivZOpImpl<
    ExecSpace, TData, LAYOUT, DERIVORDER, APPEND,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Serial> ||
                     std::is_same_v<ExecSpace, NektarSpaces::AVX>>>
{
public:
    /// \param expansionList  Cast to ExpListHomogeneous1D for the FFT and
    ///                       transposition objects.
    DerivZOpImpl(const MultiRegions::ExpListSharedPtr &expansionList)
    {
        m_homoExpList =
            std::dynamic_pointer_cast<MultiRegions::ExpListHomogeneous1D>(
                expansionList);

        ASSERTL0(m_homoExpList,
                 "The homogeneous z-derivative needs an ExpListHomogeneous1D");

        m_beta = 2.0 * M_PI / m_homoExpList->GetHomoLen();
    }

    // Non-copyable and non-movable.
    DerivZOpImpl(const DerivZOpImpl &)            = delete;
    DerivZOpImpl &operator=(const DerivZOpImpl &) = delete;
    DerivZOpImpl(DerivZOpImpl &&)                 = delete;
    DerivZOpImpl &operator=(DerivZOpImpl &&)      = delete;

    /// Write the z-derivative of @p in to @p out, moving components as
    /// LAYOUT says. Phys and Coeff both work: Homogeneous1DTrans transforms
    /// coefficients as readily as quadrature points, as
    /// ExpListHomogeneous1D::v_FwdTrans relies on.
    template <FieldState TState>
    void Launch(LibUtilities::Field<TData, TState> &in,
                LibUtilities::Field<TData, TState> &out)
    {
        const unsigned int nhomo = in.GetNumHomoModes();
        // The scalar side of the mapping carries one component per variable
        // and is the one to loop over; only VectorZToScalar has it on the
        // output side.
        const unsigned int nComp = LAYOUT == DerivZLayout::VectorZToScalar
                                       ? out.GetNumComponents()
                                       : in.GetNumComponents();

        // The vector side gives every variable three direction slots, of
        // which this reads the z one.
        if constexpr (LAYOUT == DerivZLayout::VectorZToScalar)
        {
            ASSERTL1(in.GetNumComponents() == 3 * nComp,
                     "The homogeneous z-derivative needs three directions "
                     "per variable");
        }

        // Unpadded points per plane, summed over the blocks of `in`, and the
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
        Array<OneD, double> dzFlat(nTotal);

        // z-derivative per component of the scalar side.
        for (unsigned int n = 0; n < nComp; ++n)
        {
            unsigned int srcComp;
            unsigned int dstComp;
            if constexpr (LAYOUT == DerivZLayout::ScalarToVectorZ)
            {
                // Component n of the scalar input to slot 3n + 2 of
                // the vector output, the z entry of the gradient.
                srcComp = n;
                dstComp = n * 3 + 2;
            }
            else if constexpr (LAYOUT == DerivZLayout::VectorZToScalar)
            {
                // Component 3n + 2 of the input, the z direction of
                // variable n, to component n of the scalar output.
                srcComp = n * 3 + 2;
                dstComp = n;
            }
            else
            {
                // Component for component, both sides scalar.
                srcComp = n;
                dstComp = n;
            }

            // Gather: collect unpadded per-plane data from all blocks into
            // gathered[], laid out as [plane0_all_pts, plane1_all_pts, ...].
            // Block b contributes realPts = GetNumElements * GetNumData points
            // per plane, placed at gathered[p*planePts + spatialOffset].
            size_t spatialOffset = 0;
            for (auto &inblock : in.GetBlocks())
            {
                const size_t realPts =
                    inblock.GetNumElements() * inblock.GetNumData();
                const size_t compStride = inblock.CompSize();
                const TData *phiPtr =
                    inblock
                        .template GetPtr<NektarSpaces::HostSpace, ReadOnly>() +
                    srcComp * compStride * nhomo;

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

            // Wavenumber multiply.
            std::memset(waveCoef.data(), 0, nTotal * sizeof(double));
            for (unsigned int i = 0; i < nhomo; ++i)
            {
                const double betaK =
                    m_beta *
                    m_homoExpList->m_transposition->GetK(static_cast<int>(i));

                if constexpr (DERIVORDER == DerivZOrder::First)
                {
                    // The real and imaginary halves of a Fourier mode occupy
                    // adjacent planes, so d/dz writes plane i to its partner
                    // (i ^ 1) and negates the odd one.
                    const double factor    = (i % 2 == 0 ? 1.0 : -1.0) * betaK;
                    const size_t srcOffset = i * planePts;
                    const size_t dstOffset = (i ^ 1u) * planePts;
                    for (size_t j = 0; j < planePts; ++j)
                    {
                        waveCoef[dstOffset + j] = factor * coef[srcOffset + j];
                    }
                }
                else
                {
                    // -(beta k)^2 is real, so a mode stays in its own
                    // plane rather than moving to its partner. The constant
                    // and the basis' zero slot both report k = 0 and so fall
                    // out of the multiply on their own.
                    const double factor = -betaK * betaK;
                    const size_t offset = i * planePts;
                    for (size_t j = 0; j < planePts; ++j)
                    {
                        waveCoef[offset + j] = factor * coef[offset + j];
                    }
                }
            }

            // Backward FFT: spectral -> physical dz.
            m_homoExpList->Homogeneous1DTrans(static_cast<int>(nTotal),
                                              waveCoef, dzFlat, false);

            // Scatter: write the z-derivative back to the output
            // component, block b plane p sitting at
            // (dstComp*nhomo + p) * compStride. Homogeneous1DTrans works in
            // NekDouble, which is double whatever TData is, so the copy
            // narrows and cannot go through the Math kernels, whose operands
            // share one type.
            spatialOffset = 0;
            for (auto &outblock : out.GetBlocks())
            {
                const size_t realPts =
                    outblock.GetNumElements() * outblock.GetNumData();
                const size_t compStride = outblock.CompSize();
                TData *dzPtr =
                    outblock.template GetPtr<
                        NektarSpaces::HostSpace,
                        std::conditional_t<APPEND, ReadWrite, WriteOnly>>() +
                    dstComp * compStride * nhomo;

                for (unsigned int p = 0; p < nhomo; ++p)
                {
                    const double *src =
                        dzFlat.data() + p * planePts + spatialOffset;
                    TData *dst = dzPtr + p * compStride;
                    for (size_t j = 0; j < realPts; ++j)
                    {
                        if constexpr (APPEND)
                        {
                            dst[j] += static_cast<TData>(src[j]);
                        }
                        else
                        {
                            dst[j] = static_cast<TData>(src[j]);
                        }
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
