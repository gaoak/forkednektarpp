///////////////////////////////////////////////////////////////////////////////
//
// File: PhysDerivZOpSerialAVX.hpp
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
// Description: Host (FFTW) backend for the PhysDerivOp z-derivative.
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
template <typename ExecSpace, typename TData>
class PhysDerivZOpImpl<
    ExecSpace, TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Serial> ||
                     std::is_same_v<ExecSpace, NektarSpaces::AVX>>>
{
public:
    /// \param expansionList  Cast to ExpListHomogeneous1D for the FFT and
    ///                       transposition objects.
    PhysDerivZOpImpl(const MultiRegions::ExpListSharedPtr &expansionList)
    {
        m_homoExpList =
            std::dynamic_pointer_cast<MultiRegions::ExpListHomogeneous1D>(
                expansionList);

        ASSERTL0(m_homoExpList,
                 "The homogeneous z-derivative needs an ExpListHomogeneous1D");
    }

    // Non-copyable and non-movable.
    PhysDerivZOpImpl(const PhysDerivZOpImpl &)            = delete;
    PhysDerivZOpImpl &operator=(const PhysDerivZOpImpl &) = delete;
    PhysDerivZOpImpl(PhysDerivZOpImpl &&)                 = delete;
    PhysDerivZOpImpl &operator=(PhysDerivZOpImpl &&)      = delete;

    void Init(TData beta)
    {
        m_beta = beta;
    }

    void Launch(LibUtilities::Field<TData, FieldState::Phys> &in,
                LibUtilities::Field<TData, FieldState::Phys> &out)
    {
        const unsigned int nhomo = in.GetNumHomoModes();

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

        // z-derivative per input component.
        for (unsigned int n = 0; n < in.GetNumComponents(); ++n)
        {
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

            // Wavenumber multiply. The real and imaginary halves of a Fourier
            // mode occupy adjacent planes, so d/dz writes plane i to its
            // partner (i ^ 1) and negates the odd one.
            std::memset(waveCoef.data(), 0, nTotal * sizeof(double));
            for (unsigned int i = 0; i < nhomo; ++i)
            {
                const double betaI =
                    (i % 2 == 0 ? 1.0 : -1.0) * m_beta *
                    m_homoExpList->m_transposition->GetK(static_cast<int>(i));
                const size_t srcOffset = i * planePts;
                const size_t dstOffset = (i ^ 1u) * planePts;
                for (size_t j = 0; j < planePts; ++j)
                {
                    waveCoef[dstOffset + j] = betaI * coef[srcOffset + j];
                }
            }

            // Backward FFT: spectral -> physical dz.
            m_homoExpList->Homogeneous1DTrans(static_cast<int>(nTotal),
                                              waveCoef, dzFlat, false);

            // Scatter: write z-derivative back to per-block output z-slots.
            // Output component for z of input n is at index (n*3 + 2).
            // Block b, plane p offset = ((n*3+2)*nhomo + p) * compStride.
            spatialOffset = 0;
            for (auto &outblock : out.GetBlocks())
            {
                const size_t realPts =
                    outblock.GetNumElements() * outblock.GetNumData();
                const size_t compStride = outblock.CompSize();
                TData *dzPtr =
                    outblock
                        .template GetPtr<NektarSpaces::HostSpace, WriteOnly>() +
                    (n * 3 + 2) * compStride * nhomo;

                for (unsigned int p = 0; p < nhomo; ++p)
                {
                    const double *src =
                        dzFlat.data() + p * planePts + spatialOffset;
                    std::copy(src, src + realPts, dzPtr + p * compStride);
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
