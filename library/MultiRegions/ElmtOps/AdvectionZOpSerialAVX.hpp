///////////////////////////////////////////////////////////////////////////////
//
// File: AdvectionZOpSerialAVX.hpp
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
// Description: Host (FFTW) backend for the AdvectionOp z-derivative.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <LibUtilities/BasicUtils/ErrorUtil.hpp>
#include <LibUtilities/BasicUtils/Utils/UtilsKernels.hpp>

namespace Nektar::MultiRegions::detail
{

/// \brief Host FFTW backend for the homogeneous part of the advection term.
///
/// Launch() gathers each component's unpadded per-plane data from all blocks
/// into one contiguous array, applies the forward transform, wavenumber
/// multiply and inverse transform over the full field, then adds
/// scale * w * dphi/dz to the matching output component. The gather is needed
/// because Homogeneous1DTrans' transposition object is built for the
/// full-field point count (nhomo x planePts) and cannot operate on per-block
/// subsets.
///
/// The xy backend leaves scale * (u dphi/dx + v dphi/dy) in the output,
/// having honoured its own append flag, so this pass always accumulates.
template <typename ExecSpace, typename TData>
class AdvectionZOpImpl<
    ExecSpace, TData,
    std::enable_if_t<std::is_same_v<ExecSpace, NektarSpaces::Serial> ||
                     std::is_same_v<ExecSpace, NektarSpaces::AVX>>>
{
public:
    /// \param expansionList  Cast to ExpListHomogeneous1D for the FFT and
    ///                       transposition objects.
    AdvectionZOpImpl(const MultiRegions::ExpListSharedPtr &expansionList)
    {
        m_homoExpList =
            std::dynamic_pointer_cast<MultiRegions::ExpListHomogeneous1D>(
                expansionList);

        ASSERTL0(m_homoExpList,
                 "The homogeneous advection needs an ExpListHomogeneous1D");

        m_beta = 2.0 * M_PI / m_homoExpList->GetHomoLen();
    }

    // Non-copyable and non-movable.
    AdvectionZOpImpl(const AdvectionZOpImpl &)            = delete;
    AdvectionZOpImpl &operator=(const AdvectionZOpImpl &) = delete;
    AdvectionZOpImpl(AdvectionZOpImpl &&)                 = delete;
    AdvectionZOpImpl &operator=(AdvectionZOpImpl &&)      = delete;

    void SetScale(const TData &scale)
    {
        m_scale = scale;
    }

    void SetAdvVel(LibUtilities::Field<TData, FieldState::Phys> &advVel)
    {
        m_advVel = &advVel;
    }

    void Launch(LibUtilities::Field<TData, FieldState::Phys> &in,
                LibUtilities::Field<TData, FieldState::Phys> &out)
    {
        const unsigned int nhomo = in.GetNumHomoModes();
        const unsigned int nComp = in.GetNumComponents();

        // The advection velocity along the homogeneous direction is the last
        // component; the xy backend has already consumed the others.
        const unsigned int wComp = m_advVel->GetNumComponents() - 1;

        // The xy backend reshapes the advection velocity into its own
        // interleave format and leaves it there, so it is realigned with the
        // input before being read.
        for (unsigned int blk = 0; blk < in.GetBlocks().size(); ++blk)
        {
            auto &velblock = m_advVel->GetBlocks()[blk];

            const auto interleaveWidth =
                in.GetBlocks()[blk].GetInterleaveWidth();

            if (velblock.GetInterleaveWidth() != interleaveWidth)
            {
                auto velRWPtr =
                    velblock
                        .template GetPtr<NektarSpaces::HostSpace, ReadWrite>();
                LibUtilities::ReshapeStorage<ExecSpace>(
                    interleaveWidth, velblock.GetInterleaveWidth(),
                    velblock.GetNumElementsWithPadding() *
                        velblock.GetNumComponents() *
                        velblock.GetNumHomoModes(),
                    velblock.GetNumData(), velRWPtr);
                velblock.template SetInterleaveWidth<TData>(interleaveWidth);
            }
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

        // z-derivative per input component.
        for (unsigned int n = 0; n < nComp; ++n)
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

            // Backward FFT: spectral -> physical dphi/dz.
            m_homoExpList->Homogeneous1DTrans(static_cast<int>(nTotal),
                                              waveCoef, dzFlat, false);

            // Scatter: add scale * w * dphi/dz to output component n, which
            // already holds the xy part of the advection term.
            // Homogeneous1DTrans works in NekDouble, which is double whatever
            // TData is, so the sum narrows as it accumulates and cannot go
            // through the Math kernels, whose operands share one type.
            spatialOffset = 0;
            for (unsigned int blk = 0; blk < out.GetBlocks().size(); ++blk)
            {
                auto &outblock = out.GetBlocks()[blk];
                auto &velblock = m_advVel->GetBlocks()[blk];

                const size_t realPts =
                    outblock.GetNumElements() * outblock.GetNumData();
                const size_t compStride = outblock.CompSize();
                const size_t velStride  = velblock.CompSize();

                TData *advPtr =
                    outblock
                        .template GetPtr<NektarSpaces::HostSpace, ReadWrite>() +
                    n * compStride * nhomo;
                const TData *wPtr =
                    velblock
                        .template GetPtr<NektarSpaces::HostSpace, ReadOnly>() +
                    wComp * velStride * nhomo;

                for (unsigned int p = 0; p < nhomo; ++p)
                {
                    const double *src =
                        dzFlat.data() + p * planePts + spatialOffset;
                    TData *dst     = advPtr + p * compStride;
                    const TData *w = wPtr + p * velStride;
                    for (size_t j = 0; j < realPts; ++j)
                    {
                        dst[j] += m_scale * w[j] * src[j];
                    }
                }

                spatialOffset += realPts;
            }
        }
    }

private:
    std::shared_ptr<MultiRegions::ExpListHomogeneous1D> m_homoExpList;
    LibUtilities::Field<TData, FieldState::Phys> *m_advVel = nullptr;
    double m_beta                                          = 0.0;
    TData m_scale                                          = 1.0;
};

} // namespace Nektar::MultiRegions::detail
