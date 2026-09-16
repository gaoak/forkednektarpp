///////////////////////////////////////////////////////////////////////////////
//
// File: PhysDerivZOpHost.h
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
// Description: Host (FFTW) backend for the PhysDerivOp z-derivative. Always
// built. Operates directly on LibUtilities::BlockAccessor data pointers; the
// FFT pipeline (transposition + FFTW + wavenumber multiply) is driven via the
// cached ExpListHomogeneous1D.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include "Operators/ElmtOps/PhysDeriv/PhysDerivZOp.h"

#include <MultiRegions/ExpListHomogeneous1D.h>

namespace Nektar::Operators
{

/// \brief Host FFTW backend for the homogeneous z-derivative.
///
/// For each input component, Launch() gathers all blocks' unpadded per-plane
/// data into a contiguous array of size nTotal, calls Homogeneous1DTrans
/// once on the full field, does the wavenumber multiply, inverse transforms,
/// then scatters the z-derivative results back to the blocks.
///
/// Gathering across all blocks before calling Homogeneous1DTrans is required
/// because that function's transposition object is built for the full-field
/// point count (nTotal = nhomo × planePts) and cannot operate on per-block
/// subsets.
///
/// \tparam TData Floating-point element type.
template <typename TData>
class PhysDerivZOpHost : public PhysDerivZOpBase<TData>
{
public:
    /// \param expansionList  Expansion list; dynamic-cast to
    ///                       ExpListHomogeneous1D to obtain the
    ///                       FFT/transposition objects. May be null or a
    ///                       non-homo list, and Launch() then returns without
    ///                       computing dz.
    explicit PhysDerivZOpHost(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        m_homoExpList =
            std::dynamic_pointer_cast<MultiRegions::ExpListHomogeneous1D>(
                expansionList);
    }

protected:
    void v_Init(TData beta) override
    {
        m_beta = beta;
    }

    void v_Launch(LibUtilities::Field<TData, FieldState::Phys> &in,
                  LibUtilities::Field<TData, FieldState::Phys> &out) override
    {
        const auto nhomo = in.GetNumHomoModes();

        if (nhomo <= 1)
        {
            return;
        }

        // Only ExpListHomogeneous1D carries the FFT and transposition
        // objects; without them there is no z-FFT to apply.
        if (!m_homoExpList)
        {
            return;
        }

        const int iNhomo = static_cast<int>(nhomo);

        // Unpadded points per plane, summed over the blocks of `in`, and the
        // total across all planes.
        int planePts = 0;
        for (auto &inblock : in.GetBlocks())
        {
            planePts += static_cast<int>(inblock.GetNumElements() *
                                         inblock.GetNumData());
        }
        const int nTotal = planePts * iNhomo;

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
            int spatialOffset = 0;
            for (auto &inblock : in.GetBlocks())
            {
                const int realPts = static_cast<int>(inblock.GetNumElements() *
                                                     inblock.GetNumData());
                const int compStride = static_cast<int>(inblock.CompSize());
                const TData *phiPtr =
                    inblock
                        .template GetPtr<NektarSpaces::HostSpace, ReadOnly>() +
                    static_cast<std::ptrdiff_t>(n) * compStride * iNhomo;

                for (int p = 0; p < iNhomo; ++p)
                {
                    const TData *src = phiPtr + p * compStride;
                    auto *dst = gathered.data() + p * planePts + spatialOffset;
                    if constexpr (std::is_same_v<TData, double>)
                    {
                        std::copy(src, src + realPts, dst);
                    }
                    else
                    {
                        for (int i = 0; i < realPts; ++i)
                        {
                            dst[i] = static_cast<double>(src[i]);
                        }
                    }
                }

                spatialOffset += realPts;
            }

            // Forward FFT: physical -> spectral (full field, XY-major layout).
            m_homoExpList->Homogeneous1DTrans(nTotal, gathered, coef, true);

            // Wavenumber multiply.
            std::memset(waveCoef.data(), 0, nTotal * sizeof(double));
            double sign = -1.0;
            for (int i = 0; i < iNhomo; ++i)
            {
                const double betaI =
                    -sign * m_beta * m_homoExpList->m_transposition->GetK(i);
                for (int j = 0; j < planePts; ++j)
                {
                    waveCoef[(i - static_cast<int>(sign)) * planePts + j] =
                        betaI * coef[i * planePts + j];
                }
                sign = -sign;
            }

            // Backward FFT: spectral -> physical dz.
            m_homoExpList->Homogeneous1DTrans(nTotal, waveCoef, dzFlat, false);

            // Scatter: write z-derivative back to per-block output z-slots.
            // Output component for z of input n is at index (n*3 + 2).
            // Block b, plane p offset = ((n*3+2)*nhomo + p) * compStride.
            spatialOffset = 0;
            for (auto &outblock : out.GetBlocks())
            {
                const int realPts = static_cast<int>(outblock.GetNumElements() *
                                                     outblock.GetNumData());
                const int compStride = static_cast<int>(outblock.CompSize());
                TData *dzPtr =
                    outblock
                        .template GetPtr<NektarSpaces::HostSpace, WriteOnly>() +
                    static_cast<std::ptrdiff_t>(n * 3 + 2) * compStride *
                        iNhomo;

                for (int p = 0; p < iNhomo; ++p)
                {
                    TData *dst = dzPtr + p * compStride;
                    const auto *src =
                        dzFlat.data() + p * planePts + spatialOffset;

                    if constexpr (std::is_same_v<TData, double>)
                    {
                        std::copy(src, src + realPts, dst);
                    }
                    else
                    {
                        for (int i = 0; i < realPts; ++i)
                        {
                            dst[i] = static_cast<TData>(src[i]);
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

} // namespace Nektar::Operators
