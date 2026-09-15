///////////////////////////////////////////////////////////////////////////////
//
// File: GetFwdBwdTracePhysBlockOpSerialAVXKernels.hpp
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

#pragma once
#include <LibUtilities/BasicUtils/NekInline.hpp>

#include "MultiRegions/AssemblyMap/LocTraceToTraceMap.h"

namespace Nektar::Operators::detail
{
template <bool FwdOnly, typename TData>
NEK_FORCE_INLINE static void GetFwdBwdTracePhys1DKernel(
    const size_t el, const size_t nelmt, const size_t elmtTot,
    const unsigned int nc, const unsigned int traceId,
    const unsigned int tracePts, const unsigned int nqTraceOffset,
    const unsigned int nTraces, const unsigned int *locTracePhysToElmtMapsPtr,
    const unsigned int *orientationMapsPtr,
    const size_t *orientationMapsOffsetPtr,
    const size_t *locToTracePhysOffsetPtr,
    const bool *isLocTraceLeftAdjacentPtr, const TData *phyptr, TData *fwdptr,
    [[maybe_unused]] TData *bwdptr)
{

    if (el < nelmt)
    {
        const size_t key  = traceId * elmtTot + el;
        const auto isLeft = isLocTraceLeftAdjacentPtr[key];

        if constexpr (FwdOnly)
        {
            if (!isLeft)
            {
                return;
            }
        }

        size_t offset = locToTracePhysOffsetPtr[nc * nTraces * elmtTot + key];

        auto *Fwdptr  = fwdptr + offset;
        TData *dstPtr = nullptr;

        if constexpr (FwdOnly)
        {
            dstPtr = Fwdptr;
        }
        else
        {
            dstPtr = isLeft ? Fwdptr : (bwdptr + offset);
        }

        const unsigned int orientBase =
            orientationMapsOffsetPtr[el * nTraces + traceId];
        const size_t elTraceBase = el * tracePts + nqTraceOffset;

        unsigned int orientMapIdx = orientationMapsPtr[orientBase];
        unsigned int traceMapIdx  = locTracePhysToElmtMapsPtr[elTraceBase];

        TData tphys = *(phyptr + traceMapIdx);

        dstPtr[orientMapIdx] = tphys;
    }
}

template <bool FwdOnly, typename TData>
NEK_FORCE_INLINE static void GetFwdBwdTracePhys2DKernel(
    const size_t el, const size_t nelmt, const size_t elmtTot,
    const unsigned int nc, const unsigned int traceId,
    const unsigned int tracePts, const unsigned int nqTraceOffset,
    const unsigned int nTraces, const unsigned int *locTracePhysToElmtMapsPtr,
    const unsigned int *orientationMapsPtr,
    const size_t *orientationMapsOffsetPtr,
    const size_t *locToTracePhysOffsetPtr,
    const bool *isLocTraceLeftAdjacentPtr,
    const unsigned int *interpTraceIndexPtr,
    const unsigned int *interpPointsPtr, const unsigned int *interpTypesPtr,
    const unsigned int *quadRangePtr,
    const MultiRegions::InterpLocTraceToTrace *interpTracePtr,
    const TData *interpTraceI0Ptr, const unsigned int *interpTraceI0OffsetPtr,
    const TData *interpEndPtI0Ptr, const unsigned int *interpEndPtI0OffsetPtr,
    const TData *phyptr, TData *fwdptr, [[maybe_unused]] TData *bwdptr)
{

    if (el < nelmt)
    {
        const size_t key  = traceId * elmtTot + el;
        const auto isLeft = isLocTraceLeftAdjacentPtr[key];

        if constexpr (FwdOnly)
        {
            if (!isLeft)
            {
                return;
            }
        }

        const unsigned int dir = isLeft ? 0u : 1u;
        unsigned int typeId    = interpTraceIndexPtr[key];

        size_t offset = locToTracePhysOffsetPtr[nc * nTraces * elmtTot + key];

        auto *Fwdptr  = fwdptr + offset;
        TData *dstPtr = nullptr;

        if constexpr (FwdOnly)
        {
            dstPtr = Fwdptr;
        }
        else
        {
            dstPtr = isLeft ? Fwdptr : (bwdptr + offset);
        }

        unsigned int nTypes = interpTypesPtr[0];
        unsigned int fnp = interpPointsPtr[dir * nTypes * 2 + typeId * 4 + 0];
        unsigned int tnp = interpPointsPtr[dir * nTypes * 2 + typeId * 4 + 2];

        const unsigned int orientBase =
            orientationMapsOffsetPtr[el * nTraces + traceId];
        const size_t elTraceBase = el * tracePts + nqTraceOffset;

        switch (interpTracePtr[dir * nTypes / 2 + typeId])
        {
            case MultiRegions::eNoInterp:
            {
                unsigned int fbegin =
                    quadRangePtr[dir * nTypes * 4 + typeId * 8 + 0 * 2 + 0];
                unsigned int fend =
                    quadRangePtr[dir * nTypes * 4 + typeId * 8 + 0 * 2 + 1];
                unsigned int tbegin =
                    quadRangePtr[dir * nTypes * 4 + typeId * 8 + 2 * 2 + 0];

                for (unsigned int q = fbegin, p = tbegin; q < fend; ++q, ++p)
                {
                    unsigned int orientMapIdx =
                        orientationMapsPtr[orientBase + p];
                    unsigned int traceMapIdx =
                        locTracePhysToElmtMapsPtr[elTraceBase + q];

                    TData tphys = *(phyptr + traceMapIdx);

                    dstPtr[orientMapIdx] = tphys;
                }
            }
            break;

            case MultiRegions::eInterpDir0:
            {
                auto *I0 = interpTraceI0Ptr;
                auto I0Offset =
                    interpTraceI0OffsetPtr[dir * nTypes / 2 + typeId];
                for (unsigned int i = 0; i < tnp; ++i)
                {
                    TData tphys = 0.0;
                    unsigned int orientMapIdx =
                        orientationMapsPtr[orientBase + i];

                    for (unsigned int p = 0; p < fnp; ++p)
                    {
                        unsigned int traceMapIdx =
                            locTracePhysToElmtMapsPtr[elTraceBase + p];

                        unsigned int cnt_ip = i + p * tnp;
                        TData val           = *(phyptr + traceMapIdx);
                        TData weight        = I0[I0Offset + cnt_ip];
                        tphys += val * weight;
                    }

                    dstPtr[orientMapIdx] = tphys;
                }
            }
            break;

            case MultiRegions::eInterpEndPtDir0:
            {
                auto *I0 = interpEndPtI0Ptr;
                auto I0Offset =
                    interpEndPtI0OffsetPtr[dir * nTypes / 2 + typeId];
                for (unsigned int i = 0; i < fnp; ++i)
                {
                    unsigned int traceMapIdx =
                        locTracePhysToElmtMapsPtr[elTraceBase + i];
                    unsigned int orientMapIdx =
                        orientationMapsPtr[orientBase + i];
                    unsigned int orientMapFnpIdx =
                        orientationMapsPtr[orientBase + fnp];

                    TData val    = *(phyptr + traceMapIdx);
                    TData weight = I0[I0Offset + i];

                    dstPtr[orientMapIdx] = val;
                    dstPtr[orientMapFnpIdx] += val * weight;
                }
            }
            break;
            default:
            {
            }
            break;
        }
    }
}

template <bool FwdOnly, typename TData>
NEK_FORCE_INLINE static void GetFwdBwdTracePhys3DKernel(
    const size_t el, const size_t nelmt, const size_t elmtTot,
    const unsigned int nc, const unsigned int traceId,
    const unsigned int tracePts, const unsigned int nqTraceOffset,
    const unsigned int nTraces, const unsigned int *locTracePhysToElmtMapsPtr,
    const unsigned int *orientationMapsPtr,
    const size_t *orientationMapsOffsetPtr,
    const size_t *locToTracePhysOffsetPtr,
    const bool *isLocTraceLeftAdjacentPtr,
    const unsigned int *interpTraceIndexPtr,
    const unsigned int *interpPointsPtr, const unsigned int *interpTypesPtr,
    const unsigned int *quadRangePtr,
    const MultiRegions::InterpLocTraceToTrace *interpTracePtr,
    const TData *interpTraceI0Ptr, const unsigned int *interpTraceI0OffsetPtr,
    const TData *interpTraceI1Ptr, const unsigned int *interpTraceI1OffsetPtr,
    const TData *interpEndPtI0Ptr, const unsigned int *interpEndPtI0OffsetPtr,
    const TData *interpEndPtI1Ptr, const unsigned int *interpEndPtI1OffsetPtr,
    TData *wspptr, const TData *phyptr, TData *fwdptr,
    [[maybe_unused]] TData *bwdptr)
{
    if (el < nelmt)
    {
        const size_t key  = traceId * elmtTot + el;
        const auto isLeft = isLocTraceLeftAdjacentPtr[key];

        if constexpr (FwdOnly)
        {
            if (!isLeft)
            {
                return;
            }
        }

        const unsigned int dir = isLeft ? 0u : 1u;
        unsigned int typeId    = interpTraceIndexPtr[key];

        size_t offset = locToTracePhysOffsetPtr[nc * nTraces * elmtTot + key];

        auto *Fwdptr  = fwdptr + offset;
        TData *dstPtr = nullptr;
        if constexpr (FwdOnly)
        {
            dstPtr = Fwdptr;
        }
        else
        {
            dstPtr = isLeft ? Fwdptr : (bwdptr + offset);
        }

        unsigned int nTypes = interpTypesPtr[0];
        unsigned int fnp0 = interpPointsPtr[dir * nTypes * 2 + typeId * 4 + 0];
        unsigned int fnp1 = interpPointsPtr[dir * nTypes * 2 + typeId * 4 + 1];
        unsigned int tnp0 = interpPointsPtr[dir * nTypes * 2 + typeId * 4 + 2];
        unsigned int tnp1 = interpPointsPtr[dir * nTypes * 2 + typeId * 4 + 3];

        const unsigned int orientBase =
            orientationMapsOffsetPtr[el * nTraces + traceId];
        const size_t elTraceBase = el * tracePts + nqTraceOffset;

        switch (interpTracePtr[dir * nTypes / 2 + typeId])
        {
            case MultiRegions::eNoInterp: // Just copy
            {

                unsigned int fbegin0 =
                    quadRangePtr[dir * nTypes * 4 + typeId * 8 + 0 * 2 + 0];
                unsigned int fend0 =
                    quadRangePtr[dir * nTypes * 4 + typeId * 8 + 0 * 2 + 1];
                unsigned int fbegin1 =
                    quadRangePtr[dir * nTypes * 4 + typeId * 8 + 1 * 2 + 0];
                unsigned int fend1 =
                    quadRangePtr[dir * nTypes * 4 + typeId * 8 + 1 * 2 + 1];
                unsigned int tbegin0 =
                    quadRangePtr[dir * nTypes * 4 + typeId * 8 + 2 * 2 + 0];
                unsigned int tbegin1 =
                    quadRangePtr[dir * nTypes * 4 + typeId * 8 + 3 * 2 + 0];

                for (unsigned int k = fbegin1, l = tbegin1; k < fend1; ++k, ++l)
                {
                    for (unsigned int p = fbegin0, q = tbegin0; p < fend0;
                         ++p, ++q)
                    {
                        auto tid = l * tnp0 + q;
                        auto fid = k * fnp0 + p;

                        unsigned int orientMapIdx =
                            orientationMapsPtr[orientBase + tid];
                        unsigned int traceMapIdx =
                            locTracePhysToElmtMapsPtr[elTraceBase + fid];

                        TData tphys = *(phyptr + traceMapIdx);

                        dstPtr[orientMapIdx] = tphys;
                    }
                }
            }
            break;
            case MultiRegions::eInterpDir0:
            {
                auto *I0 = interpTraceI0Ptr;
                auto I0Offset =
                    interpTraceI0OffsetPtr[dir * nTypes / 2 + typeId];
                for (unsigned int j = 0; j < tnp1; ++j)
                { // Loop over rows of A and C
                    for (unsigned int i = 0; i < tnp0; ++i)
                    { // Loop over columns of B and C
                        auto tid = j * tnp0 + i;
                        unsigned int orientMapIdx =
                            orientationMapsPtr[orientBase + tid];
                        TData tphys = 0.0;
                        for (unsigned int k = 0; k < fnp0; ++k)
                        { // Loop over columns of A and rows of B
                            auto fid = j * fnp0 + k; // unit stride
                            auto kid = k * tnp0 + i; // non-unit stride
                            unsigned int traceMapIdx =
                                locTracePhysToElmtMapsPtr[elTraceBase + fid];
                            TData val    = *(phyptr + traceMapIdx);
                            TData weight = I0[I0Offset + kid];
                            tphys += val * weight;
                        }
                        dstPtr[orientMapIdx] = tphys;
                    }
                }
            }
            break;
            case MultiRegions::eInterpEndPtDir0:
            {
                auto *I0 = interpEndPtI0Ptr;
                auto I0Offset =
                    interpEndPtI0OffsetPtr[dir * nTypes / 2 + typeId];

                // for j=0 to nfaces*tnp1-1; (tnp1 = fnp1)
                //    for i=0 to fnp0-1; (fnp0=tnp0-1)
                //       tmp[i*tnp0+k] = locfaces[i*fnp0+k]
                for (unsigned int j = 0; j < tnp1; ++j)
                {
                    for (unsigned int i = 0; i < fnp0; ++i)
                    {
                        auto fid = j * fnp0 + i;
                        auto tid = j * tnp0 + i;
                        unsigned int traceMapIdx =
                            locTracePhysToElmtMapsPtr[elTraceBase + fid];
                        unsigned int orientMapIdx =
                            orientationMapsPtr[orientBase + tid];

                        TData tphys = *(phyptr + traceMapIdx);

                        dstPtr[orientMapIdx] = tphys;

                        tid          = j * tnp0 + fnp0;
                        orientMapIdx = orientationMapsPtr[orientBase + tid];
                        TData weight = I0[I0Offset + i];

                        dstPtr[orientMapIdx] += tphys * weight;
                    }
                }
            }
            break;
            case MultiRegions::eInterpDir1:
            {
                auto *I1 = interpTraceI1Ptr;
                auto I1Offset =
                    interpTraceI1OffsetPtr[dir * nTypes / 2 + typeId];
                for (unsigned int j = 0; j < tnp1; ++j)
                { // Loop over rows of A and C
                    for (unsigned int i = 0; i < tnp0; ++i)
                    { // Loop over columns of B and C
                        auto tid = j * tnp0 + i;
                        unsigned int orientMapIdx =
                            orientationMapsPtr[orientBase + tid];
                        TData tphys = 0.0;
                        for (unsigned int k = 0; k < fnp1; ++k)
                        { // Loop over columns of A and rows of B
                            auto fid = k * fnp0 + i; // non-unit stride
                            auto kid = k * tnp1 + j; // non-unit stride

                            unsigned int traceMapIdx =
                                locTracePhysToElmtMapsPtr[elTraceBase + fid];
                            TData val    = *(phyptr + traceMapIdx);
                            TData weight = I1[I1Offset + kid];
                            tphys += val * weight;
                        }
                        dstPtr[orientMapIdx] = tphys;
                    }
                }
            }
            break;
            case MultiRegions::eInterpEndPtDir1:
            {
                auto *I1 = interpEndPtI1Ptr;
                auto I1Offset =
                    interpEndPtI1OffsetPtr[dir * nTypes / 2 + typeId];
                for (unsigned int j = 0; j < fnp1; ++j)
                {
                    for (unsigned int i = 0; i < tnp0; ++i)
                    {
                        auto tid = j * tnp0 + i;
                        auto fid = j * fnp0 + i;
                        unsigned int traceMapIdx =
                            locTracePhysToElmtMapsPtr[elTraceBase + fid];
                        unsigned int orientMapIdx =
                            orientationMapsPtr[orientBase + tid];

                        TData tphys = *(phyptr + traceMapIdx);

                        dstPtr[orientMapIdx] = tphys;

                        tid          = fnp1 * fnp0 + i;
                        orientMapIdx = orientationMapsPtr[orientBase + tid];
                        TData weight = I1[I1Offset + j];
                        dstPtr[orientMapIdx] += tphys * weight;
                    }
                }
            }
            break;
            case MultiRegions::eInterpBothDirs:
            {
                auto *I0 = interpTraceI0Ptr;
                auto I0Offset =
                    interpTraceI0OffsetPtr[dir * nTypes / 2 + typeId];
                auto *I1 = interpTraceI1Ptr;
                auto I1Offset =
                    interpTraceI1OffsetPtr[dir * nTypes / 2 + typeId];
                for (unsigned int q = 0; q < fnp1; ++q)
                {
                    for (unsigned int p = 0; p < 1; ++p)
                    {
                        // load B(p,q):
                        unsigned int cnt_pq = p + fnp0 * q;

                        unsigned int traceMapIdx =
                            locTracePhysToElmtMapsPtr[elTraceBase + cnt_pq];

                        TData tphys = *(phyptr + traceMapIdx);
                        for (unsigned int i = 0; i < tnp0; ++i)
                        {
                            unsigned int cnt_ip = p * tnp0 + i;
                            unsigned int cnt_iq = q * tnp0 + i;
                            // load A(i,p) and fill C(i,q)
                            TData weight   = I0[I0Offset + cnt_ip];
                            wspptr[cnt_iq] = weight * tphys;
                        }
                    }
                    for (unsigned int p = 1; p < fnp0; ++p)
                    {
                        // load B(p,q):
                        unsigned int cnt_pq = p + fnp0 * q;

                        unsigned int traceMapIdx =
                            locTracePhysToElmtMapsPtr[elTraceBase + cnt_pq];

                        TData tphys = *(phyptr + traceMapIdx);
                        for (unsigned int i = 0; i < tnp0; ++i)
                        {
                            unsigned int cnt_ip = p * tnp0 + i;
                            unsigned int cnt_iq = q * tnp0 + i;
                            // load A(i,p) and add to C(i,q)
                            TData weight = I0[I0Offset + cnt_ip];
                            wspptr[cnt_iq] += weight * tphys;
                        }
                    }
                }
                // C(i,j) = A(i,q) * B(q,j);  prefer fewer access on C
                // (non-strided)
                for (unsigned int j = 0; j < tnp1; ++j)
                {
                    for (unsigned int i = 0; i < tnp0; ++i)
                    {
                        unsigned int cnt_ij = i + j * tnp0;
                        unsigned int orientMapIdx =
                            orientationMapsPtr[orientBase + cnt_ij];
                        TData prod3 = 0.0;
                        for (unsigned int q = 0; q < fnp1; ++q)
                        {
                            unsigned int cnt_iq = i + q * tnp0;
                            unsigned int cnt_jq = j + q * tnp1;
                            TData weight        = I1[I1Offset + cnt_jq];
                            prod3 += wspptr[cnt_iq] * weight;
                        }

                        dstPtr[orientMapIdx] = prod3;
                    }
                }
            }
            break;
            case MultiRegions::eInterpEndPtDir0InterpDir1:
            {
                auto *I0 = interpEndPtI0Ptr;
                auto I0Offset =
                    interpEndPtI0OffsetPtr[dir * nTypes / 2 + typeId];
                auto *I1 = interpTraceI1Ptr;
                auto I1Offset =
                    interpTraceI1OffsetPtr[dir * nTypes / 2 + typeId];
                for (unsigned int j = 0; j < tnp1; ++j)
                {
                    for (unsigned int i = 0; i < fnp0; ++i)
                    {
                        auto tid = j * tnp0 + i;
                        unsigned int orientMapIdx =
                            orientationMapsPtr[orientBase + tid];

                        TData tphys = 0.0;
                        for (unsigned int k = 0; k < fnp1; ++k)
                        {
                            auto fid = k * fnp0 + i; // non-unit stride
                            auto kid = k * tnp1 + j; // non-unit stride
                            unsigned int traceMapIdx =
                                locTracePhysToElmtMapsPtr[elTraceBase + fid];
                            TData val    = *(phyptr + traceMapIdx);
                            TData weight = I1[I1Offset + kid];
                            tphys += weight * val;
                        }

                        dstPtr[orientMapIdx] = tphys;

                        tid          = j * tnp0 + fnp0;
                        orientMapIdx = orientationMapsPtr[orientBase + tid];
                        TData weight = I0[I0Offset + i];

                        dstPtr[orientMapIdx] += tphys * weight;
                    }
                }
            }
            break;
            default:
            {
            }
            break;
        }
    }
}
} // namespace Nektar::Operators::detail
