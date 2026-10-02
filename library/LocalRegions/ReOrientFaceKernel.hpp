///////////////////////////////////////////////////////////////////////////////
//
// File: ReOrientFaceKernel.hpp
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
// Description: Reorientation of face data between the element and trace frames
//
///////////////////////////////////////////////////////////////////////////////
#pragma once

#include <LibUtilities/BasicUtils/ErrorUtil.hpp>
#include <LibUtilities/BasicUtils/NekInline.hpp>
#include <StdRegions/StdRegions.hpp>

#ifndef NEK_HOSTDEVICE_INLINE
#define NEK_HOSTDEVICE_INLINE inline
#endif

namespace Nektar::LocalRegions
{

/**
 * @brief Move face trace data between an element's local face ordering and
 * the global trace ordering, applying the trace orientation.
 *
 * @p Forwards selects the direction the data travels:
 *
 *   - `true`  - @p in is the element's local face, @p out the global trace.
 *               This is the gather, from the element's local face layout
 *               to the shared trace layout.
 *   - `false` - @p in is the global trace, @p out the element's local face.
 *               This is the scatter, the inverse of the gather.
 *
 * The two are inverses of each other for the same @p orient, so a gather
 * followed by a scatter returns the original data.
 *
 * Note that in both directions the routine is written as a *gather over its
 * own output*: it walks the output points and fetches the corresponding
 * input point for each. The index arithmetic below therefore runs opposite
 * to the direction the data travels - with @p Forwards it computes, for a
 * global point, which local point feeds it. That is worth keeping in mind
 * when reading the cases, and it is the sense in which
 * TraceFluxOpImpl::ComposeOrient() combines two orientations.
 *
 * @p orient describes how the local face sits on the global trace. For a
 * face there are eight possibilities, the symmetries of a square: each of
 * the two directions may be reversed, and the pair may be transposed. The
 * enum orders them so that, counting from eDir1FwdDir1_Dir2FwdDir2, bit 0
 * reverses trace direction 1, bit 1 reverses direction 0 and bit 2
 * transposes the two.
 *
 * @param orient   - Orientation of the local face relative to the global
 *                   trace.
 * @param nq0      - Points in local face direction 0.
 * @param nq1      - Points in local face direction 1. A transposed
 *                   orientation swaps the two on the global trace, so the
 *                   caller must pass the *local* extents; see
 *                   ReorientedFaceExtents().
 * @param in       - Source, of `nq0 * nq1` points.
 * @param out      - Destination, of `nq0 * nq1` points. Must not alias
 *                   @p in; only checked in a debug build, as the assert
 *                   has to compile out for the device.
 * @param Forwards - Direction of travel, as above.
 *
 * @tparam APPEND       Accumulate into @p out rather than overwrite it.
 * @tparam NEGATE_INPUT Negate on the way through. Used to give the backward
 *                      side of an interior trace the opposite normal
 *                      without a second pass over the data.
 */
// Called from host code and from inside device kernels; its checks are
// host-device asserts, live on the host and nothing in device code, where an
// assert message cannot be built.
template <bool APPEND, bool NEGATE_INPUT, typename TData>
NEK_HOSTDEVICE_INLINE static void ReOrientFaceKernel(
    const StdRegions::Orientation orient, const unsigned nq0,
    const unsigned nq1, const TData *in, TData *out, bool Forwards)
{
    NEK_HOSTDEVICE_ASSERTL1(
        in != out, "This routine cannot use the same input and output");

    // Input sign change if required.
    TData sign = (NEGATE_INPUT) ? -1.0 : 1.0;

    switch (orient)
    {
        case StdRegions::eDir1FwdDir1_Dir2FwdDir2: // used for Tris & Quads
        {
            // straight copy
            if constexpr (APPEND)
            {
                for (unsigned i = 0; i < nq0 * nq1; ++i)
                {
                    out[i] += sign * in[i];
                }
            }
            else
            {
                for (unsigned i = 0; i < nq0 * nq1; ++i)
                {
                    out[i] = sign * in[i];
                }
            }
        }
        break;
        case StdRegions::eDir1BwdDir1_Dir2FwdDir2: // used for Tris & Quads
        {
            // Direction A negative and B positive
            TData store;
            unsigned nq0half = (nq0 + 1u) / 2;
            if constexpr (APPEND)
            {
                for (unsigned j = 0; j < nq1; j++)
                {
                    for (unsigned i = 0; i < nq0half; ++i)
                    {
                        store = sign * in[j * nq0 + i];
                        out[j * nq0 + i] += sign * in[j * nq0 + nq0 - 1u - i];
                        out[j * nq0 + nq0 - 1u - i] += store;
                    }
                }
            }
            else
            {
                for (unsigned j = 0; j < nq1; j++)
                {
                    for (unsigned i = 0; i < nq0half; ++i)
                    {
                        store            = sign * in[j * nq0 + i];
                        out[j * nq0 + i] = sign * in[j * nq0 + nq0 - 1u - i];
                        out[j * nq0 + nq0 - 1u - i] = store;
                    }
                }
            }
        }
        break;
        case StdRegions::eDir1FwdDir1_Dir2BwdDir2:
        {
            // Direction A positive and B negative
            if constexpr (APPEND)
            {
                for (int j = 0; j < nq1; j++)
                {
                    for (int i = 0; i < nq0; ++i)
                    {
                        out[j * nq0 + i] += sign * in[nq0 * (nq1 - 1u - j) + i];
                    }
                }
            }
            else
            {
                for (int j = 0; j < nq1; j++)
                {
                    for (int i = 0; i < nq0; ++i)
                    {
                        out[j * nq0 + i] = sign * in[nq0 * (nq1 - 1u - j) + i];
                    }
                }
            }
        }
        break;
        case StdRegions::eDir1BwdDir1_Dir2BwdDir2:
        {
            // Direction A positive and B negative
            if constexpr (APPEND)
            {
                for (int j = 0; j < nq1; j++)
                {
                    for (int i = 0; i < nq0; ++i)
                    {
                        out[j * nq0 + i] +=
                            sign * in[nq0 * nq1 - 1u - j * nq0 - i];
                    }
                }
            }
            else
            {
                for (int j = 0; j < nq1; j++)
                {
                    for (int i = 0; i < nq0; ++i)
                    {
                        out[j * nq0 + i] =
                            sign * in[nq0 * nq1 - 1u - j * nq0 - i];
                    }
                }
            }
        }
        break;
        case StdRegions::eDir1FwdDir2_Dir2FwdDir1:
        {
            // Transposed, Direction A and B positive
            if constexpr (APPEND)
            {
                if (Forwards)
                {
                    for (int j = 0; j < nq1; ++j)
                    {
                        for (int i = 0; i < nq0; ++i)
                        {
                            out[i * nq1 + j] += sign * in[i + j * nq0];
                        }
                    }
                }
                else // inverse case - different if nq0 != nq1
                {
                    for (int j = 0; j < nq1; ++j)
                    {
                        for (int i = 0; i < nq0; ++i)
                        {
                            out[j * nq0 + i] += sign * in[i * nq1 + j];
                        }
                    }
                }
            }
            else
            {
                if (Forwards)
                {
                    for (int j = 0; j < nq1; ++j)
                    {
                        for (int i = 0; i < nq0; ++i)
                        {
                            out[i * nq1 + j] = sign * in[i + j * nq0];
                        }
                    }
                }
                else // inverse case - different if nq0 != nq1
                {
                    for (int j = 0; j < nq1; ++j)
                    {
                        for (int i = 0; i < nq0; ++i)
                        {
                            out[j * nq0 + i] = sign * in[i * nq1 + j];
                        }
                    }
                }
            }
        }
        break;
        case StdRegions::eDir1FwdDir2_Dir2BwdDir1:
        {
            if constexpr (APPEND)
            {
                if (Forwards)
                {
                    // forward case (element to trace) or (loc trace to trace)
                    // Transposed, Direction A positive and B negative
                    for (int i = 0; i < nq0; ++i)
                    {
                        for (int j = 0; j < nq1; ++j)
                        {

                            out[i * nq1 + j] +=
                                sign * in[i + nq0 * (nq1 - 1u) - j * nq0];
                        }
                    }
                }
                else
                {
                    // inverse case (trace to element)
                    // Transposed, Direction A positive and B negative
                    for (int j = 0; j < nq1; ++j)
                    {
                        for (int i = 0; i < nq0; ++i)
                        {
                            out[j * nq0 + i] +=
                                sign * in[nq1 - 1u - j + i * nq1];
                        }
                    }
                }
            }
            else
            {
                if (Forwards)
                {
                    // forward case (element to trace) or (loc trace to trace)
                    // Transposed, Direction A positive and B negative
                    for (int i = 0; i < nq0; ++i)
                    {
                        for (int j = 0; j < nq1; ++j)
                        {

                            out[i * nq1 + j] =
                                sign * in[i + nq0 * (nq1 - 1u) - j * nq0];
                        }
                    }
                }
                else
                {
                    // inverse case (trace to element)
                    // Transposed, Direction A positive and B negative
                    for (int j = 0; j < nq1; ++j)
                    {
                        for (int i = 0; i < nq0; ++i)
                        {
                            out[j * nq0 + i] =
                                sign * in[nq1 - 1u - j + i * nq1];
                        }
                    }
                }
            }
        }
        break;
        case StdRegions::eDir1BwdDir2_Dir2FwdDir1:
        {
            // Transposed, Direction A negative and B positive
            if constexpr (APPEND)
            {
                if (Forwards)
                {
                    for (int i = 0; i < nq0; ++i)
                    {
                        for (int j = 0; j < nq1; ++j)
                        {
                            out[i * nq1 + j] +=
                                sign * in[nq0 - 1u - i + j * nq0];
                        }
                    }
                }
                else
                {
                    for (int j = 0; j < nq1; ++j)
                    {
                        for (int i = 0; i < nq0; ++i)
                        {
                            out[i + j * nq0] +=
                                sign * in[nq1 * (nq0 - 1u) - i * nq1 + j];
                        }
                    }
                }
            }
            else
            {
                if (Forwards)
                {
                    for (int i = 0; i < nq0; ++i)
                    {
                        for (int j = 0; j < nq1; ++j)
                        {
                            out[i * nq1 + j] =
                                sign * in[nq0 - 1u - i + j * nq0];
                        }
                    }
                }
                else
                {
                    for (int j = 0; j < nq1; ++j)
                    {
                        for (int i = 0; i < nq0; ++i)
                        {
                            out[i + j * nq0] =
                                sign * in[nq1 * (nq0 - 1u) - i * nq1 + j];
                        }
                    }
                }
            }
        }
        break;
        case StdRegions::eDir1BwdDir2_Dir2BwdDir1:
        {
            // Transposed, Direction A and B negative
            if constexpr (APPEND)
            {
                if (Forwards)
                {
                    for (int i = 0; i < nq0; ++i)
                    {
                        for (int j = 0; j < nq1; ++j)
                        {
                            out[i * nq1 + j] +=
                                sign * in[nq0 * nq1 - 1u - i - j * nq0];
                        }
                    }
                }
                else // inverse case - different if nq0 != nq1
                {
                    for (int j = 0; j < nq1; ++j)
                    {
                        for (int i = 0; i < nq0; ++i)
                        {
                            out[i + j * nq0] +=
                                sign * in[nq0 * nq1 - 1u - j - i * nq1];
                        }
                    }
                }
            }
            else
            {
                if (Forwards)
                {
                    for (int i = 0; i < nq0; ++i)
                    {
                        for (int j = 0; j < nq1; ++j)
                        {
                            out[i * nq1 + j] =
                                sign * in[nq0 * nq1 - 1u - i - j * nq0];
                        }
                    }
                }
                else // inverse case - different if nq0 != nq1
                {
                    for (int j = 0; j < nq1; ++j)
                    {
                        for (int i = 0; i < nq0; ++i)
                        {
                            out[i + j * nq0] =
                                sign * in[nq0 * nq1 - 1u - j - i * nq1];
                        }
                    }
                }
            }
        }
        break;
        default:
            NEK_HOSTDEVICE_ASSERTL1(false, "Unknown orientation");
            break;
    }
}

} // namespace Nektar::LocalRegions
