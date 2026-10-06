///////////////////////////////////////////////////////////////////////////////
//
// File: PhysTraceExtractKernels.hpp
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
// Description: Trace-interpolation kernels of the PhysTraceExtract
// operator, shared by the Serial, AVX and Device paths. One definition
// serves the scalar and the SIMD data types; the accumulate is chosen on
// the data type, so the file carries no execution space in its name.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <type_traits>

#include "LibUtilities/BasicUtils/ErrorUtil.hpp"
#include "LibUtilities/BasicUtils/NekInline.hpp"

namespace Nektar::MultiRegions::detail
{

/**
 * @brief Stage two in two dimensions: interpolate the extracted edge
 * values from the volume points of the in-edge direction onto that
 * edge's trace quadrature points.
 *
 * For every edge and every trace point @em i,
 * `out[i] = sum_p in[p] * tbasis[p * nqto + i]`. The edges lie
 * consecutively in both buffers, @p nqfrom entries each on input and
 * @p nqto each on output. Under @p isCollocated the trace points are the
 * volume points, the table is the identity and the call is a copy of
 * `nqto * nedg` values; the debug assertion states the precondition that
 * makes that legal.
 *
 * @note @p nedg is a count here, unlike the extraction kernels, where it
 * is one past the position of the last edge. The glue kernels pass
 * `nedg - edg`.
 *
 * @tparam TData    Element data type: a scalar, or a SIMD vector with one
 *                  element per lane. The accumulate follows it.
 * @tparam TScalar  Scalar type of the interpolation table, which is never
 *                  vectorised.
 *
 * @param   nedg            Number of edges in the buffers.
 * @param   nqfrom          Volume quadrature points of the in-edge
 *                          direction, per edge on input.
 * @param   nqto            Trace quadrature points, per edge on output.
 * @param   tbasis          eInterp table \f$h_p(\xi^{tr}_i)\f$ of the
 *                          in-edge direction, indexed
 *                          `tbasis[p * nqto + i]`.
 * @param   in              Extracted values from stage one.
 * @param   out             Values at the trace quadrature points.
 * @param   isCollocated    Trace points coincide with the volume points.
 */
template <typename TData, typename TScalar>
NEK_HOSTDEVICE_INLINE static void PhysInterpEdgeKernel(
    const unsigned nedg, const unsigned nqfrom, const unsigned nqto,
    const TScalar *tbasis, const TData *in, TData *out,
    const bool isCollocated = false)
{
    if (isCollocated)
    {
        NEK_HOSTDEVICE_ASSERTL1(nqto == nqfrom, "Basis is not collocated");
        for (unsigned i = 0; i < nqto * nedg; ++i)
        {
            out[i] = in[i];
        }
    }
    else
    {
        unsigned offset = 0;
        for (unsigned e = 0, cnt_ei = 0; e < nedg; ++e)
        {
            for (unsigned i = 0; i < nqto; ++i, ++cnt_ei)
            {
                TData tmp = in[offset] * tbasis[i];
                for (unsigned p = 1; p < nqfrom; ++p)
                {
                    if constexpr (std::is_floating_point_v<TData>)
                    {
                        tmp += in[offset + p] * tbasis[p * nqto + i];
                    }
                    else
                    {
                        tmp.fma(in[offset + p], tbasis[p * nqto + i]);
                    }
                }
                out[cnt_ei] = tmp;
            }
            offset += nqfrom; // offset to 2nd edge if required
        }
    }
}

/**
 * @brief Stage two in three dimensions: interpolate the extracted face
 * planes from the volume points of the two tangential directions onto
 * the face's trace quadrature points.
 *
 * The face contraction is a tensor product, so it runs as two
 * one-dimensional passes, direction 0 first. Which passes are needed
 * depends on the two collocation flags, giving three arms:
 * - @p isCollocated1: direction 1 is already the trace rule, so only the
 *   tangential-0 pass runs, straight from @p in to @p out;
 * - @p isCollocated0: only the tangential-1 pass runs, again straight to
 *   @p out;
 * - neither: the tangential-0 pass writes @p wsp and the tangential-1
 *   pass reads it.
 *
 * Both directions collocated does not reach here: the glue kernels take
 * their direct branch and do not call this at all.
 *
 * Faces lie consecutively in every buffer, `nqfrom0 * nqfrom1` entries
 * each on input and `nqto0 * nqto1` each on output, direction 0 varying
 * fastest throughout.
 *
 * @tparam TData    Element data type: a scalar, or a SIMD vector with one
 *                  element per lane. The accumulate follows it.
 * @tparam TScalar  Scalar type of the interpolation tables.
 *
 * @param   nfac            Number of faces in the buffers; a count here,
 *                          unlike the extraction kernels, where it is
 *                          one past the last face position. The glue
 *                          kernels pass `nfac - fac`.
 * @param   nqfrom0,nqfrom1 Volume quadrature points of the two
 *                          tangential directions, per face on input.
 * @param   nqto0,nqto1     Trace quadrature points, per face on output.
 * @param   tbasis0,tbasis1 eInterp tables of the two tangential directions,
 *                          indexed `tbasis0[p * nqto0 + i]` and
 *                          `tbasis1[q * nqto1 + j]`.
 * @param   wsp             Scratch for the tangential-0 pass; used only when
 *                          both passes run.
 * @param   in              Extracted planes from stage one.
 * @param   out             Values at the face's trace quadrature points.
 * @param   isCollocated0   Tangential-0 trace points coincide with the volume
 *                          points of that direction.
 * @param   isCollocated1   Tangential-1 trace points coincide with the volume
 *                          points of that direction.
 */
template <typename TData, typename TScalar>
NEK_HOSTDEVICE_INLINE static void PhysInterpFaceKernel(
    const unsigned nfac, const unsigned nqfrom0, const unsigned nqfrom1,
    const unsigned nqto0, const unsigned nqto1, const TScalar *tbasis0,
    const TScalar *tbasis1, TData *wsp, const TData *in, TData *out,
    const bool isCollocated0, const bool isCollocated1)
{
    if (isCollocated1) // interpolate dir 0 direction directly to output
    {
        NEK_HOSTDEVICE_ASSERTL1(nqfrom1 == nqto1, "Basis is not collocated");
        for (unsigned f = 0, cnt_fjp = 0, cnt_fji = 0; f < nfac; ++f)
        {
            for (unsigned j = 0; j < nqfrom1; ++j)
            {
                for (unsigned i = 0; i < nqto0; ++i, ++cnt_fji)
                {
                    TData tmp = in[cnt_fjp] * tbasis0[i];
                    for (unsigned p = 1; p < nqfrom0; ++p)
                    {
                        if constexpr (std::is_floating_point_v<TData>)
                        {
                            tmp += in[cnt_fjp + p] * tbasis0[p * nqto0 + i];
                        }
                        else
                        {
                            tmp.fma(in[cnt_fjp + p], tbasis0[p * nqto0 + i]);
                        }
                    }
                    out[cnt_fji] = tmp;
                }
                cnt_fjp += nqfrom0;
            }
        }
    }
    else if (isCollocated0) // interplate in dir 1
    {
        NEK_HOSTDEVICE_ASSERTL1(nqfrom0 == nqto0, "Basis is not collocated");
        for (unsigned f = 0; f < nfac; ++f)
        {
            for (unsigned i = 0; i < nqto0; ++i)
            {
                for (unsigned j = 0; j < nqto1; ++j)
                {
                    TData tmp = in[i] * tbasis1[j];
                    for (unsigned q = 1; q < nqfrom1; ++q)
                    {
                        if constexpr (std::is_floating_point_v<TData>)
                        {
                            tmp += in[i + q * nqto0] * tbasis1[q * nqto1 + j];
                        }
                        else
                        {
                            tmp.fma(in[i + q * nqto0], tbasis1[q * nqto1 + j]);
                        }
                    }

                    out[i + j * nqto0] = tmp;
                }
            }
            in += nqto0 * nqfrom1;
            out += nqto0 * nqto1;
        }
    }
    else //  interpolation in dir 0 and dir 1
    {
        // dir 0 interpolation to wsp
        for (unsigned f = 0, cnt_fjp = 0, cnt_fji = 0; f < nfac; ++f)
        {
            for (unsigned j = 0; j < nqfrom1; ++j)
            {
                for (unsigned i = 0; i < nqto0; ++i, ++cnt_fji)
                {
                    TData tmp = in[cnt_fjp] * tbasis0[i];
                    for (unsigned p = 1; p < nqfrom0; ++p)
                    {
                        if constexpr (std::is_floating_point_v<TData>)
                        {
                            tmp += in[cnt_fjp + p] * tbasis0[p * nqto0 + i];
                        }
                        else
                        {
                            tmp.fma(in[cnt_fjp + p], tbasis0[p * nqto0 + i]);
                        }
                    }
                    wsp[cnt_fji] = tmp;
                }
                cnt_fjp += nqfrom0;
            }
        }

        // dir 1 interpolation to out
        for (unsigned f = 0; f < nfac; ++f)
        {
            for (unsigned i = 0; i < nqto0; ++i)
            {
                for (unsigned j = 0; j < nqto1; ++j)
                {
                    TData tmp = wsp[i] * tbasis1[j];
                    for (unsigned q = 1; q < nqfrom1; ++q)
                    {
                        if constexpr (std::is_floating_point_v<TData>)
                        {
                            tmp += wsp[i + q * nqto0] * tbasis1[q * nqto1 + j];
                        }
                        else
                        {
                            tmp.fma(wsp[i + q * nqto0], tbasis1[q * nqto1 + j]);
                        }
                    }

                    out[i + j * nqto0] = tmp;
                }
            }
            wsp += nqto0 * nqfrom1;
            out += nqto0 * nqto1;
        }
    }
}

} // namespace Nektar::MultiRegions::detail
