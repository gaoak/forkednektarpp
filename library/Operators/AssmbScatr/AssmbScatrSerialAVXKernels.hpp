///////////////////////////////////////////////////////////////////////////////
//
// File: AssmbScatrSerialAVXKernels.hpp
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

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename TData>
NEK_FORCE_INLINE static
    typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Serial> ||
                                std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                            void>::type
    AssembleScatrKernel(const unsigned nvals, const unsigned *GSInfo,
                        const int *sign, TData *inoutptr)
{
    const unsigned *offset = GSInfo + 1;
    const unsigned *ind    = GSInfo + nvals + 2;

    for (unsigned idx = 0; idx < nvals; ++idx)
    {
        TData ass = 0;
        for (unsigned j = offset[idx]; j < offset[idx + 1]; ++j)
        {
            ass += inoutptr[ind[j]] * sign[j];
        }
        for (unsigned j = offset[idx]; j < offset[idx + 1]; ++j)
        {
            inoutptr[ind[j]] = ass * sign[j];
        }
    }
}
// When thsse are defiend and the code is run in serial we have a mapping lay
// out which is assuming with != 1 and so have access maps in a different manner
#if defined(NEKTAR_ENABLE_SIMD) || defined(NEKTAR_ENABLE_CUDA) ||              \
    defined(NEKTAR_ENABLE_HIP) || defined(NEKTAR_ENABLE_SYCL)
template <typename ExecSpace, typename TData, unsigned WIDTH>
NEK_FORCE_INLINE static
    typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Serial> ||
                                std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                            void>::type
    AssembleScatrKernel(const unsigned nvals, const unsigned *nassemble,
                        const unsigned *index, const unsigned *offset,
                        const int *sign, TData *inoutptr)
{
    for (unsigned idx = 0; idx < nvals; idx += WIDTH)
    {
        for (unsigned i = 0; i < WIDTH; ++i)
        {
            TData ass =
                inoutptr[index[offset[idx + i]]] * sign[offset[idx + i]];

            // might be able to write this in a simd loop
            // but not sure it will provide much asdditional speed.
            for (unsigned j = 1; j < nassemble[idx + i]; ++j)
            {
                ass += inoutptr[index[offset[idx + i] + j * WIDTH]] *
                       sign[offset[idx + i] + j * WIDTH];
            }

            for (unsigned j = 0; j < nassemble[idx + i]; ++j)
            {
                inoutptr[index[offset[idx + i] + j * WIDTH]] =
                    ass * sign[offset[idx + i] + j * WIDTH];
            }
        }
    }
}
#else
template <typename ExecSpace, typename TData, unsigned WIDTH>
NEK_FORCE_INLINE static
    typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Serial> ||
                                std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                            void>::type
    AssembleScatrKernel(const unsigned nvals, const unsigned *nassemble,
                        const unsigned *ind,
                        [[maybe_unused]] const unsigned *offset,
                        const int *sign, TData *inoutptr)

{
    unsigned cnt = 0;

    ASSERTL1(WIDTH == 1, "Assuming width=1 for Serial version");

    for (unsigned idx = 0; idx < nvals; ++idx)
    {
        TData ass = inoutptr[ind[cnt]] * sign[cnt];

        for (unsigned j = 1; j < nassemble[idx]; ++j)
        {
            ass += inoutptr[ind[cnt + j]] * sign[cnt + j];
        }

        for (unsigned j = 0; j < nassemble[idx]; ++j)
        {
            inoutptr[ind[cnt + j]] = ass * sign[cnt + j];
        }
        cnt += nassemble[idx];
    }
}
#endif

template <typename ExecSpace, typename TData>
NEK_FORCE_INLINE static
    typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Serial> ||
                                std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                            void>::type
    AssembleScatrBndKernel(const unsigned nvals, const unsigned *GSInfo,
                           const int *sign, TData *inoutptr, TData *bndptr)
{
    const unsigned *offset = GSInfo + 1;
    const unsigned *ind    = GSInfo + nvals + 2;

    unsigned cnt = 0;
    for (unsigned idx = 0; idx < nvals; ++idx)
    {
        TData ass  = 0;
        auto start = offset[idx];
        auto nidx  = ind[start];
        auto nbnd  = ind[start + 1];

        start += 2;
        // assemble values
        for (unsigned j = 0; j < nidx; ++j)
        {
            ass += inoutptr[ind[start + j]] * sign[cnt + j];
        }
        // copy assembled values back to local values
        // !!!! Beleive we may only need to do first point here. Rest will be
        // copied back in next kernel
        for (unsigned j = 0; j < nidx; ++j)
        {
            inoutptr[ind[start + j]] = ass * sign[cnt + j];
        }
        // put assembled values into boudnary array
        start += nidx;
        for (unsigned j = 0; j < nbnd; ++j)
        {
            bndptr[ind[start + j]] = ass;
        }
        cnt += nidx;
    }
}

template <typename ExecSpace, typename TData>
NEK_FORCE_INLINE static
    typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Serial> ||
                                std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                            void>::type
    AssembleFromBndKernel(const unsigned nvals, const unsigned *GSInfo,
                          const int *sign, const TData *bndptr, TData *inoutptr)
{
    const unsigned *offset = GSInfo + 1;
    const unsigned *ind    = GSInfo + nvals + 2;
    unsigned cnt           = 0;

    for (unsigned idx = 0; idx < nvals; ++idx)
    {
        TData ass   = 0;
        auto start  = offset[idx];
        auto nidx   = ind[start];
        auto nbnd   = ind[start + 1];
        auto norder = ind[start + 2 + nidx + nbnd];
        start += 2;

        auto startbnd = start + nidx;

        ASSERTL1(norder <= nbnd, "norder is greater than nbnd");
        // assemble bndptr components wtih local ids
        unsigned j;
        for (j = 0; j < norder; ++j)
        {
            ass += bndptr[ind[startbnd + j]];
        }

        // add in local point in rank ordered assembly
        ass += inoutptr[ind[start]] * sign[cnt];

        // assemble rest of points from where we left off
        for (; j < nbnd; ++j)
        {
            ass += bndptr[ind[startbnd + j]];
        }

        // copy rank ordered assembled values back to local values
        for (j = 0; j < nidx; ++j)
        {
            inoutptr[ind[start + j]] = ass * sign[cnt + j];
        }
        cnt += nidx;
    }
}

} // namespace Nektar::Operators::detail
