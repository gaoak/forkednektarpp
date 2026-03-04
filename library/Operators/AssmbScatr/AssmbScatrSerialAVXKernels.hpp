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
    typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                            void>::type
    AssembleScatrKernel(const unsigned nvals, const unsigned *nassemble,
                        const unsigned *index, const unsigned *offset,
                        const int *sign, TData *inoutptr)
{
    constexpr unsigned int vector_width =
        NektarSpaces::vector_width<NektarSpaces::AVX, TData>::value;
    for (unsigned idx = 0; idx < nvals; idx += vector_width)
    {
        for (unsigned i = 0; i < vector_width; ++i)
        {
            TData ass =
                inoutptr[index[offset[idx + i]]] * sign[offset[idx + i]];

            // might be able to write this in a simd loop
            // but not sure it will provide much asdditional speed.
            for (unsigned j = 1; j < nassemble[idx + i]; ++j)
            {
                ass += inoutptr[index[offset[idx + i] + j * vector_width]] *
                       sign[offset[idx + i] + j * vector_width];
            }

            for (unsigned j = 0; j < nassemble[idx + i]; ++j)
            {
                inoutptr[index[offset[idx + i] + j * vector_width]] =
                    ass * sign[offset[idx + i] + j * vector_width];
            }
        }
    }
}

template <typename ExecSpace, typename TData>
NEK_FORCE_INLINE static
    typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Serial>,
                            void>::type
    AssembleScatrKernel(const unsigned nvals, const unsigned *nassemble,
                        const unsigned *ind,
                        [[maybe_unused]] const unsigned *offset,
                        const int *sign, TData *inoutptr)

{
    unsigned cnt = 0;

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

template <typename ExecSpace, typename TData>
NEK_FORCE_INLINE static
    typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Serial> ||
                                std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                            void>::type
    AssembleScatrBndKernel(const unsigned nvals, const unsigned *nassemble,
                           const unsigned *nbndvals, const unsigned *index,
                           const unsigned *offset, const int *sign,
                           TData *inoutptr, TData *bndptr)
{
    for (unsigned idx = 0; idx < nvals; ++idx)
    {
        const unsigned ind  = offset[idx];
        const unsigned nidx = nassemble[idx];
        const unsigned nbnd = nbndvals[idx];

        TData ass = inoutptr[index[ind]] * sign[ind];

        for (unsigned j = 1; j < nidx; ++j)
        {
            ass += inoutptr[index[ind + j]] * sign[ind + j];
        }

        // keep one local copy for full assembly
        inoutptr[index[ind]] = ass * sign[ind];

        // put assembled values into boudnary array
        for (unsigned j = 0; j < nbnd; ++j)
        {
            bndptr[index[ind + nidx + j]] = ass;
        }
    }
}

template <typename ExecSpace, typename TData>
NEK_FORCE_INLINE static
    typename std::enable_if<std::is_same_v<ExecSpace, NektarSpaces::Serial> ||
                                std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                            void>::type
    AssembleFromBndKernel(const unsigned nvals, const unsigned *nassemble,
                          const unsigned *nbndvals, const unsigned *index,
                          const unsigned *offset, const int *sign,
                          const unsigned *norder, const TData *bndptr,
                          TData *inoutptr)
{
    for (unsigned idx = 0; idx < nvals; ++idx)
    {
        const unsigned ind  = offset[idx];
        const unsigned nidx = nassemble[idx];
        const unsigned nbnd = nbndvals[idx];
        const unsigned nord = norder[idx];

        TData ass = 0;

        // assemble bndptr components wtih local ids
        for (unsigned j = 0; j < nord; ++j)
        {
            ass += bndptr[index[ind + nidx + j]];
        }

        // add in local point in rank ordered assembly
        ass += inoutptr[index[ind]] * sign[ind];

        // assemble rest of points from where we left off
        for (unsigned j = nord; j < nbnd; ++j)
        {
            ass += bndptr[index[ind + nidx + j]];
        }

        // copy rank ordered assembled values back to local values
        for (unsigned j = 0; j < nidx; ++j)
        {
            inoutptr[index[ind + j]] = ass * sign[ind + j];
        }
    }
}

} // namespace Nektar::Operators::detail
