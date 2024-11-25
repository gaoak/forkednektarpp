///////////////////////////////////////////////////////////////////////////////
//
// File: IProductWRTBaseDeviceSumFacKernels.hpp
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

#include <LibUtilities/BasicUtils/ShapeType.hpp>

#include "Operators/Common/Spaces.hpp"

namespace Nektar::Operators::detail
{

// Helper function
template <bool SHMEM, bool MULTILEVEL>
inline unsigned int IProductWRTBaseSharedMemorySize(const unsigned int nq0,
                                                    const unsigned int nm0)
{
    if constexpr (MULTILEVEL)
    {
        return SHMEM * (nm0 * nq0 + nq0) + nq0;
    }
    else
    {
        return SHMEM * (nm0 * nq0 + nq0);
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, bool SHMEM, bool MULTILEVEL>
inline unsigned int IProductWRTBaseSharedMemorySize(const unsigned int nq0,
                                                    const unsigned int nq1,
                                                    const unsigned int nm0,
                                                    const unsigned int nm1)
{
    unsigned int nshared = 0;

    if constexpr (SHMEM)
    {
        if constexpr (SHAPE_TYPE == LibUtilities::Quad)
        {
            nshared += nm0 * nq0 + nm1 * nq1 + nq0 + nq1;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tri)
        {
            const unsigned int nmTot =
                LibUtilities::StdTriData::getNumberOfCoefficients(nm0, nm1);
            nshared += nm0 * nq0 + nmTot * nq1 + nq0 + nq1;
        }
    }

    if constexpr (MULTILEVEL)
    {
        if constexpr (SHAPE_TYPE == LibUtilities::Quad)
        {
            nshared += nq0 * nq1 + nm0 * nq1;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tri)
        {
            nshared += nq0 * nq1 + nm0 * nq1 + 1u;
        }
    }

    return nshared;
}

template <LibUtilities::ShapeType SHAPE_TYPE, bool SHMEM, bool MULTILEVEL>
inline unsigned int IProductWRTBaseSharedMemorySize(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2)
{
    unsigned int nshared = 0;

    if constexpr (SHMEM)
    {
        if constexpr (SHAPE_TYPE == LibUtilities::Hex)
        {
            nshared += nm0 * nq0 + nm1 * nq1 + nm2 * nq2 + nq0 + nq1 + nq2;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tet)
        {
            const unsigned int nmTot = LibUtilities::GetNumberOfCoefficients(
                SHAPE_TYPE, nm0, nm1, nm2);
            const unsigned int nmode2 =
                nmTot + nm0 * (nm2 - nm1 + 1u) * (nm2 - nm1) / 2u;
            const unsigned int nm01 = (2u * nm1 - nm0 + 1u) * nm0 / 2u;
            nshared += nm0 * nq0 + nm01 * nq1 + nmode2 * nq2 + nq0 + nq1 + nq2;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Prism)
        {
            const unsigned int nm02 = (2u * nm2 - nm0 + 1u) * nm0 / 2u;
            nshared += nm0 * nq0 + nm1 * nq1 + nm02 * nq2 + nq0 + nq1 + nq2;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
        {
            const unsigned int nmTot = LibUtilities::GetNumberOfCoefficients(
                SHAPE_TYPE, nm0, nm1, nm2);
            const unsigned int nmode2 =
                nmTot + nm0 * (nm2 - nm1 + 1u) * (nm2 - nm1) / 2u;
            nshared += nm0 * nq0 + nm1 * nq1 + nmode2 * nq2 + nq0 + nq1 + nq2;
        }
    }

    if constexpr (MULTILEVEL)
    {
        if constexpr (SHAPE_TYPE == LibUtilities::Hex)
        {
            nshared += nq0 * nq1 * nq2 + nm0 * nq1 * nq2 + nm0 * nm1 * nq2;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tet)
        {
            const unsigned int nm01 = (2u * nm1 - nm0 + 1u) * nm0 / 2u;
            nshared += nq0 * nq1 * nq2 + nm0 * nq1 * nq2 + nm01 * nq2 + nm2;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Prism)
        {
            nshared +=
                nq0 * nq1 * nq2 + nm0 * nq1 * nq2 + nm0 * nm1 * nq2 + nm1;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
        {
            nshared += nq0 * nq1 * nq2 + nm0 * nq1 * nq2 + nm0 * nm1 * nq2 + 1u;
        }
    }

    return nshared;
}

} // namespace Nektar::Operators::detail

#include "Operators/ElmtOps/IProductWRTBase/IProductWRTBaseCUDASumFacKernels.cuh"
#include "Operators/ElmtOps/IProductWRTBase/IProductWRTBaseKokkosSumFacKernels.hpp"
#include "Operators/ElmtOps/IProductWRTBase/IProductWRTBaseSYCLSumFacKernels.hpp"
