///////////////////////////////////////////////////////////////////////////////
//
// File: BwdTransDeviceSumFacKernels.hpp
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
inline unsigned int BwdTransSharedMemorySize(const unsigned int nq0,
                                             const unsigned int nm0)
{
    if constexpr (MULTILEVEL)
    {
        return nm0 + SHMEM * nm0 * nq0;
    }
    else
    {
        return SHMEM * nm0 * nq0;
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, bool SHMEM, bool MULTILEVEL>
inline unsigned int BwdTransSharedMemorySize(const unsigned int nq0,
                                             const unsigned int nq1,
                                             const unsigned int nm0,
                                             const unsigned int nm1)
{
    const unsigned int nmTot =
        LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1);

    unsigned int nshared = 0;

    if constexpr (SHMEM)
    {
        if constexpr (SHAPE_TYPE == LibUtilities::Quad)
        {
            nshared += nq0 * nm0 + nq1 * nm1;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tri)
        {
            nshared += nm0 * nq0 + nmTot * nq1;
        }
    }

    if constexpr (MULTILEVEL)
    {
        if constexpr (SHAPE_TYPE == LibUtilities::Quad)
        {
            nshared += nmTot + nq0 * nm1;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tri)
        {
            nshared += nmTot + nm0 * nq1;
        }
    }

    return nshared;
}

template <LibUtilities::ShapeType SHAPE_TYPE, bool SHMEM, bool MULTILEVEL>
inline unsigned int BwdTransSharedMemorySize(
    const unsigned int nq0, const unsigned int nq1, const unsigned int nq2,
    const unsigned int nm0, const unsigned int nm1, const unsigned int nm2)
{
    const unsigned int nmTot =
        LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1, nm2);
    const unsigned int nm01 = (2u * nm1 - nm0 + 1u) * nm0 / 2u;
    const unsigned int nm12 = (2u * nm2 - nm1 + 1u) * nm1 / 2u;
    const unsigned int nmode2 =
        nmTot + nm0 * (nm2 - nm1 + 1u) * (nm2 - nm1) / 2u;

    unsigned int nshared = 0;

    if constexpr (SHMEM)
    {
        if constexpr (SHAPE_TYPE == LibUtilities::Hex)
        {
            nshared += nq0 * nm0 + nq1 * nm1 + nq2 * nm2;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tet)
        {
            nshared += nm0 * nq0 + nm01 * nq1 + nmode2 * nq2;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Prism)
        {
            nshared += nm0 * nq0 + nm1 * nq1 + nm12 * nq2;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
        {
            nshared += nm0 * nq0 + nm1 * nq1 + nmode2 * nq2;
        }
    }

    if constexpr (MULTILEVEL)
    {
        if constexpr (SHAPE_TYPE == LibUtilities::Hex)
        {
            nshared += nq0 * nm0 + nq1 * nm1 + nq2 * nm2 + nmTot +
                       (nq0 * nm1 * nm2) + (nq0 * nq1 * nm2);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tet)
        {
            nshared += nmTot + (nm01 * nq2) + (nm0 * nq1 * nq2);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Prism)
        {
            nshared += nmTot + (nm0 * nm1 * nq2) + (nm0 * nq1 * nq2);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
        {
            nshared += nmTot + (nm0 * nm1 * nq2) + (nm0 * nq1 * nq2);
        }
    }

    return nshared;
}

} // namespace Nektar::Operators::detail

#include "Operators/ElmtOps/BwdTrans/BwdTransCUDASumFacKernels.cuh"
#include "Operators/ElmtOps/BwdTrans/BwdTransKokkosSumFacKernels.hpp"
#include "Operators/ElmtOps/BwdTrans/BwdTransSYCLSumFacKernels.hpp"
