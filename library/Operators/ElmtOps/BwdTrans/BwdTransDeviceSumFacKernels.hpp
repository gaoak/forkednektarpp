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
template <typename Implementation>
inline unsigned int BwdTransSharedMemorySize(const unsigned int nq0,
                                             const unsigned int nm0)
{
    if constexpr (std::is_same_v<Implementation, Operators::SumFacQP>)
    {
        return nm0 + nm0 * nq0;
    }
    else
    {
        return 0;
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation>
inline unsigned int BwdTransSharedMemorySize(const unsigned int nq0,
                                             const unsigned int nq1,
                                             const unsigned int nm0,
                                             const unsigned int nm1)
{
    const unsigned int nmTot =
        LibUtilities::GetNumberOfCoefficients(SHAPE_TYPE, nm0, nm1);

    if constexpr (std::is_same_v<Implementation, Operators::SumFacQP>)
    {
        if constexpr (SHAPE_TYPE == LibUtilities::Quad)
        {
            return nq0 * nm0 + nq1 * nm1 + nmTot + nq0 * nm1;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tri)
        {
            return nm0 * nq0 + nmTot * nq1 + nmTot + nm0 * nq1;
        }
    }
    else
    {
        return 0;
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, typename Implementation>
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

    if constexpr (std::is_same_v<Implementation, Operators::SumFacQP>)
    {
        if constexpr (SHAPE_TYPE == LibUtilities::Hex)
        {
            return nq0 * nm0 + nq1 * nm1 + nq2 * nm2 + nq0 * nm0 + nq1 * nm1 +
                   nq2 * nm2 + nmTot + (nq0 * nm1 * nm2) + (nq0 * nq1 * nm2);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tet)
        {
            return nm0 * nq0 + nm01 * nq1 + nmode2 * nq2 + nmTot +
                   (nm01 * nq2) + (nm0 * nq1 * nq2);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Prism)
        {
            return nm0 * nq0 + nm1 * nq1 + nm12 * nq2 + nmTot +
                   (nm0 * nm1 * nq2) + (nm0 * nq1 * nq2);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
        {
            return nm0 * nq0 + nm1 * nq1 + nmode2 * nq2 + nmTot +
                   (nm0 * nm1 * nq2) + (nm0 * nq1 * nq2);
        }
    }
    else
    {
        return 0;
    }
}

} // namespace Nektar::Operators::detail

#include "Operators/ElmtOps/BwdTrans/BwdTransCUDASumFacKernels.cuh"
#include "Operators/ElmtOps/BwdTrans/BwdTransKokkosSumFacKernels.hpp"
#include "Operators/ElmtOps/BwdTrans/BwdTransSYCLSumFacKernels.hpp"
