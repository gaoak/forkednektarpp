///////////////////////////////////////////////////////////////////////////////
//
// File: PhysDerivDeviceSumFacKernels.hpp
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
template <LibUtilities::ShapeType SHAPE_TYPE, bool SHMEM, bool MULTILEVEL>
inline unsigned int PhysDerivSharedMemorySize(const unsigned int nq0,
                                              const unsigned int nq1)
{
    if constexpr (MULTILEVEL)
    {
        return SHMEM * (nq0 * nq0 + nq1 * nq1) + nq0 * nq1;
    }
    else
    {
        if constexpr (SHAPE_TYPE == LibUtilities::Quad)
        {
            return SHMEM * (nq0 * nq0 + nq1 * nq1);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tri)
        {
            return SHMEM * (nq0 * nq0 + nq1 * nq1) + nq0 * nq1;
        }
    }
}

template <LibUtilities::ShapeType SHAPE_TYPE, bool SHMEM, bool MULTILEVEL>
inline unsigned int PhysDerivSharedMemorySize(const unsigned int nq0,
                                              const unsigned int nq1,
                                              const unsigned int nq2)
{
    if constexpr (MULTILEVEL)
    {
        return SHMEM * (nq0 * nq0 + nq1 * nq1 + nq2 * nq2) + nq0 * nq1 * nq2;
    }
    else
    {
        if constexpr (SHAPE_TYPE == LibUtilities::Hex)
        {
            return SHMEM * (nq0 * nq0 + nq1 * nq1 + nq2 * nq2);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tet)
        {
            return (SHMEM * (nq0 * nq0 + nq1 * nq1 + nq2 * nq2) + nq0 +
                    2u * nq1 + nq2);
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Prism)
        {
            return SHMEM * (nq0 * nq0 + nq1 * nq1 + nq2 * nq2) + nq0 + nq2;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Pyr)
        {
            return SHMEM * (nq0 * nq0 + nq1 * nq1 + nq2 * nq2) + nq0 + nq1 +
                   nq2;
        }
    }
}

} // namespace Nektar::Operators::detail

#include "Operators/ElmtOps/PhysDeriv/PhysDerivCUDASumFacKernels.cuh"
#include "Operators/ElmtOps/PhysDeriv/PhysDerivKokkosSumFacKernels.hpp"
#include "Operators/ElmtOps/PhysDeriv/PhysDerivSYCLSumFacKernels.hpp"
