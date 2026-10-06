///////////////////////////////////////////////////////////////////////////////
//
// File: PhysInterp1DScaledDeviceSumFacKernels.hpp
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

/**
 * @file PhysInterp1DScaledDeviceSumFacKernels.hpp
 * @brief Device SumFac kernels of the scaled physical-space
 * interpolation: the backward transform's, reused unchanged.
 *
 * @details
 * This header defines nothing of its own. A 1D interpolation matrix has
 * exactly the layout of a 1D basis table -- entry `[p * nq + i]` is the
 * interpolant of input point p evaluated at output point i -- so
 * interpolating point values onto a rescaled grid is a backward transform
 * whose modes are the input points, and the block implementation launches
 * the BwdTransKernelLauncher overloads of BwdTransDeviceSumFacKernels.hpp
 * directly (see PhysInterp1DScaledDeviceSumFac.hpp for how the launch is
 * configured). What the header provides is that indirection alone:
 * PhysInterp1DScaledDeviceSumFac.hpp includes it, and its SumFacTOP
 * sibling, in place of the BwdTrans kernel headers.
 *
 * @see PhysInterp1DScaledDeviceSumFacTOPKernels.hpp for the SumFacTOP
 * counterpart.
 */

#pragma once

#include <MultiRegions/ElmtOps/BwdTrans/BwdTransDeviceSumFacKernels.hpp>
