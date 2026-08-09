///////////////////////////////////////////////////////////////////////////////
//
// File: NekCuFFTDx.h
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
// Description: cuFFTDx-based z-derivative interface for Nektar++ PhysDeriv.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef NEKTAR_LIB_UTILITIES_FFT_NEKCUFFTDX_H
#define NEKTAR_LIB_UTILITIES_FFT_NEKCUFFTDX_H

#if defined(NEKTAR_ENABLE_CUDA) && defined(NEKTAR_USE_CUFFTDX)

#include <cuda_runtime.h>

namespace Nektar::LibUtilities
{

/// \brief Compute the z-derivative via a fused cuFFTDx kernel (plane-major
/// layout).
///
/// Applies D2Z + wavenumber multiply + Z2D in a single fused kernel.
/// Input/output are in plane-major order: element \c [plane*NXY + xy].
///
/// \param d_in     Device pointer to the input real field.
/// \param d_out    Device pointer to the output real field (may alias d_in).
/// \param NXY      Number of points per homogeneous plane (batch size).
/// \param NPlanes  Number of homogeneous planes (must be even).
/// \param beta     Wavenumber factor \f$2\pi/L_z\f$.
/// \param stream   CUDA stream on which all work is submitted.
void PhysDerivZDx(const double *d_in, double *d_out, int NXY, int NPlanes,
                  double beta, cudaStream_t stream);

/// \brief Compute the z-derivative via a fused cuFFTDx kernel (block device
/// memory layout).
///
/// Like PhysDerivZDx but uses a strided layout where successive planes are
/// separated by \c compStride elements rather than \c NXY.
///
/// \param d_in       Device pointer to the input real field.
/// \param d_out      Device pointer to the output real field (may alias d_in).
/// \param NXY        Number of active points per plane.
/// \param compStride Stride between successive planes in the buffer (>= NXY).
/// \param NPlanes    Number of homogeneous planes (must be even).
/// \param beta       Wavenumber factor \f$2\pi/L_z\f$.
/// \param stream     CUDA stream on which all work is submitted.
void PhysDerivZDxDirect(const double *d_in, double *d_out, int NXY,
                        int compStride, int NPlanes, double beta,
                        cudaStream_t stream);

} // namespace Nektar::LibUtilities

#endif // NEKTAR_ENABLE_CUDA && NEKTAR_USE_CUFFTDX

#endif // NEKTAR_LIB_UTILITIES_FFT_NEKCUFFTDX_H