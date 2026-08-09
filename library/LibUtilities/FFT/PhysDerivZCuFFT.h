///////////////////////////////////////////////////////////////////////////////
//
// File: PhysDerivZCuFFT.h
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
// Description: cuFFT-based z-derivative (D2Z + wavenumber multiply + Z2D)
//              for the Nektar++ homogeneous-1D PhysDeriv pipeline.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef NEKTAR_LIB_UTILITIES_FFT_PHYSDERIVZCUFFT_H
#define NEKTAR_LIB_UTILITIES_FFT_PHYSDERIVZCUFFT_H

namespace Nektar::LibUtilities
{

/// \brief Compute the z-derivative of a real field in the homogeneous
/// direction using cuFFT (D2Z + wavenumber multiply + Z2D).
///
/// The field is in plane-major layout: \c d_in[plane * compStride + xy].
/// \c d_in and \c d_out may alias the same buffer (in-place is safe).
///
/// \param d_in       Device pointer to the input real field.
/// \param d_out      Device pointer to the output real field.
/// \param nhomo      Number of homogeneous planes (must be even).
/// \param NXY        Number of points per plane (FFT batch size).
/// \param compStride Stride between successive planes in the buffers
///                   (>= NXY; equals NXY when there is no padding).
/// \param beta       Wavenumber factor \f$2\pi/L_z\f$.
/// \param stream     CUDA stream on which all work is submitted.
void PhysDerivZDirect(const double *d_in, double *d_out, int nhomo, int NXY,
                      int compStride, double beta, cudaStream_t stream);

} // namespace Nektar::LibUtilities

#endif // NEKTAR_LIB_UTILITIES_FFT_PHYSDERIVZCUFFT_H
