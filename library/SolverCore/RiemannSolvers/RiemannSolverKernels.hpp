///////////////////////////////////////////////////////////////////////////////
//
// File: RiemannSolverKernels.hpp
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

#include "LibUtilities/LoopExecution/LoopExecution.hpp"

namespace Nektar::SolverCore::detail
{

template <typename ExecSpace, typename TScalar>
NEK_DEVICE_INLINE static void GenerateRotationMatrices(
    const size_t blksize, const TScalar *normalsPtr, TScalar *rotMatPtr)
{
    // Explicit vectorisation for AVX backend, vec_t =
    // tinysimd::simd<TScalar> for AVX, vec_t = TScalar otherwise.
    // using vec_t =
    //    typename data_type_if<std::is_same_v<ExecSpace, NektarSpaces::AVX>,
    //                          TScalar>::type;
    constexpr unsigned int vec_width =
        (std::is_same_v<ExecSpace, NektarSpaces::AVX>)
            ? tinysimd::simd<TScalar>::width
            : 1;

    // Currently not directly vectorisable due to branching
    for (unsigned int k = 0; k < vec_width; k++)
    {
        // normals
        const TScalar *nx = normalsPtr + 0u * blksize;
        const TScalar *ny = normalsPtr + 1u * blksize;
        const TScalar *nz = normalsPtr + 2u * blksize;

        // Output rotation matrix in the same storage pattern as legacy
        // m_rotMat: m_rotMat[j][i] == rotMatPtr[j*nPts + i], where j is
        // row-major index. R maps: R * n = ex, i.e. rotates normal "n" onto
        // x-axis.
        TScalar *r0 = rotMatPtr + 0u * blksize; // R00
        TScalar *r1 = rotMatPtr + 1u * blksize; // R01
        TScalar *r2 = rotMatPtr + 2u * blksize; // R02
        TScalar *r3 = rotMatPtr + 3u * blksize; // R10
        TScalar *r4 = rotMatPtr + 4u * blksize; // R11
        TScalar *r5 = rotMatPtr + 5u * blksize; // R12
        TScalar *r6 = rotMatPtr + 6u * blksize; // R20
        TScalar *r7 = rotMatPtr + 7u * blksize; // R21
        TScalar *r8 = rotMatPtr + 8u * blksize; // R22

        // Constants (match legacy intent)
        constexpr TScalar eps  = static_cast<TScalar>(1e-6);  // EPSILON
        constexpr TScalar tiny = static_cast<TScalar>(1e-30); // avoid div0

        // "to" vector is x-axis (1,0,0)
        constexpr TScalar tox = static_cast<TScalar>(1);
        constexpr TScalar toy = static_cast<TScalar>(0);
        constexpr TScalar toz = static_cast<TScalar>(0);

        // from = n (unit)
        const TScalar fx = nx[0];
        const TScalar fy = ny[0];
        const TScalar fz = nz[0];

        // v = from x to = n x ex = (0, nz, -ny)
        const TScalar vx = fy * toz - fz * toy; // = 0
        const TScalar vy = fz * tox - fx * toz; // =  fz
        const TScalar vz = fx * toy - fy * tox; // = -fy

        // e = from . to = nx
        const TScalar e = fx * tox + fy * toy + fz * toz; // = fx
        const TScalar f = std::abs(e);

        if (f > static_cast<TScalar>(1) - eps)
        {
            // Householder-based fallback (matches legacy branch):
            // Build R = H(vvec) * H(uvec), where
            // uvec = x - from, vvec = x - to
            // and x is a basis vector least aligned with "from"
            TScalar x0 = std::abs(fx);
            TScalar x1 = std::abs(fy);
            TScalar x2 = std::abs(fz);

            TScalar xx, xy, xz;
            if (x0 < x1)
            {
                if (x0 < x2)
                {
                    xx = 1;
                    xy = 0;
                    xz = 0;
                }
                else
                {
                    xx = 0;
                    xy = 0;
                    xz = 1;
                }
            }
            else
            {
                if (x1 < x2)
                {
                    xx = 0;
                    xy = 1;
                    xz = 0;
                }
                else
                {
                    xx = 0;
                    xy = 0;
                    xz = 1;
                }
            }

            // u = x - from
            const TScalar u0 = xx - fx;
            const TScalar u1 = xy - fy;
            const TScalar u2 = xz - fz;

            // vvec = x - to (to = ex)
            const TScalar v0 = xx - tox;
            const TScalar v1 = xy - toy;
            const TScalar v2 = xz - toz;

            const TScalar uu = u0 * u0 + u1 * u1 + u2 * u2;
            const TScalar vv = v0 * v0 + v1 * v1 + v2 * v2;
            const TScalar uv = u0 * v0 + u1 * v1 + u2 * v2;

            const TScalar c1 = static_cast<TScalar>(2) / (uu + tiny);
            const TScalar c2 = static_cast<TScalar>(2) / (vv + tiny);
            const TScalar c3 = c1 * c2 * uv;

            // R = I - c1*u*u^T - c2*v*v^T + c3*v*u^T
            // Fill row-major
            const TScalar uu00 = u0 * u0, uu01 = u0 * u1, uu02 = u0 * u2;
            const TScalar uu10 = u1 * u0, uu11 = u1 * u1, uu12 = u1 * u2;
            const TScalar uu20 = u2 * u0, uu21 = u2 * u1, uu22 = u2 * u2;

            const TScalar vv00 = v0 * v0, vv01 = v0 * v1, vv02 = v0 * v2;
            const TScalar vv10 = v1 * v0, vv11 = v1 * v1, vv12 = v1 * v2;
            const TScalar vv20 = v2 * v0, vv21 = v2 * v1, vv22 = v2 * v2;

            const TScalar vu00 = v0 * u0, vu01 = v0 * u1, vu02 = v0 * u2;
            const TScalar vu10 = v1 * u0, vu11 = v1 * u1, vu12 = v1 * u2;
            const TScalar vu20 = v2 * u0, vu21 = v2 * u1, vu22 = v2 * u2;

            r0[0] = static_cast<TScalar>(1) - c1 * uu00 - c2 * vv00 + c3 * vu00;
            r1[0] = static_cast<TScalar>(0) - c1 * uu01 - c2 * vv01 + c3 * vu01;
            r2[0] = static_cast<TScalar>(0) - c1 * uu02 - c2 * vv02 + c3 * vu02;

            r3[0] = static_cast<TScalar>(0) - c1 * uu10 - c2 * vv10 + c3 * vu10;
            r4[0] = static_cast<TScalar>(1) - c1 * uu11 - c2 * vv11 + c3 * vu11;
            r5[0] = static_cast<TScalar>(0) - c1 * uu12 - c2 * vv12 + c3 * vu12;

            r6[0] = static_cast<TScalar>(0) - c1 * uu20 - c2 * vv20 + c3 * vu20;
            r7[0] = static_cast<TScalar>(0) - c1 * uu21 - c2 * vv21 + c3 * vu21;
            r8[0] = static_cast<TScalar>(1) - c1 * uu22 - c2 * vv22 + c3 * vu22;
        }
        else
        {
            // Rodrigues-like closed form:
            // h = 1/(1+e), v = from x to, with to = ex.
            const TScalar h =
                static_cast<TScalar>(1) / (static_cast<TScalar>(1) + e);

            const TScalar hvx  = h * vx;
            const TScalar hvz  = h * vz;
            const TScalar hvxy = hvx * vy;
            const TScalar hvxz = hvx * vz;
            const TScalar hvyz = hvz * vy;

            r0[0] = e + hvx * vx;
            r1[0] = hvxy - vz;
            r2[0] = hvxz + vy;

            r3[0] = hvxy + vz;
            r4[0] = e + h * vy * vy;
            r5[0] = hvyz - vx;

            r6[0] = hvxz - vy;
            r7[0] = hvyz + vx;
            r8[0] = e + hvz * vz;
        }

        normalsPtr++;
        rotMatPtr++;
    }
}

template <typename ExecSpace, unsigned int NDIM, typename TScalar,
          std::enable_if_t<NDIM == 1, bool> Enable = true>
NEK_DEVICE_INLINE static void RotateToNormalKernel(
    [[maybe_unused]] const size_t blksize, const TScalar *inptr,
    const TScalar *normalsptr, TScalar *outptr)
{
    // Explicit vectorisation for AVX backend, vec_t = tinysimd::simd<TScalar>
    // for AVX, vec_t = TScalar otherwise.
    using vec_t =
        typename data_type_if<std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                              TScalar>::type;

    // in/out layout: [rho | rhou | E]
    const vec_t *nx       = reinterpret_cast<const vec_t *>(normalsptr);
    const vec_t *invecptr = reinterpret_cast<const vec_t *>(inptr);
    vec_t *outvecptr      = reinterpret_cast<vec_t *>(outptr);

    outvecptr[0] = invecptr[0] * nx[0];
}

template <typename ExecSpace, unsigned int NDIM, typename TScalar,
          std::enable_if_t<NDIM == 1, bool> Enable = true>
NEK_DEVICE_INLINE static void RotateFromNormalKernel(const size_t blksize,
                                                     const TScalar *inptr,
                                                     const TScalar *normalsptr,
                                                     TScalar *outptr)
{
    RotateToNormalKernel<ExecSpace, 1>(blksize, inptr, normalsptr, outptr);
}

template <typename ExecSpace, unsigned int NDIM, typename TScalar,
          std::enable_if_t<NDIM == 2, bool> Enable = true>
NEK_DEVICE_INLINE static void RotateToNormalKernel(const size_t blksize,
                                                   const TScalar *inptr,
                                                   const TScalar *normalsptr,
                                                   TScalar *outptr)
{
    // Explicit vectorisation for AVX backend, vec_t = tinysimd::simd<TScalar>
    // for AVX, vec_t = TScalar otherwise.
    using vec_t =
        typename data_type_if<std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                              TScalar>::type;
    constexpr unsigned int vec_width =
        (std::is_same_v<ExecSpace, NektarSpaces::AVX>)
            ? tinysimd::simd<TScalar>::width
            : 1;

    // Block layout: [rho | u | v | E], each of length blksize
    const size_t groupsize = blksize / vec_width;
    const vec_t *invecptr  = reinterpret_cast<const vec_t *>(inptr);
    const vec_t *rhoIn     = invecptr;
    const vec_t *rhouIn    = invecptr + groupsize;
    const vec_t *rhovIn    = invecptr + 2 * groupsize;
    const vec_t *EIn       = invecptr + 3 * groupsize;

    vec_t *outvecptr = reinterpret_cast<vec_t *>(outptr);
    vec_t *rhoOut    = outvecptr;
    vec_t *rhouOut   = outvecptr + groupsize;     // stores u_n
    vec_t *rhovOut   = outvecptr + 2 * groupsize; // stores u_t
    vec_t *EOut      = outvecptr + 3 * groupsize;

    // Normals layout: [nx | ny], each of length groupsize
    const vec_t *nx = reinterpret_cast<const vec_t *>(normalsptr);
    const vec_t *ny = reinterpret_cast<const vec_t *>(normalsptr) + groupsize;

    rhoOut[0] = rhoIn[0];
    EOut[0]   = EIn[0];

    // Rotate (u,v) to (u_n, u_t) with t = (-ny, nx)
    rhouOut[0] = rhouIn[0] * nx[0] + rhovIn[0] * ny[0]; // u_n
    rhovOut[0] = rhovIn[0] * nx[0] - rhouIn[0] * ny[0]; // u_t
}

template <typename ExecSpace, unsigned int NDIM, typename TScalar,
          std::enable_if_t<NDIM == 2, bool> Enable = true>
NEK_DEVICE_INLINE static void RotateFromNormalKernel(const size_t blksize,
                                                     const TScalar *inptr,
                                                     const TScalar *normalsptr,
                                                     TScalar *outptr)
{
    // Explicit vectorisation for AVX backend, vec_t = tinysimd::simd<TScalar>
    // for AVX, vec_t = TScalar otherwise.
    using vec_t =
        typename data_type_if<std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                              TScalar>::type;
    constexpr unsigned int vec_width =
        (std::is_same_v<ExecSpace, NektarSpaces::AVX>)
            ? tinysimd::simd<TScalar>::width
            : 1;

    // Block layout: [rho | u | v | E], each of length groupsize
    const size_t groupsize = blksize / vec_width;
    const vec_t *invecptr  = reinterpret_cast<const vec_t *>(inptr);
    const vec_t *rhoIn     = invecptr;
    const vec_t *rhouIn    = invecptr + groupsize;
    const vec_t *rhovIn    = invecptr + 2 * groupsize;
    const vec_t *EIn       = invecptr + 3 * groupsize;

    vec_t *outvecptr = reinterpret_cast<vec_t *>(outptr);
    vec_t *rhoOut    = outvecptr;
    vec_t *rhouOut   = outvecptr + groupsize;
    vec_t *rhovOut   = outvecptr + 2 * groupsize;
    vec_t *EOut      = outvecptr + 3 * groupsize;

    // Normals layout: [nx | ny], each of length groupsize
    const vec_t *nx = reinterpret_cast<const vec_t *>(normalsptr);
    const vec_t *ny = reinterpret_cast<const vec_t *>(normalsptr) + groupsize;

    rhoOut[0] = rhoIn[0];
    EOut[0]   = EIn[0];

    // Rotate (u_n, u_t) to (u,v) with t = (-ny, nx)
    rhouOut[0] = rhouIn[0] * nx[0] - rhovIn[0] * ny[0];
    rhovOut[0] = rhovIn[0] * nx[0] + rhouIn[0] * ny[0];
}

template <typename ExecSpace, unsigned int NDIM, typename TScalar,
          std::enable_if_t<NDIM == 3, bool> Enable = true>
NEK_DEVICE_INLINE static void RotateToNormalKernel(const size_t blksize,
                                                   const TScalar *inptr,
                                                   const TScalar *rotMatPtr,
                                                   TScalar *outptr)
{
    // Explicit vectorisation for AVX backend, vec_t = tinysimd::simd<TScalar>
    // for AVX, vec_t = TScalar otherwise.
    using vec_t =
        typename data_type_if<std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                              TScalar>::type;
    constexpr unsigned int vec_width =
        (std::is_same_v<ExecSpace, NektarSpaces::AVX>)
            ? tinysimd::simd<TScalar>::width
            : 1;

    // inptr/outptr: [rho | rhou | rhov | rhow | E], each length groupsize
    const size_t groupsize = blksize / vec_width;
    const vec_t *invecptr  = reinterpret_cast<const vec_t *>(inptr);
    const vec_t *rhoIn     = invecptr;
    const vec_t *rhouIn    = invecptr + groupsize;
    const vec_t *rhovIn    = invecptr + 2 * groupsize;
    const vec_t *rhowIn    = invecptr + 3 * groupsize;
    const vec_t *EIn       = invecptr + 4 * groupsize;

    vec_t *outvecptr = reinterpret_cast<vec_t *>(outptr);
    vec_t *rhoOut    = outvecptr;
    vec_t *rhouOut   = outvecptr + groupsize;
    vec_t *rhovOut   = outvecptr + 2 * groupsize;
    vec_t *rhowOut   = outvecptr + 3 * groupsize;
    vec_t *EOut      = outvecptr + 4 * groupsize;

    // rotMatPtr layout (SoA like legacy m_rotMat[0..8]):
    // [R00 | R01 | R02 | R10 | R11 | R12 | R20 | R21 | R22], each length
    // groupsize
    const vec_t *rotMatVecPtr = reinterpret_cast<const vec_t *>(rotMatPtr);
    const vec_t *R00          = rotMatVecPtr + 0 * groupsize;
    const vec_t *R01          = rotMatVecPtr + 1 * groupsize;
    const vec_t *R02          = rotMatVecPtr + 2 * groupsize;
    const vec_t *R10          = rotMatVecPtr + 3 * groupsize;
    const vec_t *R11          = rotMatVecPtr + 4 * groupsize;
    const vec_t *R12          = rotMatVecPtr + 5 * groupsize;
    const vec_t *R20          = rotMatVecPtr + 6 * groupsize;
    const vec_t *R21          = rotMatVecPtr + 7 * groupsize;
    const vec_t *R22          = rotMatVecPtr + 8 * groupsize;

    // out[vx] = in[vx]*R00 + in[vy]*R01 + in[vz]*R02
    // out[vy] = in[vx]*R10 + in[vy]*R11 + in[vz]*R12
    // out[vz] = in[vx]*R20 + in[vy]*R21 + in[vz]*R22
    rhoOut[0] = rhoIn[0];
    EOut[0]   = EIn[0];

    const vec_t rhou = rhouIn[0];
    const vec_t rhov = rhovIn[0];
    const vec_t rhow = rhowIn[0];

    rhouOut[0] = rhou * R00[0] + rhov * R01[0] + rhow * R02[0];
    rhovOut[0] = rhou * R10[0] + rhov * R11[0] + rhow * R12[0];
    rhowOut[0] = rhou * R20[0] + rhov * R21[0] + rhow * R22[0];
}

template <typename ExecSpace, unsigned int NDIM, typename TScalar,
          std::enable_if_t<NDIM == 3, bool> Enable = true>
NEK_DEVICE_INLINE static void RotateFromNormalKernel(const size_t blksize,
                                                     const TScalar *inptr,
                                                     const TScalar *rotMatPtr,
                                                     TScalar *outptr)
{
    // Explicit vectorisation for AVX backend, vec_t = tinysimd::simd<TScalar>
    // for AVX, vec_t = TScalar otherwise.
    using vec_t =
        typename data_type_if<std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                              TScalar>::type;
    constexpr unsigned int vec_width =
        (std::is_same_v<ExecSpace, NektarSpaces::AVX>)
            ? tinysimd::simd<TScalar>::width
            : 1;

    // inptr/outptr: [rho | rhou | rhov | rhow | E], each length groupsize
    const size_t groupsize = blksize / vec_width;
    const vec_t *invecptr  = reinterpret_cast<const vec_t *>(inptr);
    const vec_t *rhoIn     = invecptr;
    const vec_t *rhouIn    = invecptr + groupsize;
    const vec_t *rhovIn    = invecptr + 2 * groupsize;
    const vec_t *rhowIn    = invecptr + 3 * groupsize;
    const vec_t *EIn       = invecptr + 4 * groupsize;

    vec_t *outvecptr = reinterpret_cast<vec_t *>(outptr);
    vec_t *rhoOut    = outvecptr;
    vec_t *rhouOut   = outvecptr + groupsize;
    vec_t *rhovOut   = outvecptr + 2 * groupsize;
    vec_t *rhowOut   = outvecptr + 3 * groupsize;
    vec_t *EOut      = outvecptr + 4 * groupsize;

    // Rotation matrix layout (SoA like legacy m_rotMat[0..8]):
    // [R00 | R01 | R02 | R10 | R11 | R12 | R20 | R21 | R22], each length
    // groupsize
    const vec_t *rotMatVecPtr = reinterpret_cast<const vec_t *>(rotMatPtr);
    const vec_t *R00          = rotMatVecPtr + 0 * groupsize;
    const vec_t *R01          = rotMatVecPtr + 1 * groupsize;
    const vec_t *R02          = rotMatVecPtr + 2 * groupsize;
    const vec_t *R10          = rotMatVecPtr + 3 * groupsize;
    const vec_t *R11          = rotMatVecPtr + 4 * groupsize;
    const vec_t *R12          = rotMatVecPtr + 5 * groupsize;
    const vec_t *R20          = rotMatVecPtr + 6 * groupsize;
    const vec_t *R21          = rotMatVecPtr + 7 * groupsize;
    const vec_t *R22          = rotMatVecPtr + 8 * groupsize;

    // out[vx] = in[vx]*R00 + in[vy]*R10 + in[vz]*R20
    // out[vy] = in[vx]*R01 + in[vy]*R11 + in[vz]*R21
    // out[vz] = in[vx]*R02 + in[vy]*R12 + in[vz]*R22
    rhoOut[0] = rhoIn[0];
    EOut[0]   = EIn[0];

    const vec_t a = rhouIn[0];
    const vec_t b = rhovIn[0];
    const vec_t c = rhowIn[0];

    rhouOut[0] = a * R00[0] + b * R10[0] + c * R20[0];
    rhovOut[0] = a * R01[0] + b * R11[0] + c * R21[0];
    rhowOut[0] = a * R02[0] + b * R12[0] + c * R22[0];
}

} // namespace Nektar::SolverCore::detail
