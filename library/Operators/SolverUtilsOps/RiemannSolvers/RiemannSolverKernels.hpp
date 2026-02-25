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

#include "Operators/LoopExecution/LoopExecution.hpp"
#include <LibUtilities/BasicUtils/NekInline.hpp>

// The dimension and shape kernels. NOTE: They are NOT duplicate
// templated version based on the array size like the
// operators. HOWEVER, they are forced to be INLINED. The inlining is
// critical so that when used in the templated version of the operator
// that loop unrolling occurs.

using namespace Nektar::MultiRegions;

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename TData>
NEK_FORCE_INLINE static void GenerateRotationMatrices(const size_t blksize,
                                                      const TData *normalsPtr,
                                                      TData *rotMatPtr)
{
    // normals
    const TData *nx = normalsPtr + 0u * blksize;
    const TData *ny = normalsPtr + 1u * blksize;
    const TData *nz = normalsPtr + 2u * blksize;

    // Output rotation matrix in the same storage pattern as legacy m_rotMat:
    // m_rotMat[j][i] == rotMatPtr[j*nPts + i], where j is row-major index.
    // R maps: R * n = ex, i.e. rotates normal "n" onto x-axis.
    TData *r0 = rotMatPtr + 0u * blksize; // R00
    TData *r1 = rotMatPtr + 1u * blksize; // R01
    TData *r2 = rotMatPtr + 2u * blksize; // R02
    TData *r3 = rotMatPtr + 3u * blksize; // R10
    TData *r4 = rotMatPtr + 4u * blksize; // R11
    TData *r5 = rotMatPtr + 5u * blksize; // R12
    TData *r6 = rotMatPtr + 6u * blksize; // R20
    TData *r7 = rotMatPtr + 7u * blksize; // R21
    TData *r8 = rotMatPtr + 8u * blksize; // R22

    // Constants (match legacy intent)
    constexpr TData eps  = static_cast<TData>(1e-6);  // EPSILON
    constexpr TData tiny = static_cast<TData>(1e-30); // avoid div0

    // "to" vector is x-axis (1,0,0)
    constexpr TData tox = static_cast<TData>(1);
    constexpr TData toy = static_cast<TData>(0);
    constexpr TData toz = static_cast<TData>(0);

    Nektar::parallel_for<ExecSpace>(
        0u, blksize, NEKTAR_LAMBDA(const size_t i) {
            // from = n (unit)
            const TData fx = nx[i];
            const TData fy = ny[i];
            const TData fz = nz[i];

            // v = from x to = n x ex = (0, nz, -ny)
            const TData vx = fy * toz - fz * toy; // = 0
            const TData vy = fz * tox - fx * toz; // =  fz
            const TData vz = fx * toy - fy * tox; // = -fy

            // e = from . to = nx
            const TData e = fx * tox + fy * toy + fz * toz; // = fx
            const TData f = std::abs(e);

            if (f > static_cast<TData>(1) - eps)
            {
                // Householder-based fallback (matches legacy branch):
                // Build R = H(vvec) * H(uvec), where
                // uvec = x - from, vvec = x - to
                // and x is a basis vector least aligned with "from"
                TData x0 = std::abs(fx);
                TData x1 = std::abs(fy);
                TData x2 = std::abs(fz);

                TData xx, xy, xz;
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
                const TData u0 = xx - fx;
                const TData u1 = xy - fy;
                const TData u2 = xz - fz;

                // vvec = x - to (to = ex)
                const TData v0 = xx - tox;
                const TData v1 = xy - toy;
                const TData v2 = xz - toz;

                const TData uu = u0 * u0 + u1 * u1 + u2 * u2;
                const TData vv = v0 * v0 + v1 * v1 + v2 * v2;
                const TData uv = u0 * v0 + u1 * v1 + u2 * v2;

                const TData c1 = static_cast<TData>(2) / (uu + tiny);
                const TData c2 = static_cast<TData>(2) / (vv + tiny);
                const TData c3 = c1 * c2 * uv;

                // R = I - c1*u*u^T - c2*v*v^T + c3*v*u^T
                // Fill row-major
                const TData uu00 = u0 * u0, uu01 = u0 * u1, uu02 = u0 * u2;
                const TData uu10 = u1 * u0, uu11 = u1 * u1, uu12 = u1 * u2;
                const TData uu20 = u2 * u0, uu21 = u2 * u1, uu22 = u2 * u2;

                const TData vv00 = v0 * v0, vv01 = v0 * v1, vv02 = v0 * v2;
                const TData vv10 = v1 * v0, vv11 = v1 * v1, vv12 = v1 * v2;
                const TData vv20 = v2 * v0, vv21 = v2 * v1, vv22 = v2 * v2;

                const TData vu00 = v0 * u0, vu01 = v0 * u1, vu02 = v0 * u2;
                const TData vu10 = v1 * u0, vu11 = v1 * u1, vu12 = v1 * u2;
                const TData vu20 = v2 * u0, vu21 = v2 * u1, vu22 = v2 * u2;

                r0[i] =
                    static_cast<TData>(1) - c1 * uu00 - c2 * vv00 + c3 * vu00;
                r1[i] =
                    static_cast<TData>(0) - c1 * uu01 - c2 * vv01 + c3 * vu01;
                r2[i] =
                    static_cast<TData>(0) - c1 * uu02 - c2 * vv02 + c3 * vu02;

                r3[i] =
                    static_cast<TData>(0) - c1 * uu10 - c2 * vv10 + c3 * vu10;
                r4[i] =
                    static_cast<TData>(1) - c1 * uu11 - c2 * vv11 + c3 * vu11;
                r5[i] =
                    static_cast<TData>(0) - c1 * uu12 - c2 * vv12 + c3 * vu12;

                r6[i] =
                    static_cast<TData>(0) - c1 * uu20 - c2 * vv20 + c3 * vu20;
                r7[i] =
                    static_cast<TData>(0) - c1 * uu21 - c2 * vv21 + c3 * vu21;
                r8[i] =
                    static_cast<TData>(1) - c1 * uu22 - c2 * vv22 + c3 * vu22;
            }
            else
            {
                // Rodrigues-like closed form:
                // h = 1/(1+e), v = from x to, with to = ex.
                const TData h =
                    static_cast<TData>(1) / (static_cast<TData>(1) + e);

                const TData hvx  = h * vx;
                const TData hvz  = h * vz;
                const TData hvxy = hvx * vy;
                const TData hvxz = hvx * vz;
                const TData hvyz = hvz * vy;

                r0[i] = e + hvx * vx;
                r1[i] = hvxy - vz;
                r2[i] = hvxz + vy;

                r3[i] = hvxy + vz;
                r4[i] = e + h * vy * vy;
                r5[i] = hvyz - vx;

                r6[i] = hvxz - vy;
                r7[i] = hvyz + vx;
                r8[i] = e + hvz * vz;
            }
        });
}

template <typename ExecSpace, unsigned int NDIM, typename TScalar>
NEK_FORCE_INLINE static typename std::enable_if<NDIM == 1>::type
RotateToNormalKernel(const size_t blksize, const TScalar *inptr,
                     const TScalar *normalsptr, TScalar *outptr)
{
    // Explicit vectorisation for AVX backend, vec_t = tinysimd::simd<TScalar>
    // for AVX, vec_t = TScalar otherwise.
    using vec_t =
        typename data_type_if<std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                              TScalar>::type;
    const unsigned int vec_width =
        (std::is_same_v<ExecSpace, NektarSpaces::AVX>)
            ? tinysimd::simd<TScalar>::width
            : 1;

    // in/out layout: [rho | rhou | E]
    const size_t groupsize = blksize / vec_width;
    const vec_t *nx        = reinterpret_cast<const vec_t *>(normalsptr);
    const vec_t *invecptr  = reinterpret_cast<const vec_t *>(inptr);
    vec_t *outvecptr       = reinterpret_cast<vec_t *>(outptr);

    Nektar::parallel_for<ExecSpace>(
        0u, groupsize,
        NEKTAR_LAMBDA(const size_t i) { outvecptr[i] = invecptr[i] * nx[i]; });
}

template <typename ExecSpace, unsigned int NDIM, typename TData>
NEK_FORCE_INLINE static typename std::enable_if<NDIM == 1>::type
RotateFromNormalKernel(const size_t blksize, const TData *inptr,
                       const TData *normalsptr, TData *outptr)
{
    RotateToNormalKernel<ExecSpace, 1>(blksize, inptr, normalsptr, outptr);
}

template <typename ExecSpace, unsigned int NDIM, typename TScalar>
NEK_FORCE_INLINE static typename std::enable_if<NDIM == 2>::type
RotateToNormalKernel(const size_t blksize, const TScalar *inptr,
                     const TScalar *normalsptr, TScalar *outptr)
{
    // Explicit vectorisation for AVX backend, vec_t = tinysimd::simd<TScalar>
    // for AVX, vec_t = TScalar otherwise.
    using vec_t =
        typename data_type_if<std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                              TScalar>::type;
    const unsigned int vec_width =
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

    Nektar::parallel_for<ExecSpace>(
        0u, groupsize, NEKTAR_LAMBDA(const size_t i) {
            rhoOut[i] = rhoIn[i];
            EOut[i]   = EIn[i];

            // Rotate (u,v) to (u_n, u_t) with t = (-ny, nx)
            rhouOut[i] = rhouIn[i] * nx[i] + rhovIn[i] * ny[i]; // u_n
            rhovOut[i] = rhovIn[i] * nx[i] - rhouIn[i] * ny[i]; // u_t
        });
}

template <typename ExecSpace, unsigned int NDIM, typename TScalar>
NEK_FORCE_INLINE static typename std::enable_if<NDIM == 2>::type
RotateFromNormalKernel(const size_t blksize, const TScalar *inptr,
                       const TScalar *normalsptr, TScalar *outptr)
{
    // Explicit vectorisation for AVX backend, vec_t = tinysimd::simd<TScalar>
    // for AVX, vec_t = TScalar otherwise.
    using vec_t =
        typename data_type_if<std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                              TScalar>::type;
    const unsigned int vec_width =
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

    Nektar::parallel_for<ExecSpace>(
        0u, groupsize, NEKTAR_LAMBDA(const size_t i) {
            rhoOut[i] = rhoIn[i];
            EOut[i]   = EIn[i];

            // Rotate (u_n, u_t) to (u,v) with t = (-ny, nx)
            rhouOut[i] = rhouIn[i] * nx[i] - rhovIn[i] * ny[i];
            rhovOut[i] = rhovIn[i] * nx[i] + rhouIn[i] * ny[i];
        });
}

template <typename ExecSpace, unsigned int NDIM, typename TScalar>
NEK_FORCE_INLINE static typename std::enable_if<NDIM == 3>::type
RotateToNormalKernel(const size_t blksize, const TScalar *inptr,
                     const TScalar *rotMatPtr, TScalar *outptr)
{
    // Explicit vectorisation for AVX backend, vec_t = tinysimd::simd<TScalar>
    // for AVX, vec_t = TScalar otherwise.
    using vec_t =
        typename data_type_if<std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                              TScalar>::type;
    const unsigned int vec_width =
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
    Nektar::parallel_for<ExecSpace>(
        0u, groupsize, NEKTAR_LAMBDA(const size_t i) {
            rhoOut[i] = rhoIn[i];
            EOut[i]   = EIn[i];

            const vec_t rhou = rhouIn[i];
            const vec_t rhov = rhovIn[i];
            const vec_t rhow = rhowIn[i];

            rhouOut[i] = rhou * R00[i] + rhov * R01[i] + rhow * R02[i];
            rhovOut[i] = rhou * R10[i] + rhov * R11[i] + rhow * R12[i];
            rhowOut[i] = rhou * R20[i] + rhov * R21[i] + rhow * R22[i];
        });
}

template <typename ExecSpace, unsigned int NDIM, typename TScalar>
NEK_FORCE_INLINE static typename std::enable_if<NDIM == 3>::type
RotateFromNormalKernel(const size_t blksize, const TScalar *inptr,
                       const TScalar *rotMatPtr, TScalar *outptr)
{
    // Explicit vectorisation for AVX backend, vec_t = tinysimd::simd<TScalar>
    // for AVX, vec_t = TScalar otherwise.
    using vec_t =
        typename data_type_if<std::is_same_v<ExecSpace, NektarSpaces::AVX>,
                              TScalar>::type;
    const unsigned int vec_width =
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
    Nektar::parallel_for<ExecSpace>(
        0u, groupsize, NEKTAR_LAMBDA(const size_t i) {
            rhoOut[i] = rhoIn[i];
            EOut[i]   = EIn[i];

            const vec_t a = rhouIn[i];
            const vec_t b = rhovIn[i];
            const vec_t c = rhowIn[i];

            rhouOut[i] = a * R00[i] + b * R10[i] + c * R20[i];
            rhovOut[i] = a * R01[i] + b * R11[i] + c * R21[i];
            rhowOut[i] = a * R02[i] + b * R12[i] + c * R22[i];
        });
}

} // namespace Nektar::Operators::detail
