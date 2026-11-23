///////////////////////////////////////////////////////////////////////////////
//
// File: ElmtHelper.hpp
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
 * @file ElmtHelper.hpp
 * @brief Kernel size-parameter types shared by every element-operator
 * family, the traits that recognise them and the helpers built on them.
 *
 * @details
 * Each sum-factorised (SumFac / SumFacTOP) kernel launcher, and each
 * Generic trace kernel, receives its element dimensions -- mode counts
 * `nm*`, quadrature-point counts `nq*` -- packed into one
 * size-parameter object; the StdMat paths pass their dimensions as
 * plain scalars and run mostly through BLAS. Every dimensionality
 * comes in two forms: `NonTemplated...` carries runtime values (the
 * default path, valid for any order), while `Templated...` exposes the
 * same values as compile-time constants -- instantiated by the
 * CMake-generated ShapeBlock switch code (see
 * LibUtilities/BasicUtils/Switch) for orders within the
 * NEKTAR_SWITCH_MIN/MAX range -- so that inner loops can be unrolled
 * and launch bounds fixed at compile time. Both forms expose the same
 * accessor names, so a kernel written against one compiles against
 * the other; the accompanying `Is...` traits let the kernel launchers
 * check and dispatch on whichever form they are given.
 *
 * Three kinds of size parameter exist:
 * - the plain `SizeParameter` variants carry both modes and points and
 *   serve the transform and inner-product operators;
 * - the `PhysSizeParameter` variants carry only point counts (plus, in
 *   one and two dimensions, the coordinate dimension of the possibly
 *   embedded element) for operators acting purely on physical-space
 *   data, such as PhysDeriv;
 * - the `TraceSizeParameter` variants carry the element's mode counts
 *   together with the point counts of its traces, direction by
 *   direction, for the trace operators (PhysTraceExtract,
 *   IProductWRTPhysTrace, IProductWRTPhysNormalDerivTrace).
 *
 * TransposeSizeParameter() exchanges the mode and quadrature roles of
 * a plain size parameter, and GetMaxThreadPerBlock() derives the
 * compile-time launch bound of a device kernel from its implementation
 * tag and size-parameter type.
 */

#pragma once

#include <limits>

#include <MultiRegions/ElmtOps/ElmtBlockOp.hpp>

#include <LibUtilities/BasicUtils/ShapeType.hpp>

namespace Nektar::MultiRegions::detail
{

/**
 * @brief Runtime physical-space sizes of a 1D element: coordinate
 * dimension and quadrature-point count (see the file notes on the
 * size-parameter scheme).
 */
struct NonTemplatedPhysSizeParameter1D
{
    NonTemplatedPhysSizeParameter1D(const unsigned int ncoord,
                                    const unsigned int nq0)
        : m_ncoord(ncoord), m_nq0(nq0)
    {
    }

    NEK_HOSTDEVICE_INLINE unsigned int ncoord(void) const
    {
        return m_ncoord;
    }
    NEK_HOSTDEVICE_INLINE unsigned int nq0(void) const
    {
        return m_nq0;
    }
    NEK_HOSTDEVICE_INLINE unsigned int nqTot(void) const
    {
        return m_nq0;
    }

private:
    unsigned int m_ncoord;
    unsigned int m_nq0;
};

/// Trait: T is the runtime 1D physical-space size parameter.
template <typename T> struct IsNonTemplatedPhysSizeParameter1D : std::false_type
{
};

template <>
struct IsNonTemplatedPhysSizeParameter1D<NonTemplatedPhysSizeParameter1D>
    : std::true_type
{
};

/**
 * @brief Compile-time counterpart of NonTemplatedPhysSizeParameter1D,
 * instantiated by the generated ShapeBlock switch code for sizes
 * within the switch range.
 */
template <unsigned int tncoord, unsigned int tnq0>
struct TemplatedPhysSizeParameter1D
{
    static NEK_HOSTDEVICE_INLINE constexpr unsigned int ncoord(void)
    {
        return tncoord;
    }
    static NEK_HOSTDEVICE_INLINE constexpr unsigned int nq0(void)
    {
        return tnq0;
    }
    static NEK_HOSTDEVICE_INLINE constexpr unsigned int nqTot(void)
    {
        return tnq0;
    }
};

/// Trait: T is a compile-time 1D physical-space size parameter.
template <typename T> struct IsTemplatedPhysSizeParameter1D : std::false_type
{
};

template <unsigned int tncoord, unsigned int tnq0>
struct IsTemplatedPhysSizeParameter1D<
    TemplatedPhysSizeParameter1D<tncoord, tnq0>> : std::true_type
{
};

/**
 * @brief Runtime physical-space sizes of a 2D element: coordinate
 * dimension and per-direction quadrature-point counts.
 */
struct NonTemplatedPhysSizeParameter2D
{
    NonTemplatedPhysSizeParameter2D(const unsigned int ncoord,
                                    const unsigned int nq0,
                                    const unsigned int nq1)
        : m_ncoord(ncoord), m_nq0(nq0), m_nq1(nq1)
    {
    }

    NEK_HOSTDEVICE_INLINE unsigned int ncoord(void) const
    {
        return m_ncoord;
    }
    NEK_HOSTDEVICE_INLINE unsigned int nq0(void) const
    {
        return m_nq0;
    }
    NEK_HOSTDEVICE_INLINE unsigned int nq1(void) const
    {
        return m_nq1;
    }
    NEK_HOSTDEVICE_INLINE unsigned int nqTot(void) const
    {
        return m_nq0 * m_nq1;
    }

private:
    unsigned int m_ncoord;
    unsigned int m_nq0;
    unsigned int m_nq1;
};

/// Trait: T is the runtime 2D physical-space size parameter.
template <typename T> struct IsNonTemplatedPhysSizeParameter2D : std::false_type
{
};

template <>
struct IsNonTemplatedPhysSizeParameter2D<NonTemplatedPhysSizeParameter2D>
    : std::true_type
{
};

/**
 * @brief Compile-time counterpart of NonTemplatedPhysSizeParameter2D,
 * instantiated by the generated ShapeBlock switch code for sizes
 * within the switch range.
 */
template <unsigned int tncoord, unsigned int tnq0, unsigned int tnq1>
struct TemplatedPhysSizeParameter2D
{
    static NEK_HOSTDEVICE_INLINE constexpr unsigned int ncoord(void)
    {
        return tncoord;
    }
    static NEK_HOSTDEVICE_INLINE constexpr unsigned int nq0(void)
    {
        return tnq0;
    }
    static NEK_HOSTDEVICE_INLINE constexpr unsigned int nq1(void)
    {
        return tnq1;
    }
    static NEK_HOSTDEVICE_INLINE constexpr unsigned int nqTot(void)
    {
        return tnq0 * tnq1;
    }
};

/// Trait: T is a compile-time 2D physical-space size parameter.
template <typename T> struct IsTemplatedPhysSizeParameter2D : std::false_type
{
};

template <unsigned int tncoord, unsigned int tnq0, unsigned int tnq1>
struct IsTemplatedPhysSizeParameter2D<
    TemplatedPhysSizeParameter2D<tncoord, tnq0, tnq1>> : std::true_type
{
};

/**
 * @brief Runtime physical-space sizes of a 3D element: per-direction
 * quadrature-point counts. No coordinate dimension is carried, since a
 * 3D element's coordinate dimension is always three.
 */
struct NonTemplatedPhysSizeParameter3D
{
    NonTemplatedPhysSizeParameter3D(const unsigned int nq0,
                                    const unsigned int nq1,
                                    const unsigned int nq2)
        : m_nq0(nq0), m_nq1(nq1), m_nq2(nq2)
    {
    }

    NEK_HOSTDEVICE_INLINE unsigned int nq0(void) const
    {
        return m_nq0;
    }
    NEK_HOSTDEVICE_INLINE unsigned int nq1(void) const
    {
        return m_nq1;
    }
    NEK_HOSTDEVICE_INLINE unsigned int nq2(void) const
    {
        return m_nq2;
    }
    NEK_HOSTDEVICE_INLINE unsigned int nqTot(void) const
    {
        return m_nq0 * m_nq1 * m_nq2;
    }

private:
    unsigned int m_nq0;
    unsigned int m_nq1;
    unsigned int m_nq2;
};

/// Trait: T is the runtime 3D physical-space size parameter.
template <typename T> struct IsNonTemplatedPhysSizeParameter3D : std::false_type
{
};

template <>
struct IsNonTemplatedPhysSizeParameter3D<NonTemplatedPhysSizeParameter3D>
    : std::true_type
{
};

/**
 * @brief Compile-time counterpart of NonTemplatedPhysSizeParameter3D,
 * instantiated by the generated ShapeBlock switch code for sizes
 * within the switch range.
 */
template <unsigned int tnq0, unsigned int tnq1, unsigned int tnq2>
struct TemplatedPhysSizeParameter3D
{
    static NEK_HOSTDEVICE_INLINE constexpr unsigned int nq0(void)
    {
        return tnq0;
    }
    static NEK_HOSTDEVICE_INLINE constexpr unsigned int nq1(void)
    {
        return tnq1;
    }
    static NEK_HOSTDEVICE_INLINE constexpr unsigned int nq2(void)
    {
        return tnq2;
    }
    static NEK_HOSTDEVICE_INLINE constexpr unsigned int nqTot(void)
    {
        return tnq0 * tnq1 * tnq2;
    }
};

/// Trait: T is a compile-time 3D physical-space size parameter.
template <typename T> struct IsTemplatedPhysSizeParameter3D : std::false_type
{
};

template <unsigned int tnq0, unsigned int tnq1, unsigned int tnq2>
struct IsTemplatedPhysSizeParameter3D<
    TemplatedPhysSizeParameter3D<tnq0, tnq1, tnq2>> : std::true_type
{
};

/**
 * @brief Runtime modal and quadrature sizes of a 1D element: mode
 * count and quadrature-point count.
 */
struct NonTemplatedSizeParameter1D
{
    NonTemplatedSizeParameter1D(const unsigned int nm0, const unsigned int nq0)
        : m_nm0(nm0), m_nq0(nq0)
    {
    }

    NEK_HOSTDEVICE_INLINE unsigned int nm0(void) const
    {
        return m_nm0;
    }
    NEK_HOSTDEVICE_INLINE unsigned int nmTot(void) const
    {
        return m_nm0;
    }
    NEK_HOSTDEVICE_INLINE unsigned int nq0(void) const
    {
        return m_nq0;
    }
    NEK_HOSTDEVICE_INLINE unsigned int nqTot(void) const
    {
        return m_nq0;
    }

private:
    unsigned int m_nm0;
    unsigned int m_nq0;
};

/// Trait: T is the runtime 1D modal/quadrature size parameter.
template <typename T> struct IsNonTemplatedSizeParameter1D : std::false_type
{
};

template <>
struct IsNonTemplatedSizeParameter1D<NonTemplatedSizeParameter1D>
    : std::true_type
{
};

/**
 * @brief Compile-time counterpart of NonTemplatedSizeParameter1D,
 * instantiated by the generated ShapeBlock switch code for sizes
 * within the switch range.
 */
template <unsigned int tnm0, unsigned int tnq0> struct TemplatedSizeParameter1D
{
    static NEK_HOSTDEVICE_INLINE constexpr unsigned int nm0(void)
    {
        return tnm0;
    }
    static NEK_HOSTDEVICE_INLINE constexpr unsigned int nmTot(void)
    {
        return tnm0;
    }
    static NEK_HOSTDEVICE_INLINE constexpr unsigned int nq0(void)
    {
        return tnq0;
    }
    static NEK_HOSTDEVICE_INLINE constexpr unsigned int nqTot(void)
    {
        return tnq0;
    }
};

/// Trait: T is a compile-time 1D modal/quadrature size parameter.
template <typename T> struct IsTemplatedSizeParameter1D : std::false_type
{
};

template <unsigned int tnm0, unsigned int tnq0>
struct IsTemplatedSizeParameter1D<TemplatedSizeParameter1D<tnm0, tnq0>>
    : std::true_type
{
};

/**
 * @brief Runtime modal and quadrature sizes of a 2D element.
 *
 * The total mode count `nmTot` is supplied separately because for
 * simplex-type shapes (e.g. triangles) it is not the product of the
 * per-direction counts.
 */
struct NonTemplatedSizeParameter2D
{
    NonTemplatedSizeParameter2D(const unsigned int nm0, const unsigned int nm1,
                                const unsigned int nmTot,
                                const unsigned int nq0, const unsigned int nq1)
        : m_nm0(nm0), m_nm1(nm1), m_nmTot(nmTot), m_nq0(nq0), m_nq1(nq1)
    {
    }

    NEK_HOSTDEVICE_INLINE unsigned int nm0(void) const
    {
        return m_nm0;
    }
    NEK_HOSTDEVICE_INLINE unsigned int nm1(void) const
    {
        return m_nm1;
    }
    NEK_HOSTDEVICE_INLINE unsigned int nmTot(void) const
    {
        return m_nmTot;
    }
    NEK_HOSTDEVICE_INLINE unsigned int nq0(void) const
    {
        return m_nq0;
    }
    NEK_HOSTDEVICE_INLINE unsigned int nq1(void) const
    {
        return m_nq1;
    }
    NEK_HOSTDEVICE_INLINE unsigned int nqTot(void) const
    {
        return m_nq0 * m_nq1;
    }

private:
    unsigned int m_nm0;
    unsigned int m_nm1;
    unsigned int m_nmTot;
    unsigned int m_nq0;
    unsigned int m_nq1;
};

/// Trait: T is the runtime 2D modal/quadrature size parameter.
template <typename T> struct IsNonTemplatedSizeParameter2D : std::false_type
{
};

template <>
struct IsNonTemplatedSizeParameter2D<NonTemplatedSizeParameter2D>
    : std::true_type
{
};

/**
 * @brief Compile-time counterpart of NonTemplatedSizeParameter2D,
 * instantiated by the generated ShapeBlock switch code for sizes
 * within the switch range.
 */
template <unsigned int tnm0, unsigned int tnm1, unsigned int tnmTot,
          unsigned int tnq0, unsigned int tnq1>
struct TemplatedSizeParameter2D
{
    static NEK_HOSTDEVICE_INLINE constexpr unsigned int nm0(void)
    {
        return tnm0;
    }
    static NEK_HOSTDEVICE_INLINE constexpr unsigned int nm1(void)
    {
        return tnm1;
    }
    static NEK_HOSTDEVICE_INLINE constexpr unsigned int nmTot(void)
    {
        return tnmTot;
    }
    static NEK_HOSTDEVICE_INLINE constexpr unsigned int nq0(void)
    {
        return tnq0;
    }
    static NEK_HOSTDEVICE_INLINE constexpr unsigned int nq1(void)
    {
        return tnq1;
    }
    static NEK_HOSTDEVICE_INLINE constexpr unsigned int nqTot(void)
    {
        return tnq0 * tnq1;
    }
};

/// Trait: T is a compile-time 2D modal/quadrature size parameter.
template <typename T> struct IsTemplatedSizeParameter2D : std::false_type
{
};

template <unsigned int tnm0, unsigned int tnm1, unsigned int tnmTot,
          unsigned int tnq0, unsigned int tnq1>
struct IsTemplatedSizeParameter2D<
    TemplatedSizeParameter2D<tnm0, tnm1, tnmTot, tnq0, tnq1>> : std::true_type
{
};

/**
 * @brief Runtime modal and quadrature sizes of a 3D element.
 *
 * As in 2D, the total mode count `nmTot` is supplied separately
 * because for simplex-type shapes it is not the product of the
 * per-direction counts.
 */
struct NonTemplatedSizeParameter3D
{
    NonTemplatedSizeParameter3D(const unsigned int nm0, const unsigned int nm1,
                                const unsigned int nm2,
                                const unsigned int nmTot,
                                const unsigned int nq0, const unsigned int nq1,
                                const unsigned int nq2)
        : m_nm0(nm0), m_nm1(nm1), m_nm2(nm2), m_nmTot(nmTot), m_nq0(nq0),
          m_nq1(nq1), m_nq2(nq2)
    {
    }

    NEK_HOSTDEVICE_INLINE unsigned int nm0(void) const
    {
        return m_nm0;
    }
    NEK_HOSTDEVICE_INLINE unsigned int nm1(void) const
    {
        return m_nm1;
    }
    NEK_HOSTDEVICE_INLINE unsigned int nm2(void) const
    {
        return m_nm2;
    }
    NEK_HOSTDEVICE_INLINE unsigned int nmTot(void) const
    {
        return m_nmTot;
    }
    NEK_HOSTDEVICE_INLINE unsigned int nq0(void) const
    {
        return m_nq0;
    }
    NEK_HOSTDEVICE_INLINE unsigned int nq1(void) const
    {
        return m_nq1;
    }
    NEK_HOSTDEVICE_INLINE unsigned int nq2(void) const
    {
        return m_nq2;
    }
    NEK_HOSTDEVICE_INLINE unsigned int nqTot(void) const
    {
        return m_nq0 * m_nq1 * m_nq2;
    }

private:
    unsigned int m_nm0;
    unsigned int m_nm1;
    unsigned int m_nm2;
    unsigned int m_nmTot;
    unsigned int m_nq0;
    unsigned int m_nq1;
    unsigned int m_nq2;
};

/// Trait: T is the runtime 3D modal/quadrature size parameter.
template <typename T> struct IsNonTemplatedSizeParameter3D : std::false_type
{
};

template <>
struct IsNonTemplatedSizeParameter3D<NonTemplatedSizeParameter3D>
    : std::true_type
{
};

/**
 * @brief Compile-time counterpart of NonTemplatedSizeParameter3D,
 * instantiated by the generated ShapeBlock switch code for sizes
 * within the switch range.
 */
template <unsigned int tnm0, unsigned int tnm1, unsigned int tnm2,
          unsigned int tnmTot, unsigned int tnq0, unsigned int tnq1,
          unsigned int tnq2>
struct TemplatedSizeParameter3D
{
    static NEK_HOSTDEVICE_INLINE constexpr unsigned int nm0(void)
    {
        return tnm0;
    }
    static NEK_HOSTDEVICE_INLINE constexpr unsigned int nm1(void)
    {
        return tnm1;
    }
    static NEK_HOSTDEVICE_INLINE constexpr unsigned int nm2(void)
    {
        return tnm2;
    }
    static NEK_HOSTDEVICE_INLINE constexpr unsigned int nmTot(void)
    {
        return tnmTot;
    }
    static NEK_HOSTDEVICE_INLINE constexpr unsigned int nq0(void)
    {
        return tnq0;
    }
    static NEK_HOSTDEVICE_INLINE constexpr unsigned int nq1(void)
    {
        return tnq1;
    }
    static NEK_HOSTDEVICE_INLINE constexpr unsigned int nq2(void)
    {
        return tnq2;
    }
    static NEK_HOSTDEVICE_INLINE constexpr unsigned int nqTot(void)
    {
        return tnq0 * tnq1 * tnq2;
    }
};

/// Trait: T is a compile-time 3D modal/quadrature size parameter.
template <typename T> struct IsTemplatedSizeParameter3D : std::false_type
{
};

template <unsigned int tnm0, unsigned int tnm1, unsigned int tnm2,
          unsigned int tnmTot, unsigned int tnq0, unsigned int tnq1,
          unsigned int tnq2>
struct IsTemplatedSizeParameter3D<
    TemplatedSizeParameter3D<tnm0, tnm1, tnm2, tnmTot, tnq0, tnq1, tnq2>>
    : std::true_type
{
};

/**
 * @brief Runtime trace sizes of a 1D element: mode count and
 * quadrature-point count of the segment, whose traces are its two end
 * vertices.
 */
struct NonTemplatedTraceSizeParameter1D
{
    NonTemplatedTraceSizeParameter1D(const unsigned int nm0,
                                     const unsigned int nq0)
        : m_nm0(nm0), m_nq0(nq0)
    {
    }

    NEK_HOSTDEVICE_INLINE unsigned int nm0(void) const
    {
        return m_nm0;
    }
    NEK_HOSTDEVICE_INLINE unsigned int nmTot(void) const
    {
        return m_nm0;
    }
    /// Largest of the per-direction mode counts.
    NEK_HOSTDEVICE_INLINE unsigned int nmMax(void) const
    {
        return m_nm0;
    }
    NEK_HOSTDEVICE_INLINE unsigned int nq0(void) const
    {
        return m_nq0;
    }
    /// Packed trace entries of one element: a segment's traces are its
    /// end vertices, which carry one value each, so this is the vertex
    /// count and nq0() plays no part.
    template <LibUtilities::ShapeType SHAPE_TYPE>
    NEK_HOSTDEVICE_INLINE unsigned int nqTotTrace(void) const
    {
        constexpr unsigned int ntrace0 =
            LibUtilities::ShapeTypeNumTraceInDir[SHAPE_TYPE][0];
        return ntrace0;
    }

private:
    unsigned int m_nm0;
    unsigned int m_nq0;
};

/// Trait: T is the runtime 1D trace size parameter.
template <typename T>
struct IsNonTemplatedTraceSizeParameter1D : std::false_type
{
};

template <>
struct IsNonTemplatedTraceSizeParameter1D<NonTemplatedTraceSizeParameter1D>
    : std::true_type
{
};

/**
 * @brief Compile-time counterpart of NonTemplatedTraceSizeParameter1D,
 * instantiated by the generated trace switch code for sizes within
 * the switch range.
 */
template <unsigned int tnm0, unsigned int tnq0>
struct TemplatedTraceSizeParameter1D
{
    static NEK_HOSTDEVICE_INLINE constexpr unsigned int nm0(void)
    {
        return tnm0;
    }
    static NEK_HOSTDEVICE_INLINE constexpr unsigned int nmTot(void)
    {
        return tnm0;
    }
    static NEK_HOSTDEVICE_INLINE constexpr unsigned int nmMax(void)
    {
        return tnm0;
    }
    static NEK_HOSTDEVICE_INLINE constexpr unsigned int nq0(void)
    {
        return tnq0;
    }
    template <LibUtilities::ShapeType SHAPE_TYPE>
    static NEK_HOSTDEVICE_INLINE constexpr unsigned int nqTotTrace(void)
    {
        constexpr unsigned int ntrace0 =
            LibUtilities::ShapeTypeNumTraceInDir[SHAPE_TYPE][0];
        return ntrace0;
    }
};

/// Trait: T is a compile-time 1D trace size parameter.
template <typename T> struct IsTemplatedTraceSizeParameter1D : std::false_type
{
};

template <unsigned int tnm0, unsigned int tnq0>
struct IsTemplatedTraceSizeParameter1D<
    TemplatedTraceSizeParameter1D<tnm0, tnq0>> : std::true_type
{
};

/**
 * @brief Runtime trace sizes of a 2D element: the element's
 * per-direction mode counts and the quadrature-point count of the
 * edges the shape counts in each direction
 * (LibUtilities::ShapeTypeNumTraceInDir): `nq00` for direction 0,
 * `nq10` for direction 1.
 */
struct NonTemplatedTraceSizeParameter2D
{
    NonTemplatedTraceSizeParameter2D(const unsigned int nm0,
                                     const unsigned int nm1,
                                     const unsigned int nq00,
                                     const unsigned int nq10)
        : m_nm0(nm0), m_nm1(nm1), m_nq00(nq00), m_nq10(nq10)
    {
    }

    NEK_HOSTDEVICE_INLINE unsigned int nm0(void) const
    {
        return m_nm0;
    }
    NEK_HOSTDEVICE_INLINE unsigned int nm1(void) const
    {
        return m_nm1;
    }
    NEK_HOSTDEVICE_INLINE unsigned int nmTot(void) const
    {
        return m_nm0 * m_nm1;
    }
    /// Largest of the per-direction mode counts.
    NEK_HOSTDEVICE_INLINE unsigned int nmMax(void) const
    {
        return std::max(m_nm0, m_nm1);
    }
    NEK_HOSTDEVICE_INLINE unsigned int nq00(void) const
    {
        return m_nq00;
    }
    NEK_HOSTDEVICE_INLINE unsigned int nq10(void) const
    {
        return m_nq10;
    }
    /// Packed trace entries of one element: the direction-0 edges
    /// followed by the direction-1 edges, the per-direction edge counts
    /// coming from the shape.
    template <LibUtilities::ShapeType SHAPE_TYPE>
    NEK_HOSTDEVICE_INLINE unsigned int nqTotTrace(void) const
    {
        constexpr unsigned int ntrace0 =
            LibUtilities::ShapeTypeNumTraceInDir[SHAPE_TYPE][0];
        constexpr unsigned int ntrace1 =
            LibUtilities::ShapeTypeNumTraceInDir[SHAPE_TYPE][1];
        return ntrace0 * m_nq00 + ntrace1 * m_nq10;
    }

private:
    unsigned int m_nm0;
    unsigned int m_nm1;
    unsigned int m_nq00;
    unsigned int m_nq10;
};

/// Trait: T is the runtime 2D trace size parameter.
template <typename T>
struct IsNonTemplatedTraceSizeParameter2D : std::false_type
{
};

template <>
struct IsNonTemplatedTraceSizeParameter2D<NonTemplatedTraceSizeParameter2D>
    : std::true_type
{
};

/**
 * @brief Compile-time counterpart of NonTemplatedTraceSizeParameter2D,
 * instantiated by the generated trace switch code for sizes within
 * the switch range.
 */
template <unsigned int tnm0, unsigned int tnm1, unsigned int tnq00,
          unsigned int tnq10>
struct TemplatedTraceSizeParameter2D
{
    static NEK_HOSTDEVICE_INLINE constexpr unsigned int nm0(void)
    {
        return tnm0;
    }
    static NEK_HOSTDEVICE_INLINE constexpr unsigned int nm1(void)
    {
        return tnm1;
    }
    static NEK_HOSTDEVICE_INLINE constexpr unsigned int nmTot(void)
    {
        return tnm0 * tnm1;
    }
    static NEK_HOSTDEVICE_INLINE constexpr unsigned int nmMax(void)
    {
        return tnm0 > tnm1 ? tnm0 : tnm1;
    }
    static NEK_HOSTDEVICE_INLINE constexpr unsigned int nq00(void)
    {
        return tnq00;
    }
    static NEK_HOSTDEVICE_INLINE constexpr unsigned int nq10(void)
    {
        return tnq10;
    }
    template <LibUtilities::ShapeType SHAPE_TYPE>
    static NEK_HOSTDEVICE_INLINE constexpr unsigned int nqTotTrace(void)
    {
        constexpr unsigned int ntrace0 =
            LibUtilities::ShapeTypeNumTraceInDir[SHAPE_TYPE][0];
        constexpr unsigned int ntrace1 =
            LibUtilities::ShapeTypeNumTraceInDir[SHAPE_TYPE][1];
        return ntrace0 * tnq00 + ntrace1 * tnq10;
    }
};

/// Trait: T is a compile-time 2D trace size parameter.
template <typename T> struct IsTemplatedTraceSizeParameter2D : std::false_type
{
};

template <unsigned int tnm0, unsigned int tnm1, unsigned int tnq00,
          unsigned int tnq10>
struct IsTemplatedTraceSizeParameter2D<
    TemplatedTraceSizeParameter2D<tnm0, tnm1, tnq00, tnq10>> : std::true_type
{
};

/**
 * @brief Runtime trace sizes of a 3D element: the element's
 * per-direction mode counts and the two in-face quadrature-point
 * counts of the faces the shape counts in each direction
 * (LibUtilities::ShapeTypeNumTraceInDir): `nq00`/`nq01` for direction
 * 0, `nq10`/`nq11` for direction 1 and `nq20`/`nq21` for direction 2.
 */
struct NonTemplatedTraceSizeParameter3D
{
    NonTemplatedTraceSizeParameter3D(
        const unsigned int nm0, const unsigned int nm1, const unsigned int nm2,
        const unsigned int nq00, const unsigned int nq01,
        const unsigned int nq10, const unsigned int nq11,
        const unsigned int nq20, const unsigned int nq21)
        : m_nm0(nm0), m_nm1(nm1), m_nm2(nm2), m_nq00(nq00), m_nq01(nq01),
          m_nq10(nq10), m_nq11(nq11), m_nq20(nq20), m_nq21(nq21)
    {
    }

    NEK_HOSTDEVICE_INLINE unsigned int nm0(void) const
    {
        return m_nm0;
    }
    NEK_HOSTDEVICE_INLINE unsigned int nm1(void) const
    {
        return m_nm1;
    }
    NEK_HOSTDEVICE_INLINE unsigned int nm2(void) const
    {
        return m_nm2;
    }
    NEK_HOSTDEVICE_INLINE unsigned int nmTot(void) const
    {
        return m_nm0 * m_nm1 * m_nm2;
    }
    /// Largest of the per-direction mode counts.
    NEK_HOSTDEVICE_INLINE unsigned int nmMax(void) const
    {
        return std::max(m_nm0, std::max(m_nm1, m_nm2));
    }
    NEK_HOSTDEVICE_INLINE unsigned int nq00(void) const
    {
        return m_nq00;
    }
    NEK_HOSTDEVICE_INLINE unsigned int nq01(void) const
    {
        return m_nq01;
    }
    NEK_HOSTDEVICE_INLINE unsigned int nq10(void) const
    {
        return m_nq10;
    }
    NEK_HOSTDEVICE_INLINE unsigned int nq11(void) const
    {
        return m_nq11;
    }
    NEK_HOSTDEVICE_INLINE unsigned int nq20(void) const
    {
        return m_nq20;
    }
    NEK_HOSTDEVICE_INLINE unsigned int nq21(void) const
    {
        return m_nq21;
    }
    /// Packed trace entries of one element: the direction-0 faces
    /// first, then the direction-1 faces, then the direction-2 faces,
    /// the per-direction face counts coming from the shape.
    template <LibUtilities::ShapeType SHAPE_TYPE>
    NEK_HOSTDEVICE_INLINE unsigned int nqTotTrace(void) const
    {
        constexpr unsigned int nface0 =
            LibUtilities::ShapeTypeNumTraceInDir[SHAPE_TYPE][0];
        constexpr unsigned int nface1 =
            LibUtilities::ShapeTypeNumTraceInDir[SHAPE_TYPE][1];
        constexpr unsigned int nface2 =
            LibUtilities::ShapeTypeNumTraceInDir[SHAPE_TYPE][2];
        return nface0 * m_nq00 * m_nq01 + nface1 * m_nq10 * m_nq11 +
               nface2 * m_nq20 * m_nq21;
    }

private:
    unsigned int m_nm0;
    unsigned int m_nm1;
    unsigned int m_nm2;
    unsigned int m_nq00;
    unsigned int m_nq01;
    unsigned int m_nq10;
    unsigned int m_nq11;
    unsigned int m_nq20;
    unsigned int m_nq21;
};

/// Trait: T is the runtime 3D trace size parameter.
template <typename T>
struct IsNonTemplatedTraceSizeParameter3D : std::false_type
{
};

template <>
struct IsNonTemplatedTraceSizeParameter3D<NonTemplatedTraceSizeParameter3D>
    : std::true_type
{
};

/**
 * @brief Compile-time counterpart of NonTemplatedTraceSizeParameter3D,
 * instantiated by the generated trace switch code for sizes within
 * the switch range.
 */
template <unsigned int tnm0, unsigned int tnm1, unsigned int tnm2,
          unsigned int tnq00, unsigned int tnq01, unsigned int tnq10,
          unsigned int tnq11, unsigned int tnq20, unsigned int tnq21>
struct TemplatedTraceSizeParameter3D
{
    static NEK_HOSTDEVICE_INLINE constexpr unsigned int nm0(void)
    {
        return tnm0;
    }
    static NEK_HOSTDEVICE_INLINE constexpr unsigned int nm1(void)
    {
        return tnm1;
    }
    static NEK_HOSTDEVICE_INLINE constexpr unsigned int nm2(void)
    {
        return tnm2;
    }
    static NEK_HOSTDEVICE_INLINE constexpr unsigned int nmTot(void)
    {
        return tnm0 * tnm1 * tnm2;
    }
    static NEK_HOSTDEVICE_INLINE constexpr unsigned int nmMax(void)
    {
        constexpr unsigned int tnmMax01 = tnm0 > tnm1 ? tnm0 : tnm1;
        return tnmMax01 > tnm2 ? tnmMax01 : tnm2;
    }
    static NEK_HOSTDEVICE_INLINE constexpr unsigned int nq00(void)
    {
        return tnq00;
    }
    static NEK_HOSTDEVICE_INLINE constexpr unsigned int nq01(void)
    {
        return tnq01;
    }
    static NEK_HOSTDEVICE_INLINE constexpr unsigned int nq10(void)
    {
        return tnq10;
    }
    static NEK_HOSTDEVICE_INLINE constexpr unsigned int nq11(void)
    {
        return tnq11;
    }
    static NEK_HOSTDEVICE_INLINE constexpr unsigned int nq20(void)
    {
        return tnq20;
    }
    static NEK_HOSTDEVICE_INLINE constexpr unsigned int nq21(void)
    {
        return tnq21;
    }
    template <LibUtilities::ShapeType SHAPE_TYPE>
    static NEK_HOSTDEVICE_INLINE constexpr unsigned int nqTotTrace(void)
    {
        constexpr unsigned int nface0 =
            LibUtilities::ShapeTypeNumTraceInDir[SHAPE_TYPE][0];
        constexpr unsigned int nface1 =
            LibUtilities::ShapeTypeNumTraceInDir[SHAPE_TYPE][1];
        constexpr unsigned int nface2 =
            LibUtilities::ShapeTypeNumTraceInDir[SHAPE_TYPE][2];
        return nface0 * tnq00 * tnq01 + nface1 * tnq10 * tnq11 +
               nface2 * tnq20 * tnq21;
    }
};

/// Trait: T is a compile-time 3D trace size parameter.
template <typename T> struct IsTemplatedTraceSizeParameter3D : std::false_type
{
};

template <unsigned int tnm0, unsigned int tnm1, unsigned int tnm2,
          unsigned int tnq00, unsigned int tnq01, unsigned int tnq10,
          unsigned int tnq11, unsigned int tnq20, unsigned int tnq21>
struct IsTemplatedTraceSizeParameter3D<TemplatedTraceSizeParameter3D<
    tnm0, tnm1, tnm2, tnq00, tnq01, tnq10, tnq11, tnq20, tnq21>>
    : std::true_type
{
};

// Helper traits combining the runtime and compile-time forms: true
// when T is either form of the corresponding size parameter, with an
// `Is..._v` variable shorthand for each. The kernel headers use them in
// static_assert guards on the dimensionality of the size parameter a
// kernel launcher was passed and, in the trace kernels, to constrain
// the launcher overloads via enable_if, so one signature per
// dimensionality serves both forms.

/// T is a 1D physical-space size parameter (either form).
template <typename T> struct IsPhysSizeParameter1D
{
    static constexpr bool value = IsNonTemplatedPhysSizeParameter1D<T>::value ||
                                  IsTemplatedPhysSizeParameter1D<T>::value;
};
template <typename T>
inline constexpr bool IsPhysSizeParameter1D_v = IsPhysSizeParameter1D<T>::value;

/// T is a 2D physical-space size parameter (either form).
template <typename T> struct IsPhysSizeParameter2D
{
    static constexpr bool value = IsNonTemplatedPhysSizeParameter2D<T>::value ||
                                  IsTemplatedPhysSizeParameter2D<T>::value;
};
template <typename T>
inline constexpr bool IsPhysSizeParameter2D_v = IsPhysSizeParameter2D<T>::value;

/// T is a 3D physical-space size parameter (either form).
template <typename T> struct IsPhysSizeParameter3D
{
    static constexpr bool value = IsNonTemplatedPhysSizeParameter3D<T>::value ||
                                  IsTemplatedPhysSizeParameter3D<T>::value;
};
template <typename T>
inline constexpr bool IsPhysSizeParameter3D_v = IsPhysSizeParameter3D<T>::value;

/// T is a 1D modal/quadrature size parameter (either form).
template <typename T> struct IsSizeParameter1D
{
    static constexpr bool value = IsNonTemplatedSizeParameter1D<T>::value ||
                                  IsTemplatedSizeParameter1D<T>::value;
};
template <typename T>
inline constexpr bool IsSizeParameter1D_v = IsSizeParameter1D<T>::value;

/// T is a 2D modal/quadrature size parameter (either form).
template <typename T> struct IsSizeParameter2D
{
    static constexpr bool value = IsNonTemplatedSizeParameter2D<T>::value ||
                                  IsTemplatedSizeParameter2D<T>::value;
};
template <typename T>
inline constexpr bool IsSizeParameter2D_v = IsSizeParameter2D<T>::value;

/// T is a 3D modal/quadrature size parameter (either form).
template <typename T> struct IsSizeParameter3D
{
    static constexpr bool value = IsNonTemplatedSizeParameter3D<T>::value ||
                                  IsTemplatedSizeParameter3D<T>::value;
};
template <typename T>
inline constexpr bool IsSizeParameter3D_v = IsSizeParameter3D<T>::value;

/**
 * @brief Exchange the mode and quadrature roles of a size parameter.
 *
 * A switch describes an operator's two grids in the direction it was
 * written for: BlockOpSwitchPhysInterp1D.h.in sweeps the native counts
 * as the mode side and the scaled counts derived from them as the
 * quadrature side. A kernel always reads the mode side as its source
 * and the quadrature side as its target, so an operator consuming the
 * scaled grid and producing the native one -- PhysGalerkinProject1DScaled,
 * the reverse of PhysInterp1DScaled -- needs those two roles exchanged
 * before it can reuse both the switch and the kernels. Transposing
 * keeps a templated size parameter templated, so the compile-time sizes
 * survive the swap.
 *
 * Both sides are full tensor-product grids here, so the transposed
 * `nmTot` is the product of the original quadrature counts.
 *
 * @tparam TSizeParameter  Plain (modal/quadrature) size parameter of
 *                         any dimensionality, in either form.
 *
 * @param   sizeParam   Size parameter to transpose; read only for the
 *                      runtime forms.
 *
 * @return A size parameter of the same dimensionality and form with the
 * mode and quadrature counts exchanged.
 */
template <typename TSizeParameter>
NEK_FORCE_INLINE static auto TransposeSizeParameter(
    [[maybe_unused]] const TSizeParameter sizeParam)
{
    if constexpr (IsTemplatedSizeParameter1D<TSizeParameter>::value)
    {
        return TemplatedSizeParameter1D<TSizeParameter::nq0(),
                                        TSizeParameter::nm0()>();
    }
    else if constexpr (IsTemplatedSizeParameter2D<TSizeParameter>::value)
    {
        return TemplatedSizeParameter2D<
            TSizeParameter::nq0(), TSizeParameter::nq1(),
            TSizeParameter::nq0() * TSizeParameter::nq1(),
            TSizeParameter::nm0(), TSizeParameter::nm1()>();
    }
    else if constexpr (IsTemplatedSizeParameter3D<TSizeParameter>::value)
    {
        return TemplatedSizeParameter3D<
            TSizeParameter::nq0(), TSizeParameter::nq1(), TSizeParameter::nq2(),
            TSizeParameter::nq0() * TSizeParameter::nq1() *
                TSizeParameter::nq2(),
            TSizeParameter::nm0(), TSizeParameter::nm1(),
            TSizeParameter::nm2()>();
    }
    else if constexpr (IsNonTemplatedSizeParameter1D<TSizeParameter>::value)
    {
        return NonTemplatedSizeParameter1D(sizeParam.nq0(), sizeParam.nm0());
    }
    else if constexpr (IsNonTemplatedSizeParameter2D<TSizeParameter>::value)
    {
        return NonTemplatedSizeParameter2D(sizeParam.nq0(), sizeParam.nq1(),
                                           sizeParam.nq0() * sizeParam.nq1(),
                                           sizeParam.nm0(), sizeParam.nm1());
    }
    else
    {
        static_assert(IsNonTemplatedSizeParameter3D<TSizeParameter>::value,
                      "TransposeSizeParameter expects a 1D, 2D or 3D size "
                      "parameter.");

        return NonTemplatedSizeParameter3D(
            sizeParam.nq0(), sizeParam.nq1(), sizeParam.nq2(),
            sizeParam.nq0() * sizeParam.nq1() * sizeParam.nq2(),
            sizeParam.nm0(), sizeParam.nm1(), sizeParam.nm2());
    }
}

/// T is a 1D trace size parameter (either form).
template <typename T> struct IsTraceSizeParameter1D
{
    static constexpr bool value =
        IsNonTemplatedTraceSizeParameter1D<T>::value ||
        IsTemplatedTraceSizeParameter1D<T>::value;
};
template <typename T>
inline constexpr bool IsTraceSizeParameter1D_v =
    IsTraceSizeParameter1D<T>::value;

/// T is a 2D trace size parameter (either form).
template <typename T> struct IsTraceSizeParameter2D
{
    static constexpr bool value =
        IsNonTemplatedTraceSizeParameter2D<T>::value ||
        IsTemplatedTraceSizeParameter2D<T>::value;
};
template <typename T>
inline constexpr bool IsTraceSizeParameter2D_v =
    IsTraceSizeParameter2D<T>::value;

/// T is a 3D trace size parameter (either form).
template <typename T> struct IsTraceSizeParameter3D
{
    static constexpr bool value =
        IsNonTemplatedTraceSizeParameter3D<T>::value ||
        IsTemplatedTraceSizeParameter3D<T>::value;
};
template <typename T>
inline constexpr bool IsTraceSizeParameter3D_v =
    IsTraceSizeParameter3D<T>::value;

/// Size of the 1D basis matrix held in shared memory.
struct BasisDataSize1D
{
    template <typename TSizeParameter1D>
    NEK_HOSTDEVICE_INLINE constexpr BasisDataSize1D(
        const TSizeParameter1D &sizeParam1D)
    {
        static_assert(
            IsSizeParameter1D_v<TSizeParameter1D>,
            "Template argument must be either of type "
            "NonTemplatedSizeParameter1D or TemplatedSizeParameter1D.");

        const unsigned int nm0 = sizeParam1D.nm0();
        const unsigned int nq0 = sizeParam1D.nq0();

        m_basis0 = nm0 * nq0;

        m_total = m_basis0;
    }

    NEK_HOSTDEVICE_INLINE constexpr unsigned int basis0(void) const
    {
        return m_basis0;
    }
    NEK_HOSTDEVICE_INLINE constexpr unsigned int total(void) const
    {
        return m_total;
    }

protected:
    unsigned int m_basis0 = 0u;
    unsigned int m_total  = 0u;
};

/// Sizes of the two 2D basis matrices held in shared memory.
template <LibUtilities::ShapeType SHAPE_TYPE> struct BasisDataSize2D
{
    template <typename TSizeParameter2D>
    NEK_HOSTDEVICE_INLINE constexpr BasisDataSize2D(
        const TSizeParameter2D &sizeParam2D)
    {
        static_assert(
            IsSizeParameter2D_v<TSizeParameter2D>,
            "Template argument must be either of type "
            "NonTemplatedSizeParameter2D or TemplatedSizeParameter2D.");
        static_assert(SHAPE_TYPE == LibUtilities::Quad ||
                          SHAPE_TYPE == LibUtilities::Tri ||
                          SHAPE_TYPE == LibUtilities::NodalTri,
                      "SHAPE_TYPE must be Quad, Tri or NodalTri.");

        const unsigned int nm0 = sizeParam2D.nm0();
        const unsigned int nm1 = sizeParam2D.nm1();
        const unsigned int nq0 = sizeParam2D.nq0();
        const unsigned int nq1 = sizeParam2D.nq1();

        if constexpr (SHAPE_TYPE == LibUtilities::Quad)
        {
            m_basis0 = nm0 * nq0;
            m_basis1 = nm1 * nq1;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tri ||
                           SHAPE_TYPE == LibUtilities::NodalTri)
        {
            const unsigned int nmTot = sizeParam2D.nmTot();

            m_basis0 = nm0 * nq0;
            m_basis1 = nmTot * nq1;
        }

        m_total = m_basis0 + m_basis1;
    }

    NEK_HOSTDEVICE_INLINE constexpr unsigned int basis0(void) const
    {
        return m_basis0;
    }
    NEK_HOSTDEVICE_INLINE constexpr unsigned int basis1(void) const
    {
        return m_basis1;
    }
    NEK_HOSTDEVICE_INLINE constexpr unsigned int total(void) const
    {
        return m_total;
    }

protected:
    unsigned int m_basis0 = 0u;
    unsigned int m_basis1 = 0u;
    unsigned int m_total  = 0u;
};

/// Sizes of the three 3D basis matrices held in shared memory.
template <LibUtilities::ShapeType SHAPE_TYPE> struct BasisDataSize3D
{
    template <typename TSizeParameter3D>
    NEK_HOSTDEVICE_INLINE constexpr BasisDataSize3D(
        const TSizeParameter3D &sizeParam3D)
    {
        static_assert(
            IsSizeParameter3D_v<TSizeParameter3D>,
            "Template argument must be either of type "
            "NonTemplatedSizeParameter3D or TemplatedSizeParameter3D.");
        static_assert(SHAPE_TYPE == LibUtilities::Hex ||
                          SHAPE_TYPE == LibUtilities::Tet ||
                          SHAPE_TYPE == LibUtilities::NodalTet ||
                          SHAPE_TYPE == LibUtilities::Prism ||
                          SHAPE_TYPE == LibUtilities::NodalPrism ||
                          SHAPE_TYPE == LibUtilities::Pyr ||
                          SHAPE_TYPE == LibUtilities::NodalPyr,
                      "SHAPE_TYPE must be Hex, Tet, NodalTet, Prism, "
                      "NodalPrism, Pyr or NodalPyr.");

        const unsigned int nm0 = sizeParam3D.nm0();
        const unsigned int nm1 = sizeParam3D.nm1();
        const unsigned int nm2 = sizeParam3D.nm2();
        const unsigned int nq0 = sizeParam3D.nq0();
        const unsigned int nq1 = sizeParam3D.nq1();
        const unsigned int nq2 = sizeParam3D.nq2();

        if constexpr (SHAPE_TYPE == LibUtilities::Hex)
        {
            m_basis0 = nm0 * nq0;
            m_basis1 = nm1 * nq1;
            m_basis2 = nm2 * nq2;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Tet ||
                           SHAPE_TYPE == LibUtilities::NodalTet)
        {
            const unsigned int nmTot = sizeParam3D.nmTot();

            m_basis0 = nm0 * nq0;
            m_basis1 = (2u * nm1 - nm0 + 1u) * nm0 / 2u * nq1;
            m_basis2 =
                (nmTot + nm0 * (nm2 - nm1 + 1u) * (nm2 - nm1) / 2u) * nq2;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Prism ||
                           SHAPE_TYPE == LibUtilities::NodalPrism)
        {
            m_basis0 = nm0 * nq0;
            m_basis1 = nm1 * nq1;
            m_basis2 = (2u * nm2 - nm0 + 1u) * nm0 / 2u * nq2;
        }
        else if constexpr (SHAPE_TYPE == LibUtilities::Pyr ||
                           SHAPE_TYPE == LibUtilities::NodalPyr)
        {
            const unsigned int nmTot = sizeParam3D.nmTot();

            m_basis0 = nm0 * nq0;
            m_basis1 = nm1 * nq1;
            m_basis2 =
                (nmTot + nm0 * (nm2 - nm1 + 1u) * (nm2 - nm1) / 2u) * nq2;
        }

        m_total = m_basis0 + m_basis1 + m_basis2;
    }

    NEK_HOSTDEVICE_INLINE constexpr unsigned int basis0(void) const
    {
        return m_basis0;
    }
    NEK_HOSTDEVICE_INLINE constexpr unsigned int basis1(void) const
    {
        return m_basis1;
    }
    NEK_HOSTDEVICE_INLINE constexpr unsigned int basis2(void) const
    {
        return m_basis2;
    }
    NEK_HOSTDEVICE_INLINE constexpr unsigned int total(void) const
    {
        return m_total;
    }

protected:
    unsigned int m_basis0 = 0u;
    unsigned int m_basis1 = 0u;
    unsigned int m_basis2 = 0u;
    unsigned int m_total  = 0u;
};

#if defined(NEKTAR_ENABLE_DEVICE) && defined(DEVICE_COMPILE_ONLY)
/**
 * @brief Compile-time upper bound on the threads per block a kernel of
 * the given implementation is launched with; used in the
 * `__LAUNCH_BOUNDS__` annotation of the device kernel launchers.
 *
 * For SumFac the bound is the warp size, matching GetElmtsPerBlock().
 * For SumFacTOP with a templated (compile-time) plain or physical-space
 * size parameter the bound is the parameter's total mode count (plain
 * forms) or total point count (physical-space forms) rounded up to whole
 * warps, at most the device's default block size, or the default block size
 * where that count is half a warp or less, as such an element may share its
 * block with others (GetElmtsPerBlock()); with any other size
 * parameter -- the runtime
 * forms and the trace forms -- the device's default block size is
 * used, which is the cap GetDeviceKernelParameters() applies at run time.
 * Other implementations return 0.
 *
 * @tparam Implementation   Implementation tag.
 * @tparam TSizeParameter   Size-parameter type the kernel is
 *                          instantiated with.
 *
 * @return Maximum threads per block for the launch-bounds annotation.
 */
template <typename Implementation, typename TSizeParameter>
static constexpr unsigned int GetMaxThreadPerBlock(void)
{
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    if constexpr (std::is_same_v<Implementation, MultiRegions::SumFac>)
    {
        return warpsize;
    }
    else if constexpr (std::is_same_v<Implementation, MultiRegions::SumFacTOP>)
    {
        if constexpr (IsTemplatedSizeParameter1D<TSizeParameter>::value ||
                      IsTemplatedSizeParameter2D<TSizeParameter>::value ||
                      IsTemplatedSizeParameter3D<TSizeParameter>::value)
        {
            constexpr unsigned int ndofs = TSizeParameter::nmTot();
            return (2u * ndofs <= warpsize)
                       ? NektarSpaces::Device::defaultBlockSize
                       : std::min((ndofs + warpsize - 1u) / warpsize * warpsize,
                                  NektarSpaces::Device::defaultBlockSize);
        }
        else if constexpr (
            IsTemplatedPhysSizeParameter1D<TSizeParameter>::value ||
            IsTemplatedPhysSizeParameter2D<TSizeParameter>::value ||
            IsTemplatedPhysSizeParameter3D<TSizeParameter>::value)
        {
            constexpr unsigned int ndofs = TSizeParameter::nqTot();
            return (2u * ndofs <= warpsize)
                       ? NektarSpaces::Device::defaultBlockSize
                       : std::min((ndofs + warpsize - 1u) / warpsize * warpsize,
                                  NektarSpaces::Device::defaultBlockSize);
        }
        else
        {
            return NektarSpaces::Device::defaultBlockSize;
        }
    }
    else
    {
        return 0;
    }
}

/**
 * @brief Call @p func with `std::true_type` if @p useSharedMemoryWsp and
 * `std::false_type` otherwise.
 *
 * A SumFac kernel body selects its workspace from that compile-time
 * flag, so that each branch addresses one memory space only.
 *
 * @param   useSharedMemoryWsp    Stage intermediates in shared memory.
 * @param   func            Kernel body, called with the flag.
 */
template <typename TFunc>
NEK_HOSTDEVICE_INLINE void DispatchWorkSpace(const bool useSharedMemoryWsp,
                                             TFunc &&func)
{
    if (useSharedMemoryWsp)
    {
        func(std::true_type{});
    }
    else
    {
        func(std::false_type{});
    }
}

/// One element per thread block as a compile-time constant. Passed to a
/// SumFacTOP kernel in place of elmtsPerBlock, it folds the element-group
/// indexing of GetDeviceElmtBlock() and GetDeviceSharedWorkSpaceIdx() away.
using SingleElmtPerBlock = std::integral_constant<unsigned int, 1u>;

/// Elements per thread block as a run-time count, which may be several.
/// Passed to a SumFacTOP kernel in place of elmtsPerBlock where its blocks may
/// take several elements (IsPackableSize()).
class MultipleElmtPerBlock
{
public:
    NEK_HOSTDEVICE_INLINE constexpr explicit MultipleElmtPerBlock(
        const unsigned int elmtsPerBlock)
        : m_elmtsPerBlock(elmtsPerBlock)
    {
    }

    NEK_HOSTDEVICE_INLINE constexpr operator unsigned int() const
    {
        return m_elmtsPerBlock;
    }

private:
    unsigned int m_elmtsPerBlock;
};

/**
 * @brief Whether a SumFacTOP block of size parameter @p TSizeParameter may
 * take several elements per thread block (GetElmtsPerBlock()): always for a
 * run-time size, and for a templated one where its smallest mode or point
 * count is half a warp or less; for a templated trace size, the volume point
 * count its modes stand for in the lift operators.
 */
template <typename TSizeParameter> static constexpr bool IsPackableSize(void)
{
    constexpr unsigned int halfWarp = NektarSpaces::Device::warpSize / 2u;

    if constexpr (IsTemplatedSizeParameter1D<TSizeParameter>::value ||
                  IsTemplatedSizeParameter2D<TSizeParameter>::value ||
                  IsTemplatedSizeParameter3D<TSizeParameter>::value)
    {
        return std::min(TSizeParameter::nmTot(), TSizeParameter::nqTot()) <=
               halfWarp;
    }
    else if constexpr (IsTemplatedPhysSizeParameter1D<TSizeParameter>::value ||
                       IsTemplatedPhysSizeParameter2D<TSizeParameter>::value ||
                       IsTemplatedPhysSizeParameter3D<TSizeParameter>::value)
    {
        return TSizeParameter::nqTot() <= halfWarp;
    }
    else if constexpr (IsTemplatedTraceSizeParameter1D<TSizeParameter>::value ||
                       IsTemplatedTraceSizeParameter2D<TSizeParameter>::value ||
                       IsTemplatedTraceSizeParameter3D<TSizeParameter>::value)
    {
        // The volume's point counts, as the lift operators use them.
        return TSizeParameter::nmTot() <= halfWarp;
    }
    else
    {
        return true;
    }
}

/**
 * @brief Call @p func with the elements per thread block a kernel takes:
 * SingleElmtPerBlock for a SumFacTOP launch of one element per block, so
 * that its kernel is compiled for that case alone, MultipleElmtPerBlock for
 * one of several, and @p elmtsPerBlock itself for SumFac.
 *
 * A templated size that cannot pack (IsPackableSize()) takes
 * SingleElmtPerBlock only, and one that can takes MultipleElmtPerBlock only;
 * a run-time size takes either, chosen at run time.
 *
 * @param   sizeParam       Size parameter of the launch.
 * @param   elmtsPerBlock   Elements per thread block, from
 *                          GetDeviceKernelParameters().
 * @param   func            Launch, called with the elements per block.
 */
template <typename Implementation, typename TSizeParameter, typename TFunc>
NEK_FORCE_INLINE static void DispatchElmtsPerBlock(
    [[maybe_unused]] const TSizeParameter sizeParam,
    const unsigned int elmtsPerBlock, TFunc &&func)
{
    constexpr bool templated =
        IsTemplatedSizeParameter1D<TSizeParameter>::value ||
        IsTemplatedSizeParameter2D<TSizeParameter>::value ||
        IsTemplatedSizeParameter3D<TSizeParameter>::value ||
        IsTemplatedPhysSizeParameter1D<TSizeParameter>::value ||
        IsTemplatedPhysSizeParameter2D<TSizeParameter>::value ||
        IsTemplatedPhysSizeParameter3D<TSizeParameter>::value ||
        IsTemplatedTraceSizeParameter1D<TSizeParameter>::value ||
        IsTemplatedTraceSizeParameter2D<TSizeParameter>::value ||
        IsTemplatedTraceSizeParameter3D<TSizeParameter>::value;

    if constexpr (!std::is_same_v<Implementation, SumFacTOP>)
    {
        func(elmtsPerBlock);
    }
    else if constexpr (!IsPackableSize<TSizeParameter>())
    {
        ASSERTL1(elmtsPerBlock == 1u,
                 "This templated size takes one element per thread block.");
        func(SingleElmtPerBlock{});
    }
    else if constexpr (templated)
    {
        func(MultipleElmtPerBlock{elmtsPerBlock});
    }
    else if (elmtsPerBlock == 1u)
    {
        func(SingleElmtPerBlock{});
    }
    else
    {
        func(MultipleElmtPerBlock{elmtsPerBlock});
    }
}
#endif

} // namespace Nektar::MultiRegions::detail

#if defined(NEKTAR_ENABLE_DEVICE) && defined(DEVICE_COMPILE_ONLY)
namespace Nektar
{

/**
 * @brief A thread's share of a SumFacTOP thread block that takes
 * @p elmtsPerBlock elements at a time (GetDeviceKernelParameters()), passed
 * to the shape kernels in place of the thread block.
 *
 * The block's dimension 0 is split evenly between its elements, element first,
 * `getLocalRange<0> / elmtsPerBlock` threads each: getLocalIdx() and
 * getLocalRange() give the thread's place in its element's share in dimension
 * 0, and in the thread block in the others; localBarrier() and the warp
 * functions act on the whole block. A thread with no element, past the block's
 * last or the element count, takes an index past every loop bound and only
 * reaches the barriers; its element is the group's first, so that its pointers
 * stay in range.
 */
template <typename TthreadBlock> struct DeviceElmtBlock
{
    /// The whole thread block.
    const TthreadBlock &threadBlock;
    /// The thread's element.
    size_t elmtIdx;
    /// The thread's index among its element's threads.
    unsigned int elmtThreadIdx;
    /// Threads per element.
    unsigned int elmtThreadRange;
};

/**
 * @brief The calling thread's share of the thread block for the element
 * group starting at @p e0.
 *
 * A kernel loops over the block's element groups with `e0 = getBlockIdx *
 * elmtsPerBlock`, stepping `getBlockRange * elmtsPerBlock`, so that every
 * thread reaches the barriers of each group.
 *
 * @param   e0             First element of the group, below @p nelmt.
 * @param   nelmt          Elements in the block, padding included.
 * @param   elmtsPerBlock  Elements per thread block, from
 *                         GetDeviceKernelParameters().
 * @param   threadBlock    Thread-block handle the index helpers read.
 */
template <typename TthreadBlock>
NEK_DEVICE_INLINE static DeviceElmtBlock<TthreadBlock> GetDeviceElmtBlock(
    const size_t e0, const size_t nelmt, const unsigned int elmtsPerBlock,
    const TthreadBlock &threadBlock)
{
    if (elmtsPerBlock == 1u)
    {
        return {threadBlock, e0, getLocalIdx<0>(threadBlock),
                getLocalRange<0>(threadBlock)};
    }

    constexpr unsigned int inactiveIdx =
        std::numeric_limits<unsigned int>::max() / 2u;

    const unsigned int range = getLocalRange<0>(threadBlock) / elmtsPerBlock;
    const unsigned int tid   = getLocalIdx<0>(threadBlock);
    const unsigned int slot  = tid / range;
    const size_t e           = e0 + slot;
    const bool active        = slot < elmtsPerBlock && e < nelmt;
    return {threadBlock, active ? e : e0, active ? tid % range : inactiveIdx,
            range};
}

/**
 * @brief The index, below @p elmtsPerBlock, of the calling thread's share of
 * its thread block's workspace: `getLocalIdx / (getLocalRange /
 * elmtsPerBlock)`, the last for a thread past the block's last element.
 *
 * @param   elmtsPerBlock  Elements per thread block, from
 *                         GetDeviceKernelParameters().
 * @param   threadBlock    Thread-block handle the index helpers read.
 */
template <typename TthreadBlock>
NEK_DEVICE_INLINE static unsigned int GetDeviceSharedWorkSpaceIdx(
    const unsigned int elmtsPerBlock, const TthreadBlock &threadBlock)
{
    if (elmtsPerBlock == 1u)
    {
        return 0u;
    }
    else
    {
        const unsigned int range =
            getLocalRange<0>(threadBlock) / elmtsPerBlock;
        const unsigned int idx = getLocalIdx<0>(threadBlock) / range;
        return (idx < elmtsPerBlock) ? idx : elmtsPerBlock - 1u;
    }
}

template <unsigned int dim = 0, typename TthreadBlock>
NEK_DEVICE_INLINE static unsigned int getLocalIdx(
    const DeviceElmtBlock<TthreadBlock> &elmtBlock)
{
    if constexpr (dim == 0)
    {
        return elmtBlock.elmtThreadIdx;
    }
    else
    {
        return getLocalIdx<dim>(elmtBlock.threadBlock);
    }
}

template <unsigned int dim = 0, typename TthreadBlock>
NEK_DEVICE_INLINE static unsigned int getLocalRange(
    const DeviceElmtBlock<TthreadBlock> &elmtBlock)
{
    if constexpr (dim == 0)
    {
        return elmtBlock.elmtThreadRange;
    }
    else
    {
        return getLocalRange<dim>(elmtBlock.threadBlock);
    }
}

template <typename TthreadBlock>
NEK_DEVICE_INLINE static void localBarrier(
    const DeviceElmtBlock<TthreadBlock> &elmtBlock)
{
    localBarrier(elmtBlock.threadBlock);
}

template <typename TthreadBlock>
NEK_DEVICE_INLINE static unsigned int getLaneIdx(
    const DeviceElmtBlock<TthreadBlock> &elmtBlock)
{
    return getLaneIdx(elmtBlock.threadBlock);
}

template <typename TData, typename TthreadBlock>
NEK_DEVICE_INLINE static TData warpReduceSum(
    const TData val, const DeviceElmtBlock<TthreadBlock> &elmtBlock)
{
    return warpReduceSum(val, elmtBlock.threadBlock);
}

/**
 * @brief Add @p val, summed over an element's share of the thread block,
 * onto @p red with block-local atomics.
 *
 * Each warp reduces first when the shares are whole warps, and each aligned
 * group of a share's lanes when the shares are a power of two below a warp
 * (GetElmtThreads()), so that the lanes reduced together are of one element;
 * otherwise, and on the SYCL CPU back-ends, where a sub-group need not be
 * consecutive work-items, each thread adds its own value. A thread with no
 * element adds nothing.
 */
template <typename Scope, typename TthreadBlock, typename TData>
NEK_DEVICE_INLINE static void blockReduceSum(
    const TData val, const DeviceElmtBlock<TthreadBlock> &elmtBlock, TData *red)
{
    static_assert(std::is_same_v<Scope, NektarSpaces::LocalScope>,
                  "An element's share reduces into a block-local result.");

    const bool active = elmtBlock.elmtThreadIdx < elmtBlock.elmtThreadRange;

#if !(defined(SYCL_ENABLE_CPU) &&                                              \
      (defined(__ADAPTIVECPP__) || defined(__DPCPP_COMPILER_NATIVE_CPU)))
    constexpr unsigned int warpsize = NektarSpaces::Device::warpSize;

    if (elmtBlock.elmtThreadRange % warpsize == 0u)
    {
        // A share with no element is whole warps, which return together.
        if (active)
        {
            blockReduceSum<Scope>(val, elmtBlock.threadBlock, red);
        }
        return;
    }

    if (warpsize % elmtBlock.elmtThreadRange == 0u)
    {
        // Every lane of the warp takes part, one with no element adding zero.
        const TData tmp = warpSegmentReduceSum(active ? val : TData(0),
                                               elmtBlock.elmtThreadRange,
                                               elmtBlock.threadBlock);
        if (elmtBlock.elmtThreadIdx == 0u)
        {
            atomic_add<Scope>(red, tmp);
        }
        return;
    }
#endif

    if (active)
    {
        atomic_add<Scope>(red, val);
    }
}

} // namespace Nektar
#endif
