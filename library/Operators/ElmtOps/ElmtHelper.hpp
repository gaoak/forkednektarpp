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

#pragma once

#include <LibUtilities/BasicUtils/ShapeType.hpp>

namespace Nektar::Operators::detail
{

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

template <typename T> struct IsNonTemplatedPhysSizeParameter1D : std::false_type
{
};

template <>
struct IsNonTemplatedPhysSizeParameter1D<NonTemplatedPhysSizeParameter1D>
    : std::true_type
{
};

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

template <typename T> struct IsTemplatedPhysSizeParameter1D : std::false_type
{
};

template <unsigned int tncoord, unsigned int tnq0>
struct IsTemplatedPhysSizeParameter1D<
    TemplatedPhysSizeParameter1D<tncoord, tnq0>> : std::true_type
{
};

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

template <typename T> struct IsNonTemplatedPhysSizeParameter2D : std::false_type
{
};

template <>
struct IsNonTemplatedPhysSizeParameter2D<NonTemplatedPhysSizeParameter2D>
    : std::true_type
{
};

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

template <typename T> struct IsTemplatedPhysSizeParameter2D : std::false_type
{
};

template <unsigned int tncoord, unsigned int tnq0, unsigned int tnq1>
struct IsTemplatedPhysSizeParameter2D<
    TemplatedPhysSizeParameter2D<tncoord, tnq0, tnq1>> : std::true_type
{
};

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

template <typename T> struct IsNonTemplatedPhysSizeParameter3D : std::false_type
{
};

template <>
struct IsNonTemplatedPhysSizeParameter3D<NonTemplatedPhysSizeParameter3D>
    : std::true_type
{
};

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

template <typename T> struct IsTemplatedPhysSizeParameter3D : std::false_type
{
};

template <unsigned int tnq0, unsigned int tnq1, unsigned int tnq2>
struct IsTemplatedPhysSizeParameter3D<
    TemplatedPhysSizeParameter3D<tnq0, tnq1, tnq2>> : std::true_type
{
};

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

template <typename T> struct IsNonTemplatedSizeParameter1D : std::false_type
{
};

template <>
struct IsNonTemplatedSizeParameter1D<NonTemplatedSizeParameter1D>
    : std::true_type
{
};

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

template <typename T> struct IsTemplatedSizeParameter1D : std::false_type
{
};

template <unsigned int tnm0, unsigned int tnq0>
struct IsTemplatedSizeParameter1D<TemplatedSizeParameter1D<tnm0, tnq0>>
    : std::true_type
{
};

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

template <typename T> struct IsNonTemplatedSizeParameter2D : std::false_type
{
};

template <>
struct IsNonTemplatedSizeParameter2D<NonTemplatedSizeParameter2D>
    : std::true_type
{
};

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

template <typename T> struct IsTemplatedSizeParameter2D : std::false_type
{
};

template <unsigned int tnm0, unsigned int tnm1, unsigned int tnmTot,
          unsigned int tnq0, unsigned int tnq1>
struct IsTemplatedSizeParameter2D<
    TemplatedSizeParameter2D<tnm0, tnm1, tnmTot, tnq0, tnq1>> : std::true_type
{
};

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

template <typename T> struct IsNonTemplatedSizeParameter3D : std::false_type
{
};

template <>
struct IsNonTemplatedSizeParameter3D<NonTemplatedSizeParameter3D>
    : std::true_type
{
};

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
    /// Longest volume direction, which the launches size against.
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

template <typename T>
struct IsNonTemplatedTraceSizeParameter1D : std::false_type
{
};

template <>
struct IsNonTemplatedTraceSizeParameter1D<NonTemplatedTraceSizeParameter1D>
    : std::true_type
{
};

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

template <typename T> struct IsTemplatedTraceSizeParameter1D : std::false_type
{
};

template <unsigned int tnm0, unsigned int tnq0>
struct IsTemplatedTraceSizeParameter1D<
    TemplatedTraceSizeParameter1D<tnm0, tnq0>> : std::true_type
{
};

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
    /// Longest volume direction, which the launches size against.
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

template <typename T>
struct IsNonTemplatedTraceSizeParameter2D : std::false_type
{
};

template <>
struct IsNonTemplatedTraceSizeParameter2D<NonTemplatedTraceSizeParameter2D>
    : std::true_type
{
};

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

template <typename T> struct IsTemplatedTraceSizeParameter2D : std::false_type
{
};

template <unsigned int tnm0, unsigned int tnm1, unsigned int tnq00,
          unsigned int tnq10>
struct IsTemplatedTraceSizeParameter2D<
    TemplatedTraceSizeParameter2D<tnm0, tnm1, tnq00, tnq10>> : std::true_type
{
};

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
    /// Longest volume direction, which the launches size against.
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

template <typename T>
struct IsNonTemplatedTraceSizeParameter3D : std::false_type
{
};

template <>
struct IsNonTemplatedTraceSizeParameter3D<NonTemplatedTraceSizeParameter3D>
    : std::true_type
{
};

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

// Helper traits
template <typename T> struct IsPhysSizeParameter1D
{
    static constexpr bool value = IsNonTemplatedPhysSizeParameter1D<T>::value ||
                                  IsTemplatedPhysSizeParameter1D<T>::value;
};
template <typename T>
inline constexpr bool IsPhysSizeParameter1D_v = IsPhysSizeParameter1D<T>::value;

template <typename T> struct IsPhysSizeParameter2D
{
    static constexpr bool value = IsNonTemplatedPhysSizeParameter2D<T>::value ||
                                  IsTemplatedPhysSizeParameter2D<T>::value;
};
template <typename T>
inline constexpr bool IsPhysSizeParameter2D_v = IsPhysSizeParameter2D<T>::value;

template <typename T> struct IsPhysSizeParameter3D
{
    static constexpr bool value = IsNonTemplatedPhysSizeParameter3D<T>::value ||
                                  IsTemplatedPhysSizeParameter3D<T>::value;
};
template <typename T>
inline constexpr bool IsPhysSizeParameter3D_v = IsPhysSizeParameter3D<T>::value;

template <typename T> struct IsSizeParameter1D
{
    static constexpr bool value = IsNonTemplatedSizeParameter1D<T>::value ||
                                  IsTemplatedSizeParameter1D<T>::value;
};
template <typename T>
inline constexpr bool IsSizeParameter1D_v = IsSizeParameter1D<T>::value;

template <typename T> struct IsSizeParameter2D
{
    static constexpr bool value = IsNonTemplatedSizeParameter2D<T>::value ||
                                  IsTemplatedSizeParameter2D<T>::value;
};
template <typename T>
inline constexpr bool IsSizeParameter2D_v = IsSizeParameter2D<T>::value;

template <typename T> struct IsSizeParameter3D
{
    static constexpr bool value = IsNonTemplatedSizeParameter3D<T>::value ||
                                  IsTemplatedSizeParameter3D<T>::value;
};
template <typename T>
inline constexpr bool IsSizeParameter3D_v = IsSizeParameter3D<T>::value;

template <typename T> struct IsTraceSizeParameter1D
{
    static constexpr bool value =
        IsNonTemplatedTraceSizeParameter1D<T>::value ||
        IsTemplatedTraceSizeParameter1D<T>::value;
};
template <typename T>
inline constexpr bool IsTraceSizeParameter1D_v =
    IsTraceSizeParameter1D<T>::value;

template <typename T> struct IsTraceSizeParameter2D
{
    static constexpr bool value =
        IsNonTemplatedTraceSizeParameter2D<T>::value ||
        IsTemplatedTraceSizeParameter2D<T>::value;
};
template <typename T>
inline constexpr bool IsTraceSizeParameter2D_v =
    IsTraceSizeParameter2D<T>::value;

template <typename T> struct IsTraceSizeParameter3D
{
    static constexpr bool value =
        IsNonTemplatedTraceSizeParameter3D<T>::value ||
        IsTemplatedTraceSizeParameter3D<T>::value;
};
template <typename T>
inline constexpr bool IsTraceSizeParameter3D_v =
    IsTraceSizeParameter3D<T>::value;

#if defined(NEKTAR_ENABLE_DEVICE)
template <typename Implementation, typename TSizeParameter>
static constexpr unsigned int GetMaxThreadPerBlock(void)
{
    if constexpr (std::is_same_v<Implementation, Operators::SumFac>)
    {
        constexpr auto warpsize = NektarSpaces::Device::warpSize;
        return warpsize;
    }
    else if constexpr (std::is_same_v<Implementation, Operators::SumFacTOP>)
    {
        if constexpr (IsTemplatedSizeParameter1D<TSizeParameter>::value ||
                      IsTemplatedSizeParameter2D<TSizeParameter>::value ||
                      IsTemplatedSizeParameter3D<TSizeParameter>::value)
        {
            return GetDeviceBlockSize<Implementation>(TSizeParameter::nmTot());
        }
        else if constexpr (
            IsTemplatedPhysSizeParameter1D<TSizeParameter>::value ||
            IsTemplatedPhysSizeParameter2D<TSizeParameter>::value ||
            IsTemplatedPhysSizeParameter3D<TSizeParameter>::value)
        {
            return GetDeviceBlockSize<Implementation>(TSizeParameter::nqTot());
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
#endif

} // namespace Nektar::Operators::detail
