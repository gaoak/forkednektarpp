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

} // namespace Nektar::Operators::detail
