///////////////////////////////////////////////////////////////////////////////
//
// File: ElmtBlockOp.hpp
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

#include <algorithm>

#include "LibUtilities/Backends/DeviceProperties.hpp"
#include "Operators/Common/BlockOperator.hpp"

#include "Operators/Common/DataWarehouse/BasisDataWarehouse.hpp"
#include "Operators/Common/DataWarehouse/GeometricDataWarehouse.hpp"
#include "Operators/Common/DataWarehouse/ModeIndexDataWarehouse.hpp"
#include "Operators/Common/DataWarehouse/StdMatDataWarehouse.hpp"

namespace Nektar::Operators
{

#if defined(NEKTAR_ENABLE_CUDA) || defined(SYCL_ENABLE_CUDA)
static constexpr unsigned int defaultElmtsPerBlock = 8u;
#elif defined(NEKTAR_ENABLE_HIP) || defined(SYCL_ENABLE_HIP)
static constexpr unsigned int defaultElmtsPerBlock = 4u;
#elif defined(SYCL_ENABLE_INTEL)
static constexpr unsigned int defaultElmtsPerBlock = 4u;
#else
static constexpr unsigned int defaultElmtsPerBlock = 1u;
#endif

// Core implementation types.
struct StdMat
{
    static inline const std::string name = "StdMat";
};

struct SumFac
{
    static inline const std::string name = "SumFac";
};

struct SumFacTOP
{
    static inline const std::string name = "SumFacTOP";
};

struct Generic
{
    static inline const std::string name = "Generic";
};

template <FieldState TFieldIn, FieldState TFieldOut, typename TData>
class ElmtBlockOp : public BlockOperator<TData>
{
public:
    ~ElmtBlockOp() override = default;

    template <template <typename> typename TOperator>
    static std::shared_ptr<TOperator<TData>> Create(
        const unsigned int block_idx,
        const LocalRegions::ExpansionSharedPtr &exp,
        NekDataWarehouseSharedPtr dataWarehouse, const std::string &execStr,
        std::string implStr)
    {
        std::string requestedKey = TOperator<TData>::name + execStr + implStr;

        BlockOperatorFactory<TData> &factory = GetBlockOperatorFactory<TData>();

        // No suitable operator was found.
        if (!factory.ModuleExists(requestedKey))
        {
            // See if there is a Generic implementation.
            auto requestedKey0 = requestedKey;
            requestedKey       = TOperator<TData>::name + execStr + "Generic";

            if (!factory.ModuleExists(requestedKey))
            {
                std::stringstream msg;
                msg << "No such operator: " << requestedKey0 << std::endl;
                factory.PrintAvailableClasses(msg);
                NEKERROR(ErrorUtil::efatal, msg.str());
            }
        }

        return std::static_pointer_cast<TOperator<TData>>(
            factory.CreateInstance(requestedKey, block_idx, exp,
                                   dataWarehouse));
    }

    void Apply(MultiRegions::BlockAccessor<TData, TFieldIn> &inblock,
               MultiRegions::BlockAccessor<TData, TFieldOut> &outblock)
    {
        this->v_Apply(inblock, outblock);
    }

    void operator()(MultiRegions::BlockAccessor<TData, TFieldIn> &inblock,
                    MultiRegions::BlockAccessor<TData, TFieldOut> &outblock)
    {
        this->v_Apply(inblock, outblock);
    }

protected:
    bool m_warnOnceTemplate = false; /// boolean flag to allow one warning

    ElmtBlockOp(const unsigned int block_idx,
                const LocalRegions::ExpansionSharedPtr &exp,
                NekDataWarehouseSharedPtr dataWarehouse)
        : BlockOperator<TData>(block_idx, exp, dataWarehouse)
    {
    }

    virtual void v_Apply(
        MultiRegions::BlockAccessor<TData, TFieldIn> &inblock,
        MultiRegions::BlockAccessor<TData, TFieldOut> &outblock) = 0;
};

#if defined(NEKTAR_ENABLE_DEVICE)
template <typename Implementation>
NEK_FORCE_INLINE static constexpr unsigned int GetDeviceBlockSize(
    [[maybe_unused]] const unsigned int elmtBlockSize)
{
    if constexpr (std::is_same_v<Implementation, Operators::SumFac>)
    {
        constexpr auto warpsize = NektarSpaces::Device::warpSize;
        return warpsize;
    }
    else if constexpr (std::is_same_v<Implementation, Operators::SumFacTOP>)
    {
        constexpr auto warpsize = NektarSpaces::Device::warpSize;
        return std::min(((elmtBlockSize + warpsize - 1u) / warpsize) * warpsize,
                        NektarSpaces::Device::defaultBlockSize);
    }
    else
    {
        return 0;
    }
}

template <typename Implementation,
          unsigned int elmtsPerBlock = defaultElmtsPerBlock>
NEK_FORCE_INLINE static unsigned int GetDeviceGridSize(
    const size_t nelmt, const unsigned int blockSize,
    [[maybe_unused]] const unsigned int shmemsize)
{
    constexpr size_t maxGridSize                 = 2147483647;
    constexpr unsigned int persistentQueueFactor = 2u;

    if constexpr (std::is_same_v<Implementation, Operators::SumFac>)
    {
        return std::min((nelmt + blockSize - 1u) / blockSize, maxGridSize);
    }
    else if constexpr (std::is_same_v<Implementation, Operators::SumFacTOP>)
    {
        GetDeviceProperties::CheckSharedMemoryUsage(shmemsize);

        const auto smCount =
            static_cast<size_t>(GetDeviceProperties::NumMultiProcessors());
        // Number of blocks that can achieve full occupancy on each SM
        const size_t blocksByThreads =
            GetDeviceProperties::MaxThreadsPerMultiprocessor() / blockSize;
        // Number of blocks that can fully utilize shared memory on each SM
        const size_t blocksByShared =
            (shmemsize == 0)
                ? blocksByThreads
                : GetDeviceProperties::SharedMemoryPerMultiprocessor() /
                      shmemsize;
        const size_t residentBlocks = std::min(blocksByThreads, blocksByShared);
        // Number of blocks to launch to keep all SMs busy, controlled by
        // a customizable QueueFactor. This serves as a lower bound on grid size
        const size_t launchBlocks =
            smCount * residentBlocks * persistentQueueFactor;
        const size_t cappedLaunchBlocks =
            std::max<size_t>(1u, std::min(launchBlocks, maxGridSize));

        // Each block processes elmtsPerBlock elements per queue
        // iteration.
        const size_t ngroups = (nelmt + elmtsPerBlock - 1u) / elmtsPerBlock;

        // 1. If nelmt is big, then grid size is determined by ngroups
        // to ensure each block has enough work to do;
        // 2. If nelmt is small, then we try to ensure SMs are busy by
        // launching enough blocks;
        // 3. If nelmt is very small, then each block only process
        // one element to avoid unnecessary idling.
        const unsigned int gridsize = static_cast<unsigned int>(
            std::min(nelmt, std::max(ngroups, cappedLaunchBlocks)));

        return gridsize;
    }
    else
    {
        return 0;
    }
}

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

#endif

} // namespace Nektar::Operators
