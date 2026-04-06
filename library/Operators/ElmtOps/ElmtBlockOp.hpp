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

#include "Operators/Common/BlockOperator.hpp"

#include "Operators/Common/DataWarehouse/BasisDataWarehouse.hpp"
#include "Operators/Common/DataWarehouse/GeometricDataWarehouse.hpp"
#include "Operators/Common/DataWarehouse/ModeIndexDataWarehouse.hpp"
#include "Operators/Common/DataWarehouse/StdMatDataWarehouse.hpp"

namespace Nektar::Operators
{

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

        // No suitible operator was found.
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

    void Apply(BlockAccessor<TData, TFieldIn> &inblock,
               BlockAccessor<TData, TFieldOut> &outblock)
    {
        this->v_Apply(inblock, outblock);
    }

    void operator()(BlockAccessor<TData, TFieldIn> &inblock,
                    BlockAccessor<TData, TFieldOut> &outblock)
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

    virtual void v_Apply(BlockAccessor<TData, TFieldIn> &inblock,
                         BlockAccessor<TData, TFieldOut> &outblock) = 0;
};

template <typename Implementation>
NEK_FORCE_INLINE static constexpr unsigned int GetDeviceBlockSize(
    [[maybe_unused]] const unsigned int blockSize)
{
    if constexpr (std::is_same_v<Implementation, Operators::SumFac>)
    {
        constexpr auto warpsize = NektarSpaces::Device::warpSize;
        return warpsize;
    }
    else if constexpr (std::is_same_v<Implementation, Operators::SumFacTOP>)
    {
        constexpr auto warpsize = NektarSpaces::Device::warpSize;
        return std::min(((blockSize + warpsize - 1u) / warpsize) * warpsize,
                        NektarSpaces::Device::defaultBlockSize);
    }
    else
    {
        return 0;
    }
}

template <typename Implementation>
NEK_FORCE_INLINE static constexpr unsigned int GetDeviceGridSize(
    const size_t nelmt)
{
    size_t maxGridSize = 2147483647;

    if constexpr (std::is_same_v<Implementation, Operators::SumFac>)
    {
        constexpr auto warpsize = NektarSpaces::Device::warpSize;
        return std::min((nelmt + warpsize - 1u) / warpsize, maxGridSize);
    }
    else if constexpr (std::is_same_v<Implementation, Operators::SumFacTOP>)
    {
        return std::min(nelmt, maxGridSize);
    }
    else
    {
        return 0;
    }
}

struct NonTemplated1DPhysSizeParameters
{
    NonTemplated1DPhysSizeParameters(const unsigned int ncoord,
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

private:
    unsigned int m_ncoord;
    unsigned int m_nq0;
};

template <unsigned int tncoord, unsigned int tnq0>
struct Templated1DPhysSizeParameters
{
    static NEK_HOSTDEVICE_INLINE constexpr unsigned int ncoord(void)
    {
        return tncoord;
    }
    static NEK_HOSTDEVICE_INLINE constexpr unsigned int nq0(void)
    {
        return tnq0;
    }
};

struct NonTemplated2DPhysSizeParameters
{
    NonTemplated2DPhysSizeParameters(const unsigned int ncoord,
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

private:
    unsigned int m_ncoord;
    unsigned int m_nq0;
    unsigned int m_nq1;
};

template <unsigned int tncoord, unsigned int tnq0, unsigned int tnq1>
struct Templated2DPhysSizeParameters
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
};

struct NonTemplated3DPhysSizeParameters
{
    NonTemplated3DPhysSizeParameters(const unsigned int nq0,
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

private:
    unsigned int m_nq0;
    unsigned int m_nq1;
    unsigned int m_nq2;
};

template <unsigned int tnq0, unsigned int tnq1, unsigned int tnq2>
struct Templated3DPhysSizeParameters
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
};

struct NonTemplated1DSizeParameters
{
    NonTemplated1DSizeParameters(const unsigned int nm0, const unsigned int nq0)
        : m_nm0(nm0), m_nq0(nq0)
    {
    }

    NEK_HOSTDEVICE_INLINE unsigned int nm0(void) const
    {
        return m_nm0;
    }
    NEK_HOSTDEVICE_INLINE unsigned int nq0(void) const
    {
        return m_nq0;
    }

private:
    unsigned int m_nm0;
    unsigned int m_nq0;
};

template <unsigned int tnm0, unsigned int tnq0> struct Templated1DSizeParameters
{
    static NEK_HOSTDEVICE_INLINE constexpr unsigned int nm0(void)
    {
        return tnm0;
    }
    static NEK_HOSTDEVICE_INLINE constexpr unsigned int nq0(void)
    {
        return tnq0;
    }
};

struct NonTemplated2DSizeParameters
{
    NonTemplated2DSizeParameters(const unsigned int nm0, const unsigned int nm1,
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

private:
    unsigned int m_nm0;
    unsigned int m_nm1;
    unsigned int m_nmTot;
    unsigned int m_nq0;
    unsigned int m_nq1;
};

template <unsigned int tnm0, unsigned int tnm1, unsigned int tnmTot,
          unsigned int tnq0, unsigned int tnq1>
struct Templated2DSizeParameters
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
};

struct NonTemplated3DSizeParameters
{
    NonTemplated3DSizeParameters(const unsigned int nm0, const unsigned int nm1,
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

private:
    unsigned int m_nm0;
    unsigned int m_nm1;
    unsigned int m_nm2;
    unsigned int m_nmTot;
    unsigned int m_nq0;
    unsigned int m_nq1;
    unsigned int m_nq2;
};

template <unsigned int tnm0, unsigned int tnm1, unsigned int tnm2,
          unsigned int tnmTot, unsigned int tnq0, unsigned int tnq1,
          unsigned int tnq2>
struct Templated3DSizeParameters
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
};

} // namespace Nektar::Operators
