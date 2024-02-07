///////////////////////////////////////////////////////////////////////////////
//
// File: MemoryRegionCUDA.hpp
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

#include <utility>

#include "MemoryRegionCPU.hpp"

/**
 * @brief Memory backend for CUDA devices
 * @tparam TData Floating point datatype
 *
 * MemoryRegionCUDA represents and manages the memory stored on a CUDA device.
 * This class also manages access to the CPU part
 * of the memory by inheriting from MemoryRegionCPU.
 */
template <typename TData> class MemoryRegionCUDA : public MemoryRegionCPU<TData>
{
public:
    MemoryRegionCUDA(const MemoryRegionCUDA<TData> &rhs) = delete;
    MemoryRegionCUDA(MemoryRegionCUDA &&rhs)
        : MemoryRegionCPU<TData>(std::move(rhs))
    {
        m_device     = rhs.m_device;
        m_size       = rhs.m_size;
        rhs.m_device = nullptr;
        rhs.m_size   = 0;
    }

    MemoryRegionCUDA(size_t n, size_t alignment = 1);
    virtual ~MemoryRegionCUDA() override;

    /**
     * @brief Create MemoryRegionCUDA from MemoryRegionCPU r-value
     *
     * This method allows for a Field to construct a new MemoryRegionCUDA from a
     * MemoryRegion of any other type, through the MemoryRegionCPU base class
     */
    static MemoryRegionCUDA<TData> fromCPU(MemoryRegionCPU<TData> &&cpu)
    {
        return MemoryRegionCUDA<TData>(std::move(cpu));
    }

    void operator=(MemoryRegionCUDA &&rhs)
    {
        MemoryRegionCPU<TData>::operator=(std::move(rhs));
        m_device                        = rhs.m_device;
        m_size                          = rhs.m_size;
        rhs.m_device                    = nullptr;
        rhs.m_size                      = 0;
    }

    virtual TData *GetCPUPtr() override
    {
        if (m_ondevice)
        {
            DeviceToHost(); // Move to CPU if necessary
        }

        return this->m_host;
    }

    virtual void ToCPU() override
    {
        DeviceToHost();
    }

    TData *GetGPUPtr()
    {
        if (!m_ondevice)
        {
            HostToDevice();
        }

        return m_device;
    }

    void HostToDevice();
    void DeviceToHost();

    bool IsOnDevice() const
    {
        return m_ondevice;
    }

protected:
    MemoryRegionCUDA<TData>(MemoryRegionCPU<TData> &&cpu)
        : MemoryRegionCPU<TData>(std::move(cpu))
    {
        initFromSize(cpu.size());
    }

private:
    void initFromSize(size_t n);

    TData *m_device = nullptr; ///< Device memory pointer
    size_t m_size   = 0;       ///< Device storage size
    bool m_ondevice = false;   ///< Flag indicating if data is on device
};
