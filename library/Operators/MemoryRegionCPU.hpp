///////////////////////////////////////////////////////////////////////////////
//
// File: MemoryRegionCPU.hpp
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

#include <new>

/**
 * @brief Stores underlying data for a Field on the CPU.
 *
 * It acts as a holder for a contiguous block of memory, allocated on the
 * host system.
 *
 * This class also acts as a base class for device-aware builds.
 */
template <typename TData> class MemoryRegionCPU
{
public:
    MemoryRegionCPU()                           = delete;
    MemoryRegionCPU(const MemoryRegionCPU &rhs) = delete;
    MemoryRegionCPU &operator=(const MemoryRegionCPU &rhs) = delete;

    MemoryRegionCPU(MemoryRegionCPU &&rhs)
    {
        m_host      = rhs.m_host;
        m_size      = rhs.m_size;
        m_alignment = rhs.m_alignment;
        rhs.m_host  = nullptr;
    }

    MemoryRegionCPU(size_t n,
                    size_t alignment = __STDCPP_DEFAULT_NEW_ALIGNMENT__)
    {
        // C++17 aligned new
        m_host = static_cast<TData*>(::operator new[](n * sizeof(TData), std::align_val_t(alignment)));
        m_alignment = alignment;
        m_size      = n;
    }

    virtual ~MemoryRegionCPU()
    {
        if (m_host != nullptr)
        {
            operator delete[](m_host, std::align_val_t(m_alignment));
            m_host = nullptr;
        }
    }

    MemoryRegionCPU &operator=(MemoryRegionCPU &&rhs)
    {
        if (m_host)
            operator delete[](m_host, std::align_val_t(m_alignment));

        m_host      = rhs.m_host;
        m_size      = rhs.m_size;
        m_alignment = rhs.m_alignment;
        rhs.m_host  = nullptr;

        return *this;
    }

    /**
     * @brief Get the pointer to the CPU memory.
     *
     * This is a virtual function so that subclasses can move memory to the
     * CPU from a device if needed.
     */
    virtual TData *GetCPUPtr()
    {
        return m_host;
    }

    /**
     * @brief Move memory to the CPU.
     *
     * This is a virtual function so that subclasses can move memory to the
     * CPU from a device if needed.
     */
    virtual void ToCPU()
    {
    }

    size_t size()
    {
        return m_size;
    }

protected:
    TData *m_host      = nullptr;
    size_t m_size      = 0;
    size_t m_alignment = __STDCPP_DEFAULT_NEW_ALIGNMENT__;
};
