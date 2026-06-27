///////////////////////////////////////////////////////////////////////////////
//
// File: SYCLQueue.hpp
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

#include <map>
#include <sycl/sycl.hpp>

extern unsigned int internalSYCLDeviceId;

/**
 * @brief wrapper around sycl::queue to ensure queues are not constantly
 * instansiated. Can also use this class for implementing custom device
 * selectors.
 *
 */
class SYCLQueue
{
public:
    static sycl::queue &GetInstance(unsigned int id = 0)
    {
        if (queue.find(id) == queue.end())
        {
#if defined(SYCL_ENABLE_CPU)
            queue[id] = new sycl::queue(sycl::cpu_selector_v,
                                        sycl::property::queue::in_order());
#else
            std::vector<sycl::device> gpu_devices =
                sycl::device::get_devices(sycl::info::device_type::gpu);
            queue[id] = new sycl::queue(gpu_devices[internalSYCLDeviceId],
                                        sycl::property::queue::in_order());
#endif
        }

        return *queue[id];
    }

private:
    static std::map<unsigned int, sycl::queue *> queue;
};
