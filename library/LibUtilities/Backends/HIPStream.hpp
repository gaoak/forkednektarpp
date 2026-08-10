///////////////////////////////////////////////////////////////////////////////
//
// File: HIPStream.hpp
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

#include <hip/hip_runtime.h>
#include <iostream>
#include <unordered_map>
#include <vector>

namespace Nektar
{

#define CHECK_LAST_HIPCUDA_ERROR()                                             \
    {                                                                          \
        hipError_t err = hipGetLastError();                                    \
        if (err != hipSuccess)                                                 \
        {                                                                      \
            std::cerr << "HIP Runtime Error at: " << __FILE__ << ":"           \
                      << __LINE__ << std::endl;                                \
            std::cerr << hipGetErrorString(err) << std::endl;                  \
            exit(0);                                                           \
        }                                                                      \
    }
#define CHECK_HIPCUDA_ERROR(err)                                               \
    if (err != hipSuccess)                                                     \
    {                                                                          \
        std::cerr << "HIP Runtime Error at: " << __FILE__ << ":" << __LINE__   \
                  << std::endl;                                                \
        std::cerr << hipGetErrorString(err) << std::endl;                      \
        exit(0);                                                               \
    }

class HIPStream
{
public:
    static hipStream_t &GetInstance(unsigned int id)
    {
        if (streams.find(id) == streams.end())
        {
            if (id == 0)
            {
                // Default stream.
                streams[id] = nullptr;
            }
            else
            {
                hipStream_t stream;
                CHECK_HIPCUDA_ERROR(hipStreamCreate(&stream));
                streams[id] = stream;
            }
        }

        return streams[id];
    }

    static std::unordered_map<unsigned int, hipStream_t> &GetAllInstances(void)
    {
        return streams;
    }

    static void RecordEvent(unsigned int id)
    {
        if (events.find(id) == events.end())
        {
            hipEvent_t e;
            CHECK_HIPCUDA_ERROR(
                hipEventCreateWithFlags(&e, hipEventDisableTiming));
            events[id] = e;
        }

        CHECK_HIPCUDA_ERROR(
            hipEventRecord(events[id], HIPStream::GetInstance(id)));
    }

    static hipEvent_t &GetEvent(unsigned int id)
    {
        if (events.find(id) == events.end())
        {
            events[id] = nullptr;
        }

        return events[id];
    }

    static std::unordered_map<unsigned int, hipEvent_t> &GetAllEvents(void)
    {
        return events;
    }

private:
    static std::unordered_map<unsigned int, hipStream_t> streams;
    static std::unordered_map<unsigned int, hipEvent_t> events;
};

} // namespace Nektar
