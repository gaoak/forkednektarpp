///////////////////////////////////////////////////////////////////////////////
//
// File: DeviceOnHost_Host_API.hpp
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

#if defined(NEKTAR_ENABLE_DEVICEONHOST)
static unsigned int deviceOnHostLocalIdxX = 0;
static unsigned int deviceOnHostLocalIdxY = 0;
static unsigned int deviceOnHostLocalIdxZ = 0;
static unsigned int deviceOnHostBlockIdxX = 0;
static unsigned int deviceOnHostBlockIdxY = 0;
static unsigned int deviceOnHostBlockIdxZ = 0;
static unsigned int deviceOnHostBlockDimX = 1;
static unsigned int deviceOnHostBlockDimY = 1;
static unsigned int deviceOnHostBlockDimZ = 1;
static unsigned int deviceOnHostGridDimX  = 1;
static unsigned int deviceOnHostGridDimY  = 1;
static unsigned int deviceOnHostGridDimZ  = 1;

template <unsigned int ndim> class deviceOnHostBlock
{
};

template <typename TData> static void nektar_unused([[maybe_unused]] TData x)
{
    return;
}

// Optional optimisation decorator for a NEK_DEVICE_KERNEL kernel function. This
// should NOT be used in a NEK_DEVCICE_INLINE function. This allows register
// usage optimisation for CUDA/HIP backend by specifying the maximum GPU
// blocksize. Has no effect for SYCL and/or DEVICEONHOST backend.
#define __LAUNCH_BOUNDS__(x)

// Shared memory must be fetched from a NEK_DEVICE_KERNEL kernel function. This
// should NOT be used in a NEK_DEVCICE_INLINE function. Use for compatibility
// with CUDA/HIP backend. Has no effect for SYCL and/or DEVICEONHOST backend.
#define FETCH_SHARED_MEMORY(ptr)

// Kernel launcher on a one-dimensional GPU grid with shared memory provision.
// KERNEL must be a kernel function decorated by NEK_DEVICE_KERNEL. The last two
// arguments of the KERNEL function MUST be of type unsigned char * and
// deviceOnHostBlock<1>. The shared memory size must be specified in bytes. The
// shared memory is declared as unsigned char* type. The shmemptr must then cast
// to the appropriate type before use (e.g. auto ptr = (TData *)shmemptr).
#define DEVICE_1DGRID_KERNEL_LAUNCHER(KERNEL, GRIDSIZE, BLOCKSIZE, SHMEMSIZE,  \
                                      STREAMID, ...)                           \
    {                                                                          \
        deviceOnHostGridDimX  = GRIDSIZE;                                      \
        deviceOnHostBlockDimX = BLOCKSIZE;                                     \
        std::vector<unsigned char> shmem(SHMEMSIZE);                           \
        for (unsigned int i = 0; i < deviceOnHostGridDimX; i++)                \
        {                                                                      \
            for (unsigned int ii = 0; ii < deviceOnHostBlockDimX; ii++)        \
            {                                                                  \
                deviceOnHostLocalIdxX = ii;                                    \
                deviceOnHostBlockIdxX = i;                                     \
                KERNEL(__VA_ARGS__, shmem.data(), deviceOnHostBlock<1>());     \
            }                                                                  \
        }                                                                      \
    }

// Kernel launcher on a two-dimensional GPU grid with shared memory provision.
// KERNEL must be a kernel function decorated by NEK_DEVICE_KERNEL. The last two
// arguments of the KERNEL function MUST be of type unsigned char * and
// deviceOnHostBlock<2>. The shared memory size must be specified in bytes. The
// shared memory is declared as unsigned char* type. The shmemptr must then cast
// to the appropriate type before use (e.g. auto ptr = (TData *)shmemptr).
#define DEVICE_2DGRID_KERNEL_LAUNCHER(KERNEL, GRIDSIZEX, GRIDSIZEY,            \
                                      BLOCKSIZEX, BLOCKSIZEY, SHMEMSIZE,       \
                                      STREAMID, ...)                           \
    {                                                                          \
        deviceOnHostGridDimX  = GRIDSIZEX;                                     \
        deviceOnHostGridDimY  = GRIDSIZEY;                                     \
        deviceOnHostBlockDimX = BLOCKSIZEX;                                    \
        deviceOnHostBlockDimY = BLOCKSIZEY;                                    \
        nektar_unused(STREAMID);                                               \
        std::vector<unsigned char> shmem(SHMEMSIZE);                           \
        for (unsigned int j = 0; j < deviceOnHostGridDimY; j++)                \
        {                                                                      \
            for (unsigned int i = 0; i < deviceOnHostGridDimX; i++)            \
            {                                                                  \
                for (unsigned int jj = 0; jj < deviceOnHostBlockDimY; jj++)    \
                {                                                              \
                    for (unsigned int ii = 0; ii < deviceOnHostBlockDimX;      \
                         ii++)                                                 \
                    {                                                          \
                        deviceOnHostLocalIdxX = ii;                            \
                        deviceOnHostLocalIdxY = jj;                            \
                        deviceOnHostBlockIdxX = i;                             \
                        deviceOnHostBlockIdxY = j;                             \
                        KERNEL(__VA_ARGS__, shmem.data(),                      \
                               deviceOnHostBlock<2>());                        \
                    }                                                          \
                }                                                              \
            }                                                                  \
        }                                                                      \
    }

// Kernel launcher on a three-dimensional GPU grid with shared memory provision.
// KERNEL must be a kernel function decorated by NEK_DEVICE_KERNEL. The last two
// arguments of the KERNEL function MUST be of type unsigned char * and
// deviceOnHostBlock<3>. The shared memory size must be specified in bytes. The
// shared memory is declared as unsigned char* type. The shmemptr must then cast
// to the appropriate type before use (e.g. auto ptr = (TData *)shmemptr).
#define DEVICE_3DGRID_KERNEL_LAUNCHER(KERNEL, GRIDSIZEX, GRIDSIZEY, GRIDSIZEZ, \
                                      BLOCKSIZEX, BLOCKSIZEY, BLOCKSIZEZ,      \
                                      SHMEMSIZE, STREAMID, ...)                \
    {                                                                          \
        deviceOnHostGridDimX  = GRIDSIZEX;                                     \
        deviceOnHostGridDimY  = GRIDSIZEY;                                     \
        deviceOnHostGridDimZ  = GRIDSIZEZ;                                     \
        deviceOnHostBlockDimX = BLOCKSIZEX;                                    \
        deviceOnHostBlockDimY = BLOCKSIZEY;                                    \
        deviceOnHostBlockDimZ = BLOCKSIZEZ;                                    \
        nektar_unused(STREAMID);                                               \
        std::vector<unsigned char> shmem(SHMEMSIZE);                           \
        for (unsigned int k = 0; k < deviceOnHostGridDimZ; k++)                \
        {                                                                      \
            for (unsigned int j = 0; j < deviceOnHostGridDimY; j++)            \
            {                                                                  \
                for (unsigned int i = 0; i < deviceOnHostGridDimX; i++)        \
                {                                                              \
                    for (unsigned int kk = 0; kk < deviceOnHostBlockDimZ;      \
                         kk++)                                                 \
                    {                                                          \
                        for (unsigned int jj = 0; jj < deviceOnHostBlockDimY;  \
                             jj++)                                             \
                        {                                                      \
                            for (unsigned int ii = 0;                          \
                                 ii < deviceOnHostBlockDimX; ii++)             \
                            {                                                  \
                                deviceOnHostLocalIdxX = ii;                    \
                                deviceOnHostLocalIdxY = jj;                    \
                                deviceOnHostLocalIdxZ = kk;                    \
                                deviceOnHostBlockIdxX = i;                     \
                                deviceOnHostBlockIdxY = j;                     \
                                deviceOnHostBlockIdxZ = k;                     \
                                KERNEL(__VA_ARGS__, shmem.data(),              \
                                       deviceOnHostBlock<3>());                \
                            }                                                  \
                        }                                                      \
                    }                                                          \
                }                                                              \
            }                                                                  \
        }                                                                      \
    }

// Kernel launcher on a one-dimensional GPU grid without shared memory
// provision. KERNEL must be a kernel function decorated by NEK_DEVICE_KERNEL.
// The last argument of the KERNEL function MUST be of type
// deviceOnHostBlock<1>.
#define DEVICE_1DGRID_KERNEL_LAUNCHER_NOSHMEM(KERNEL, GRIDSIZE, BLOCKSIZE,     \
                                              STREAMID, ...)                   \
    {                                                                          \
        deviceOnHostGridDimX  = GRIDSIZE;                                      \
        deviceOnHostBlockDimX = BLOCKSIZE;                                     \
        nektar_unused(STREAMID);                                               \
        for (unsigned int i = 0; i < deviceOnHostGridDimX; i++)                \
        {                                                                      \
            for (unsigned int ii = 0; ii < deviceOnHostBlockDimX; ii++)        \
            {                                                                  \
                deviceOnHostLocalIdxX = ii;                                    \
                deviceOnHostBlockIdxX = i;                                     \
                KERNEL(__VA_ARGS__, deviceOnHostBlock<1>());                   \
            }                                                                  \
        }                                                                      \
    }

// Kernel launcher on a two-dimensional GPU grid without shared memory
// provision. KERNEL must be a kernel function decorated by NEK_DEVICE_KERNEL.
// The last argument of the KERNEL function MUST be of type
// deviceOnHostBlock<2>.
#define DEVICE_2DGRID_KERNEL_LAUNCHER_NOSHMEM(                                 \
    KERNEL, GRIDSIZEX, GRIDSIZEY, BLOCKSIZEX, BLOCKSIZEY, STREAMID, ...)       \
    {                                                                          \
        deviceOnHostGridDimX  = GRIDSIZEX;                                     \
        deviceOnHostGridDimY  = GRIDSIZEY;                                     \
        deviceOnHostBlockDimX = BLOCKSIZEX;                                    \
        deviceOnHostBlockDimY = BLOCKSIZEY;                                    \
        nektar_unused(STREAMID);                                               \
        for (unsigned int j = 0; j < deviceOnHostGridDimY; j++)                \
        {                                                                      \
            for (unsigned int i = 0; i < deviceOnHostGridDimX; i++)            \
            {                                                                  \
                for (unsigned int jj = 0; jj < deviceOnHostBlockDimY; jj++)    \
                {                                                              \
                    for (unsigned int ii = 0; ii < deviceOnHostBlockDimX;      \
                         ii++)                                                 \
                    {                                                          \
                        deviceOnHostLocalIdxX = ii;                            \
                        deviceOnHostLocalIdxY = jj;                            \
                        deviceOnHostBlockIdxX = i;                             \
                        deviceOnHostBlockIdxY = j;                             \
                        KERNEL(__VA_ARGS__, deviceOnHostBlock<2>());           \
                    }                                                          \
                }                                                              \
            }                                                                  \
        }                                                                      \
    }

// Kernel launcher on a three-dimensional GPU grid without shared memory
// provision. KERNEL must be a kernel function decorated by NEK_DEVICE_KERNEL.
// The last argument of the KERNEL function MUST be of type
// deviceOnHostBlock<3>.
#define DEVICE_3DGRID_KERNEL_LAUNCHER_NOSHMEM(                                 \
    KERNEL, GRIDSIZEX, GRIDSIZEY, GRIDSIZEZ, BLOCKSIZEX, BLOCKSIZEY,           \
    BLOCKSIZEZ, STREAMID, ...)                                                 \
    {                                                                          \
        deviceOnHostGridDimX  = GRIDSIZEX;                                     \
        deviceOnHostGridDimY  = GRIDSIZEY;                                     \
        deviceOnHostGridDimZ  = GRIDSIZEZ;                                     \
        deviceOnHostBlockDimX = BLOCKSIZEX;                                    \
        deviceOnHostBlockDimY = BLOCKSIZEY;                                    \
        deviceOnHostBlockDimZ = BLOCKSIZEZ;                                    \
        nektar_unused(STREAMID);                                               \
        for (unsigned int k = 0; k < deviceOnHostGridDimZ; k++)                \
        {                                                                      \
            for (unsigned int j = 0; j < deviceOnHostGridDimY; j++)            \
            {                                                                  \
                for (unsigned int i = 0; i < deviceOnHostGridDimX; i++)        \
                {                                                              \
                    for (unsigned int kk = 0; kk < deviceOnHostBlockDimZ;      \
                         kk++)                                                 \
                    {                                                          \
                        for (unsigned int jj = 0; jj < deviceOnHostBlockDimY;  \
                             jj++)                                             \
                        {                                                      \
                            for (unsigned int ii = 0;                          \
                                 ii < deviceOnHostBlockDimX; ii++)             \
                            {                                                  \
                                deviceOnHostLocalIdxX = ii;                    \
                                deviceOnHostLocalIdxY = jj;                    \
                                deviceOnHostLocalIdxZ = kk;                    \
                                deviceOnHostBlockIdxX = i;                     \
                                deviceOnHostBlockIdxY = j;                     \
                                deviceOnHostBlockIdxZ = k;                     \
                                KERNEL(__VA_ARGS__, deviceOnHostBlock<3>());   \
                            }                                                  \
                        }                                                      \
                    }                                                          \
                }                                                              \
            }                                                                  \
        }                                                                      \
    }
#endif
