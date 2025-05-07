///////////////////////////////////////////////////////////////////////////////
//
// File: UtilsKernels.hpp
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

#include "Operators/LoopExecution/LoopExecution.hpp"
#include "Operators/Utils/UtilsDeviceKernels.hpp"
#include "Operators/Utils/UtilsSerialAVXKernels.hpp"

/**
 * @brief Reshapes the storage to a prescribed vector width.
 *
 * This routine reorders the elemental data to interleave elements to a
 * prescribed vector width VW. This therefore puts the same DOF for groups
 * of VW elements contiguously in memory, enabling the efficient use of
 * vectorised instructions. At the moment this routine reshapes from VW0 to
 * VW1 by first reshaping from VW0 to a vector width of 1, and then to VW1.
 *
 * @tparam  interleave_width     Target vector width.
 * @tparam  alignment            Memory alignment to use.
 */
template <typename ExecSpace, unsigned int interleave_width, typename TData>
void ReshapeStorage(const unsigned int curr_interleave_width,
                    const size_t numElmt, const unsigned int ndata,
                    TData *inoutptr)
{
    if (curr_interleave_width != interleave_width)
    {
        // Reshape to scalar shape, if necessary
        if (curr_interleave_width != 1)
        {
            deInterleave<ExecSpace>(curr_interleave_width,
                                    numElmt / curr_interleave_width, ndata,
                                    inoutptr);
        }

        // Reshape to required shape, if necessary
        if (interleave_width != 1)
        {
            ASSERTL0(
                numElmt % interleave_width == 0,
                "Number of elements is not divisible by interleave width.");

            interleave<interleave_width, ExecSpace>(numElmt / interleave_width,
                                                    ndata, inoutptr);
        }
    }
}

/*/// A generic function to reshuffle the map, based on the given interleave or
/// deinterleave map.
template <typename ExecSpace>
void ReshuffleMap(MemoryRegion<int> &deInterleaveMap, MemoryRegion<int> &map)
{
    // assume the map is always in the device memory space
    using MemSpace = typename ExecSpace::memory_space;

    // temporary storage for the map
    MemoryRegion<int> temp =
        MemoryRegion<int>::Create(map.size(), ExecSpace::alignment);
    // copy map to temp
    temp.template Copy<MemSpace>(map);

    // ReMapping using the deinterleave map, temp is used as workspace
    auto deInterleaveMapPtr =
        deInterleaveMap.template GetPtr<MemSpace, ReadWrite>();
    auto tempPtr = temp.template GetPtr<MemSpace, ReadOnly>();
    auto mapPtr  = map.template GetPtr<MemSpace, WriteOnly>();

    // use deinterleave map to reshuffle the temp
    Nektar::parallel_for<ExecSpace>(
        0, map.size(), NEKTAR_LAMBDA(unsigned int i) {
            mapPtr[i] = deInterleaveMapPtr[tempPtr[i]];
        });
}*/

// TODO: Need single block version
/*/// A generic function to build the interleave map for a given field.
template <typename ExecSpace>
void BuildInterleaveMap(std::vector<BlockAttributes> &blocks,
                        const int new_interleave_width,
                        MemoryRegion<int> &deInterleaveMap,
                        MemoryRegion<int> &InterleaveMap)
{
    // assume the map is always in the device memory space
    using MemSpace = typename ExecSpace::memory_space;

    auto deInterleaveMapPtr =
        deInterleaveMap.template GetPtr<MemSpace, WriteOnly>();
    auto InterleaveMapPtr =
        InterleaveMap.template GetPtr<MemSpace, WriteOnly>();

    // Counting the subindex that has been processed so far
    unsigned int offset = 0;

    for (auto &block : blocks)
    {
        block.interleave_width   = new_interleave_width;
        const auto ncoeff        = block.ndata;
        const unsigned int nElmtGroups = block.GetNumElmtGroups();

        // this function fills both InterleaveMap and deInterleaveMap;
        // deInterleaveMap is saved as a member for later use;
        BuildInterleaveMap<ExecSpace>(
            nElmtGroups, ncoeff, new_interleave_width, offset,
            deInterleaveMapPtr, InterleaveMapPtr);

        deInterleaveMapPtr += ncoeff * new_interleave_width * nElmtGroups;
        offset += ncoeff * new_interleave_width * nElmtGroups;
    }
}*/
