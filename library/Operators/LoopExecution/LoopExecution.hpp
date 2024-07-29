///////////////////////////////////////////////////////////////////////////////
//
// File: LoopExecution.hpp
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

#include <limits>

#include "Operators/Common/Spaces.hpp"

#include <LibUtilities/BasicUtils/MiscUtils.hpp>
#include <LibUtilities/BasicUtils/SessionReader.h>

namespace Nektar
{

// Kokkos execution policy and user settable parameters. All of the
// parameters are static (global).
class LoopExecution
{

public:
    static void SetCmdLineArguments(
        std::shared_ptr<LibUtilities::SessionReader> session);

    static bool s_using_device;

#if defined(NEKTAR_ENABLE_KOKKOS)

    enum Kokkos_Policy
    {
        Kokkos_Team_Policy,
        Kokkos_Range_Policy,
        Kokkos_MDRange_Policy
    };

    //////////
    // Sets/Returns whether or not to use available accelerators or
    // co-processors (e.g. GPU, MIC, etc)
    static void setUsingDevice(bool state);
    static bool usingDevice();

    //////////
    // Sets/Gets the number of Kokkos instances per task
    static void setKokkosInstancesPerTask(unsigned int num);
    static unsigned int getKokkosInstancesPerTask();

    //////////
    // Sets/Gets the number of Kokkos leagues that should be used
    // for each loop
    static void setKokkosLeaguesPerLoop(unsigned int num);
    static unsigned int getKokkosLeaguesPerLoop();

    //////////
    // Sets/Gets the number of Kokkos teams to use within an SM for a loop
    static void setKokkosTeamsPerLeague(unsigned int num);
    static unsigned int getKokkosTeamsPerLeague();

    //////////
    // Sets/Gets the Kokkos execution policy
    static void setKokkosPolicy(Kokkos_Policy policy);
    static Kokkos_Policy getKokkosPolicy();

    //////////
    // Sets/Gets the Kokkos chuck size for Kokkos::RangePolicy &
    // Kokkos::TeamPolicy
    static void setKokkosChunkSize(int size);
    static int getKokkosChunkSize();

    //////////
    // Sets/Gets the Kokkos chuck size for Kokkos::MDRangePolicy
    static void setKokkosTileSize(int isize, int jsize, int ksize);
    static void getKokkosTileSize(int &isize, int &jsize, int &ksize);

    static int s_kokkos_instances_per_task;
    static int s_kokkos_leagues_per_loop;
    static int s_kokkos_teams_per_league;

    static Kokkos_Policy s_kokkos_policy;
    static int s_kokkos_chunk_size;
    static int s_kokkos_tile_i_size;
    static int s_kokkos_tile_j_size;
    static int s_kokkos_tile_k_size;

#endif // #if defined(NEKTAR_ENABLE_KOKKOS)
};

// If a functor can take three indices (i,j,k) then this class can be
// used to schlep the range for each. Currently, the funtors use a
// single index.
class BlockRange
{
public:
    enum
    {
        rank = 3
    };

    BlockRange()
    {
    }

    BlockRange(const BlockRange &obj)
    {
        for (int i = 0; i < rank; ++i)
        {
            this->m_offset[i] = obj.m_offset[i];
            this->m_dim[i]    = obj.m_dim[i];
        }
    }

    template <typename ArrayType>
    BlockRange(ArrayType const &c0, ArrayType const &c1)
    {
        setValues(c0, c1);
    }

    template <typename ArrayType>
    void setValues(ArrayType const &c0, ArrayType const &c1)
    {
        for (int i = 0; i < rank; ++i)
        {
            m_offset[i] = c0[i] < c1[i] ? c0[i] : c1[i];
            m_dim[i]    = (c0[i] < c1[i] ? c1[i] : c0[i]) - m_offset[i];
        }
    }

    int begin(int r) const
    {
        return m_offset[r];
    }
    int end(int r) const
    {
        return m_offset[r] + m_dim[r];
    }

    size_t size() const
    {
        size_t result = 1u;
        for (int i = 0; i < rank; ++i)
        {
            result *= m_dim[i];
        }
        return result;
    }

private:
    int m_offset[rank];
    int m_dim[rank];
};

} // namespace Nektar

#include "Operators/LoopExecution/LoopExecutionSerialAVX.hpp"

#include "Operators/LoopExecution/LoopExecutionKokkos.hpp"

#include "Operators/LoopExecution/LoopExecutionCUDA.cuh"
