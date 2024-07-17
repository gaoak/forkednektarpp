///////////////////////////////////////////////////////////////////////////////
//
// File: LoopExecution.cpp
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

#include "LoopExecution.hpp"

#include <LibUtilities/BasicUtils/ErrorUtil.hpp>

namespace Nektar
{

bool LoopExecution::s_using_device = false;

// Kokkos command line arguments.
#if defined(NEKTAR_ENABLE_KOKKOS)
std::string kokkosCmdPolicy =
    LibUtilities::SessionReader::RegisterCmdLineArgument(
        "kokkos_policy", "",
        "Kokkos Execution Policy - range (default), mdrange, team");

std::string kokkosCmdLeaguesPerLoop =
    LibUtilities::SessionReader::RegisterCmdLineArgument(
        "kokkos_leagues_per_loop", "",
        "Kokkos TeamPolicy number of leagues (work items)"
        "per loop (default 1).");

std::string kokkosCmdTeamsPerLeague =
    LibUtilities::SessionReader::RegisterCmdLineArgument(
        "kokkos_teams_per_league", "",
        "Kokkos TeamPolicy number of teams (threads) "
        "per Kokkos TeamPolicy league (default 256/16).");

std::string kokkosCmdChunkSize =
    LibUtilities::SessionReader::RegisterCmdLineArgument(
        "kokkos_chunk_size", "",
        "Kokkos TeamPolicy and RangePolicy chunk size.");

std::string kokkosCmdTileSize =
    LibUtilities::SessionReader::RegisterCmdLineArgument(
        "kokkos_tile_size", "", "Kokkos MDRangePolicy tile size.");

// Kokkos execution policy and user settable parameters.
LoopExecution::Kokkos_Policy LoopExecution::s_kokkos_policy =
    LoopExecution::Kokkos_Range_Policy;
int LoopExecution::s_kokkos_chunk_size       = -1;
int LoopExecution::s_kokkos_tile_i_size      = -1;
int LoopExecution::s_kokkos_tile_j_size      = -1;
int LoopExecution::s_kokkos_tile_k_size      = -1;
int LoopExecution::s_kokkos_teams_per_league = -1;
int LoopExecution::s_kokkos_leagues_per_loop = -1;

#endif // #if !defined(NEKTAR_ENABLE_KOKKOS)

// Set/Get Kokkos command line functions.

// This function is called when an operator is created. The call is
// redundant as it only needs to be called once.
void LoopExecution::SetCmdLineArguments(
    [[maybe_unused]] std::shared_ptr<LibUtilities::SessionReader> session)
{
#if defined(NEKTAR_ENABLE_KOKKOS)
    // Kokkos specific defaults.

    // Set GPU parameters (NOTE: This could be autotuned if knowledge of
    // how many elements are assigned to this MPI rank and how many SMs
    // are on this particular machine.)

    // TODO, only display if gpu mode is turned on and if these values
    // weren't set.
#if defined(KOKKOS_USING_GPU)
    if (s_using_device)
    {
        if (s_kokkos_leagues_per_loop <= 0)
        {
            s_kokkos_leagues_per_loop = 1;
        }

        if (s_kokkos_teams_per_league <= 0)
        {
            s_kokkos_teams_per_league = 256;
        }
    }
#endif

#if defined(KOKKOS_ENABLE_OPENMP)
    if (s_kokkos_leagues_per_loop <= 0)
    {
        s_kokkos_leagues_per_loop = 1;
    }

    if (s_kokkos_teams_per_league <= 0)
    {
        s_kokkos_teams_per_league = 16;
    }
#endif

    if (session->DefinesCmdLineArgument("kokkos_policy"))
    {
        std::string cmdValue =
            session->GetCmdLineArgument<std::string>("kokkos_policy");

        if (cmdValue == "Range" || cmdValue == "range")
            LoopExecution::s_kokkos_policy = Kokkos_Range_Policy;
        else if (cmdValue == "MDRange" || cmdValue == "mdrange")
            LoopExecution::s_kokkos_policy = Kokkos_MDRange_Policy;
        else if (cmdValue == "Team" || cmdValue == "team")
            LoopExecution::s_kokkos_policy = Kokkos_Team_Policy;
        else
        {
            NEKERROR(Nektar::ErrorUtil::efatal,
                     "Bad command line argument for kokkos_policy: " +
                         cmdValue);
        }
    }

    if (session->DefinesCmdLineArgument("kokkos_leagues_per_loop"))
    {
        if (LoopExecution::s_kokkos_policy == Kokkos_Team_Policy)
        {
            int cmdValue =
                session->GetCmdLineArgument<int>("kokkos_leagues_per_loop");
            if (cmdValue > 0)
                s_kokkos_leagues_per_loop = cmdValue;
            else
            {
                NEKERROR(
                    Nektar::ErrorUtil::efatal,
                    "Bad command line argument for kokkos_leagues_per_loop. "
                    "Value must be greater than zero.");
            }
        }
        else
        {
            NEKERROR(Nektar::ErrorUtil::efatal,
                     "Bad command line argument: kokkos_leagues_per_loop "
                     "set but not using the Kokkos Team Policy");
        }
    }

    if (session->DefinesCmdLineArgument("kokkos_teams_per_league"))
    {
        if (LoopExecution::s_kokkos_policy == Kokkos_Team_Policy)
        {
            int cmdValue =
                session->GetCmdLineArgument<int>("kokkos_teams_per_league");
            if (cmdValue > 0)
                s_kokkos_teams_per_league = cmdValue;
            else
            {
                NEKERROR(
                    Nektar::ErrorUtil::efatal,
                    "Bad command line argument for kokkos_teams_per_league. "
                    "Value must be greater than zero.");
            }
        }
        else
        {
            NEKERROR(Nektar::ErrorUtil::efatal,
                     "Bad command line argument: kokkos_teams_per_league "
                     "set but not using the Kokkos Team Policy");
        }
    }

    if (session->DefinesCmdLineArgument("kokkos_chunk_size"))
    {
        if (LoopExecution::s_kokkos_policy == Kokkos_Range_Policy ||
            LoopExecution::s_kokkos_policy == Kokkos_Team_Policy)
        {
            int cmdValue =
                session->GetCmdLineArgument<int>("kokkos_chunk_size");
            if (cmdValue > 0)
                s_kokkos_chunk_size = cmdValue;
            else
            {
                NEKERROR(Nektar::ErrorUtil::efatal,
                         "Bad command line argument for kokkos_chunk_size. "
                         "Value must be greater than zero.");
            }
        }
        else
        {
            NEKERROR(Nektar::ErrorUtil::efatal,
                     "Bad command line argument: kokkos_teams_chunk_size "
                     "set but not using the Kokkos Team or Range Policy");
        }
    }

    if (session->DefinesCmdLineArgument("kokkos_teams_tile_size"))
    {
        if (LoopExecution::s_kokkos_policy == Kokkos_Team_Policy)
        {
            int cmdValue = session->GetCmdLineArgument<int>("kokkos_tile_size");
            if (cmdValue > 0)
            {
                s_kokkos_tile_i_size = cmdValue;
                s_kokkos_tile_j_size = cmdValue;
                s_kokkos_tile_k_size = cmdValue;
            }
            else
            {
                NEKERROR(Nektar::ErrorUtil::efatal,
                         "Bad command line argument for kokkos_tile_size. "
                         "Value must be greater than zero.");
            }
        }
        else
        {
            NEKERROR(Nektar::ErrorUtil::efatal,
                     "Bad command line argument: kokkos_tile_size "
                     "set but not using the Kokkos MDRange Policy");
        }
    }

#endif // Kokkos specific defaults.
}

#if defined(NEKTAR_ENABLE_KOKKOS)
//_____________________________________________________________________________
//
void LoopExecution::setKokkosLeaguesPerLoop(unsigned int num)
{
    s_kokkos_leagues_per_loop = num;
}

//_____________________________________________________________________________
//
unsigned int LoopExecution::getKokkosLeaguesPerLoop()
{
    return s_kokkos_leagues_per_loop;
}

//_____________________________________________________________________________
//
void LoopExecution::setKokkosTeamsPerLeague(unsigned int num)
{
    s_kokkos_teams_per_league = num;
}

//_____________________________________________________________________________
//
unsigned int LoopExecution::getKokkosTeamsPerLeague()
{
    return s_kokkos_teams_per_league;
}

//_____________________________________________________________________________
//  Sets the Kokkos execution policy
void LoopExecution::setKokkosPolicy(LoopExecution::Kokkos_Policy policy)
{
    s_kokkos_policy = policy;
}

LoopExecution::Kokkos_Policy LoopExecution::getKokkosPolicy()
{
    return s_kokkos_policy;
}

//_____________________________________________________________________________
//  Sets/gets the Kokkos chuck size for Kokkos::TeamPolicy & Kokkos::RangePolicy
void LoopExecution::setKokkosChunkSize(int size)
{
    s_kokkos_chunk_size = size;
}

int LoopExecution::getKokkosChunkSize()
{
#if defined(KOKKOS_USING_GPU)
    if (s_kokkos_chunk_size < 0)
    {
        // Get the default chunk size using a dummy policy.
        if (s_kokkos_policy == LoopExecution::Kokkos_Team_Policy)
            s_kokkos_chunk_size =
                Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>(128, 512)
                    .chunk_size();
        else if (s_kokkos_policy == LoopExecution::Kokkos_Range_Policy)
            s_kokkos_chunk_size =
                Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace, int>(0, 512)
                    .chunk_size();
    }
#endif

    return s_kokkos_chunk_size;
}

//_____________________________________________________________________________
//  Sets/gets the Kokkos tile size for Kokkos::MDRangePolicy
void LoopExecution::setKokkosTileSize(int isize, int jsize, int ksize)
{
    s_kokkos_tile_i_size = isize;
    s_kokkos_tile_j_size = jsize;
    s_kokkos_tile_k_size = ksize;

    // Can not be used if Kokkos is not yet initialized.
#if defined(KOKKOS_USING_GPU)
    if (Kokkos::is_initialized())
    {
        // Use the Kokkos::MDRangePolicy default tile size.
        Kokkos::DefaultExecutionSpace execSpace;

        int max_threads =
            Kokkos::Impl::get_tile_size_properties(execSpace).max_threads;

        if (isize * jsize * ksize > max_threads)
        {
            std::stringstream msg;
            msg << "The product of tile dimensions (" << isize << "x" << jsize
                << "x" << ksize << ") " << isize * jsize * ksize << " "
                << "exceed maximum number of threads per block: "
                << max_threads;

            NEKERROR(Nektar::ErrorUtil::efatal,
                     "Kokkos ExecSpace Error: "
                     "MDRange tile dims exceed maximum number "
                     "of threads per block - reduce the tile dims. " +
                         msg.str());
        }
    }
#endif
}

void LoopExecution::getKokkosTileSize(int &isize, int &jsize, int &ksize)
{
#if defined(KOKKOS_USING_GPU)
    if (s_kokkos_tile_i_size < 0 || s_kokkos_tile_j_size < 0 ||
        s_kokkos_tile_k_size < 0)
    {
        // Use the Kokkos::MDRangePolicy default tile size.
        Kokkos::DefaultExecutionSpace execSpace;

        // Get the cube root of the max_threads
        int tileSize = std::cbrt(
            Kokkos::Impl::get_tile_size_properties(execSpace).max_threads);

        // Find the largest exponent so the tile size is a power of two.
        unsigned int exp = 0;
        while (tileSize >>= 1)
            exp++;

        if (exp > 1)
            tileSize = pow(2, exp - 1);
        else
            tileSize = 1;

        s_kokkos_tile_i_size = tileSize;
        s_kokkos_tile_j_size = tileSize;
        s_kokkos_tile_k_size = tileSize;
    }
#endif

    isize = s_kokkos_tile_i_size;
    jsize = s_kokkos_tile_j_size;
    ksize = s_kokkos_tile_k_size;
}

#endif // Kokkos specific defaults.

} // namespace Nektar
