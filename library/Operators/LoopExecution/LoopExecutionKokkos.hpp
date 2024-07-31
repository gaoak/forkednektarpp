///////////////////////////////////////////////////////////////////////////////
//
// File: LoopExecutionKokkos.hpp
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

#if defined(NEKTAR_ENABLE_KOKKOS)

#include <LibUtilities/BasicUtils/SessionReader.h>

#include <LibUtilities/BasicUtils/MiscUtils.hpp>

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
};

// Simple 1D Range parallel_for
template <typename ExecSpace, typename Functor>
inline typename std::enable_if<
    std::is_same<ExecSpace, NektarSpaces::KOKKOS>::value, void>::type
parallel_for(const int begin, const int end, const Functor &functor)
{
    std::string name = Nektar::demangleTypeName(typeid(Functor));

    Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace> rangePolicy(begin, end);

    Kokkos::parallel_for(name, rangePolicy, functor);
}

// Simple 1D Range parallel_reduce
template <typename ExecSpace, typename Reduction, typename Functor>
inline typename std::enable_if<
    std::is_same<ExecSpace, NektarSpaces::KOKKOS>::value, void>::type
parallel_reduce(const int begin, const int end, const Functor &functor,
                typename Reduction::value_type &red)
{
    std::string name = Nektar::demangleTypeName(typeid(Functor));

    Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace> rangePolicy(begin, end);

    Kokkos::parallel_reduce(name, rangePolicy, functor, Reduction(red));
}

// Block range parallel_for
template <typename ExecSpace, typename Functor>
inline typename std::enable_if<
    std::is_same<ExecSpace, NektarSpaces::KOKKOS>::value, void>::type
parallel_for(BlockRange const &r, const Functor &functor)
{
    std::string name = Nektar::demangleTypeName(typeid(Functor));

    const int i_size = r.end(0) - r.begin(0);
    const int j_size = r.end(1) - r.begin(1);
    const int k_size = r.end(2) - r.begin(2);

    const int rbegin0 = r.begin(0);
    const int rbegin1 = r.begin(1);
    const int rbegin2 = r.begin(2);

    const int rend0 = r.end(0);
    const int rend1 = r.end(1);
    const int rend2 = r.end(2);

    const unsigned int numItems =
        ((i_size > 0 ? i_size : 1) * (j_size > 0 ? j_size : 1) *
         (k_size > 0 ? k_size : 1));

    // Get from the session. Set up in LoopExecution.cpp
    LoopExecution::Kokkos_Policy kokkos_policy =
        LoopExecution::getKokkosPolicy();

    // Range Policy
    if (kokkos_policy == LoopExecution::Kokkos_Range_Policy)
    {
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace> rangePolicy(
            0, numItems);

        // Get from the session. Set up in LoopExecution.cpp
        int size = LoopExecution::getKokkosChunkSize();
        if (size > 0)
            rangePolicy.set_chunk_size(size);

        Kokkos::parallel_for(
            name, rangePolicy, KOKKOS_LAMBDA(int n) {
                const int k = n / (j_size * i_size) + rbegin2;
                const int j = (n / i_size) % j_size + rbegin1;
                const int i = n % i_size + rbegin0;

                functor(i, j, k);
            });
    }
    // MDRange Policy
    else if (kokkos_policy == LoopExecution::Kokkos_MDRange_Policy)
    {
        // Get from the session. Set up in LoopExecution.cpp
        int i_tile, j_tile, k_tile;
        LoopExecution::getKokkosTileSize(i_tile, j_tile, k_tile);

        if (i_tile > 0 || j_tile > 0 || k_tile > 0)
        {
            Kokkos::MDRangePolicy<Kokkos::DefaultExecutionSpace,
                                  Kokkos::Rank<3>, int>
                mdRangePolicy({rbegin0, rbegin1, rbegin2},
                              {rend0, rend1, rend2}, {i_tile, j_tile, k_tile});

            Kokkos::parallel_for(name, mdRangePolicy, functor);
        }
        else
        {
            Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>, int>
                mdRangePolicy({rbegin0, rbegin1, rbegin2},
                              {rend0, rend1, rend2}, {i_size, j_size, k_size});

            Kokkos::parallel_for(name, mdRangePolicy, functor);
        }
    }
    // Team Policy
    else if (kokkos_policy == LoopExecution::Kokkos_Team_Policy)
    {
        // The team implementation needs some work. The first is too
        // simple but works. The commented out version is more robust
        // (complicated but does not always work).

        // Team policy approach.
        const int teams_per_league = 1; // getKokkosTeamsPerLeague();
        const int team_range_size  = numItems;

        const int actualTeams = teams_per_league < team_range_size
                                    ? teams_per_league
                                    : team_range_size;

        // Assumption - there is only one league
        Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace> teamPolicy(
            1, actualTeams);
        typedef Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace> policy_type;

        // Get from the session. Set up in LoopExecution.cpp
        int size = LoopExecution::getKokkosChunkSize();
        if (size > 0)
            teamPolicy.set_chunk_size(size);

        Kokkos::parallel_for(
            name, teamPolicy,
            KOKKOS_LAMBDA(typename policy_type::member_type thread) {
                // printf("i is %d\n", thread.team_rank());
                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(thread, team_range_size),
                    [&](const int &n) {
                        const int i = n / (j_size * k_size) + rbegin0;
                        const int j = (n / k_size) % j_size + rbegin1;
                        const int k = n % k_size + rbegin2;

                        functor(i, j, k);
                    });
            });

#ifdef COMMENT_OUT
        // Overall goal, split a 3D range requested by the user into various
        // SMs on the GPU.  (In essence, this would be a Kokkos
        // MD_Team+Policy, if one existed) The process requires going from
        // 3D range to a 1D range, partitioning the 1D range into groups
        // that are multiples of 32, then converting that group of 32 range
        // back into a 3D (i,j,k) index.

        // The user has two partitions available.  1) One is the total
        // number of streaming multiprocessors.  2) The other is splitting a
        // task into multiple streams and execution units.

        // Get from the session. Set up in LoopExecution.cpp
        const int kokkos_leagues_per_loop =
            LoopExecution::getKokkosLeaguesPerLoop();
        const int kokkos_teams_per_league =
            LoopExecution::getKokkosTeamsPerLeague();

        // The requested range of data may not have enough work for the
        // requested command line arguments, so shrink them if necessary.
        int teams_per_loop  = kokkos_teams_per_league * kokkos_leagues_per_loop;
        int team_range_size = numItems;

        const unsigned int actual_teams =
            teams_per_loop < team_range_size ? teams_per_loop : team_range_size;

        const unsigned int actual_teams_per_league =
            kokkos_teams_per_league < team_range_size ? kokkos_teams_per_league
                                                      : team_range_size;

        const unsigned int actual_leagues_per_loop =
            (actual_teams - 1) / kokkos_teams_per_league + 1;

        // Use a Team Policy, this allows us to control how many threads
        // per league and how many leagues are used.
        Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace> teamPolicy(
            actual_leagues_per_loop, actual_teams_per_league);
        typedef Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace> policy_type;

        // Get from the session. Set up in LoopExecution.cpp
        int size = LoopExecution::getKokkosChunkSize();
        if (size > 0)
            teamPolicy.set_chunk_size(size);

        Kokkos::parallel_for(
            name, teamPolicy,
            KOKKOS_LAMBDA(typename policy_type::member_type thread) {
                // Within an SM, and all SMs share the same amount of
                // assigned Kokkos threads.  Figure out which range of
                // N items this SM should work on (as a multiple of 32).
                const unsigned int currentPartition = thread.league_rank();
                unsigned int estimatedThreadAmount =
                    numItems * currentPartition / actual_leagues_per_loop;
                const unsigned int startingN =
                    estimatedThreadAmount +
                    ((estimatedThreadAmount % 32 == 0)
                         ? 0
                         : (32 - estimatedThreadAmount % 32));
                unsigned int endingN;

                // Check if this is the last partition
                if (currentPartition + 1 == actual_leagues_per_loop)
                {
                    endingN = numItems;
                }
                else
                {
                    estimatedThreadAmount = numItems * (currentPartition + 1) /
                                            actual_leagues_per_loop;
                    endingN = estimatedThreadAmount +
                              ((estimatedThreadAmount % 32 == 0)
                                   ? 0
                                   : (32 - estimatedThreadAmount % 32));
                }

                const unsigned int totalN = endingN - startingN;
                // printf("league_rank: %d, team_size: %d, team_rank: %d,
                // startingN: %d, endingN: %d, totalN: %d\n",
                // thread.league_rank(), thread.team_size(), thread.team_rank(),
                // startingN, endingN, totalN);

                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(thread, totalN),
                    [&, startingN, i_size, j_size, k_size, rbegin0, rbegin1,
                     rbegin2](const int &N) {
                        // Craft an i,j,k out of this range.  This approach
                        // works with row-major layout so that consecutive
                        // Kokkos threads work along consecutive slots in
                        // memory.

                        // printf("parallel_for team demo - n is %d, league_rank
                        // is %d, true n is %d\n", N, thread.league_rank(),
                        // (startingN + N));

                        const int i = ((startingN + N) % i_size) + rbegin0;
                        const int j =
                            ((startingN + N) / i_size) % j_size + rbegin1;
                        const int k =
                            (startingN + N) / (i_size * j_size) + rbegin2;

                        functor(i, j, k);
                    });
            });
#endif
    }
}

// Block range parallel_reduce
template <typename ExecSpace, typename Reduction, typename Functor>
inline typename std::enable_if<
    std::is_same<ExecSpace, NektarSpaces::KOKKOS>::value, void>::type
parallel_reduce(BlockRange const &r, const Functor &functor,
                typename Reduction::value_type &red)
{
    std::string name = Nektar::demangleTypeName(typeid(Functor));

    const int i_size = r.end(0) - r.begin(0);
    const int j_size = r.end(1) - r.begin(1);
    const int k_size = r.end(2) - r.begin(2);

    const int rbegin0 = r.begin(0);
    const int rbegin1 = r.begin(1);
    const int rbegin2 = r.begin(2);

    const int rend0 = r.end(0);
    const int rend1 = r.end(1);
    const int rend2 = r.end(2);

    const unsigned int numItems =
        ((i_size > 0 ? i_size : 1) * (j_size > 0 ? j_size : 1) *
         (k_size > 0 ? k_size : 1));

    // Get from the session. Set up in LoopExecution.cpp
    LoopExecution::Kokkos_Policy kokkos_policy =
        LoopExecution::getKokkosPolicy();

    // Range Policy
    if (kokkos_policy == LoopExecution::Kokkos_Range_Policy)
    {
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace> rangePolicy(
            0, numItems);

        // Get from the session. Set up in LoopExecution.cpp
        int size = LoopExecution::getKokkosChunkSize();
        if (size > 0)
            rangePolicy.set_chunk_size(size);

        Kokkos::parallel_reduce(
            name, rangePolicy,
            KOKKOS_LAMBDA(int n, typename Reduction::value_type &tmp) {
                const int k = n / (j_size * i_size) + rbegin2;
                const int j = (n / i_size) % j_size + rbegin1;
                const int i = n % i_size + rbegin0;

                functor(i, j, k, tmp);
            },
            Reduction(red));
    }
    // MDRange Policy
    else if (kokkos_policy == LoopExecution::Kokkos_MDRange_Policy)
    {
        // Get from the session. Set up in LoopExecution.cpp
        int i_tile, j_tile, k_tile;
        LoopExecution::getKokkosTileSize(i_tile, j_tile, k_tile);

        if (i_tile > 0 || j_tile > 0 || k_tile > 0)
        {
            Kokkos::MDRangePolicy<Kokkos::DefaultExecutionSpace,
                                  Kokkos::Rank<3>, int>
                mdRangePolicy({rbegin0, rbegin1, rbegin2},
                              {rend0, rend1, rend2}, {i_tile, j_tile, k_tile});

            Kokkos::parallel_reduce(name, mdRangePolicy, functor,
                                    Reduction(red));
        }
        else
        {
            Kokkos::MDRangePolicy<Kokkos::DefaultExecutionSpace,
                                  Kokkos::Rank<3>, int>
                mdRangePolicy({rbegin0, rbegin1, rbegin2},
                              {rend0, rend1, rend2}, {i_size, j_size, k_size});

            Kokkos::parallel_reduce(name, mdRangePolicy, functor,
                                    Reduction(red));
        }
    }
    // Team Policy
    else if (kokkos_policy == LoopExecution::Kokkos_Team_Policy)
    {
        // The team implementation needs some work. The first is too
        // simple but works. The commented out version is more robust
        // (complicated but does not always work).

        // Team policy approach.
        const int teams_per_league = 1; // getKokkosTeamsPerLeague();
        const int team_range_size  = numItems;

        const int actualTeams = teams_per_league < team_range_size
                                    ? teams_per_league
                                    : team_range_size;

        // Assumption - there is only one league
        Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace> teamPolicy(
            1, actualTeams);
        typedef Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace> policy_type;

        // Get from the session. Set up in LoopExecution.cpp
        int size = LoopExecution::getKokkosChunkSize();
        if (size > 0)
            teamPolicy.set_chunk_size(size);

        Reduction reduction(red);

        Kokkos::parallel_reduce(
            name, teamPolicy,
            KOKKOS_LAMBDA(typename policy_type::member_type thread,
                          typename Reduction::value_type & inner_val) {
                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(thread, team_range_size),
                    [&](const int &n) {
                        const int i = n / (j_size * k_size) + rbegin0;
                        const int j = (n / k_size) % j_size + rbegin1;
                        const int k = n % k_size + rbegin2;

                        typename Reduction::value_type tmp;
                        reduction.init(tmp);

                        functor(i, j, k, tmp);

                        reduction.join(inner_val, tmp);
                    });
            },
            Reduction(red));

#ifdef COMMENT_OUT
        // Overall goal, split a 3D range requested by the user into various
        // SMs on the GPU.  (In essence, this would be a Kokkos
        // MD_Team+Policy, if one existed) The process requires going from
        // 3D range to a 1D range, partitioning the 1D range into groups
        // that are multiples of 32, then converting that group of 32 range
        // back into a 3D (i,j,k) index.

        // The user has two partitions available.  1) One is the total
        // number of streaming multiprocessors.  2) The other is splitting a
        // task into multiple streams and execution units.

        // Get from the session. Set up in LoopExecution.cpp
        const int kokkos_leagues_per_loop =
            LoopExecution::getKokkosLeaguesPerLoop();
        const int kokkos_teams_per_league =
            LoopExecution::getKokkosTeamsPerLeague();

        // The requested range of data may not have enough work for the
        // requested command line arguments, so shrink them if necessary.
        int teams_per_loop  = kokkos_teams_per_league * kokkos_leagues_per_loop;
        int team_range_size = numItems;

        const unsigned int actual_teams =
            teams_per_loop < team_range_size ? teams_per_loop : team_range_size;

        const unsigned int actual_teams_per_league =
            kokkos_teams_per_league < team_range_size ? kokkos_teams_per_league
                                                      : team_range_size;

        const unsigned int actual_leagues_per_loop =
            (actual_teams - 1) / kokkos_teams_per_league + 1;

        // Use a Team Policy, this allows us to control how many threads
        // per league and how many leagues are used.
        Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace> teamPolicy(
            actual_leagues_per_loop, actual_teams_per_league);

        typedef Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace> policy_type;

        // Get from the session. Set up in LoopExecution.cpp
        int size = LoopExecution::getKokkosChunkSize();
        if (size > 0)
            teamPolicy.set_chunk_size(size);

        Reduction reduction(red);

        Kokkos::parallel_reduce(
            name, teamPolicy,
            KOKKOS_LAMBDA(typename policy_type::member_type thread,
                          typename Reduction::value_type & inner_val) {
                // We are within an SM, and all SMs share the same amount of
                // assigned Kokkos threads.  Figure out which range of N items
                // this SM should work on (as a multiple of 32).
                const unsigned int currentPartition =
                    actual_leagues_per_loop + thread.league_rank();
                unsigned int estimatedThreadAmount =
                    numItems * currentPartition / actual_leagues_per_loop;
                const unsigned int startingN =
                    estimatedThreadAmount +
                    ((estimatedThreadAmount % 32 == 0)
                         ? 0
                         : (32 - estimatedThreadAmount % 32));
                unsigned int endingN;

                // Check if this is the last partition
                if (currentPartition + 1 == actual_leagues_per_loop)
                {
                    endingN = numItems;
                }
                else
                {
                    estimatedThreadAmount = numItems * (currentPartition + 1) /
                                            actual_leagues_per_loop;
                    endingN = estimatedThreadAmount +
                              ((estimatedThreadAmount % 32 == 0)
                                   ? 0
                                   : (32 - estimatedThreadAmount % 32));
                }

                const unsigned int totalN = endingN - startingN;
                // printf("league_rank: %d, team_size: %d, team_rank: %d,
                // startingN: %d, endingN: %d, totalN: %d\n",
                // thread.league_rank(), thread.team_size(), thread.team_rank(),
                // startingN, endingN, totalN);

                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(thread, totalN),
                    [&, startingN, i_size, j_size, k_size, rbegin0, rbegin1,
                     rbegin2](const int &N) {
                        // Craft an i,j,k out of this range.  This approach
                        // works with row-major layout so that consecutive
                        // Kokkos threads work along consecutive slots in
                        // memory.

                        // printf("parallel_reduce team demo - n is %d,
                        // league_rank is %d, true n is %d\n", N,
                        // thread.league_rank(), (startingN + N));

                        const int i = ((startingN + N) % i_size) + rbegin0;
                        const int j =
                            ((startingN + N) / i_size) % j_size + rbegin1;
                        const int k =
                            (startingN + N) / (j_size * i_size) + rbegin2;

                        typename Reduction::value_type tmp;
                        reduction.init(tmp);

                        functor(i, j, k, tmp);

                        reduction.join(inner_val, tmp);
                    });
            },
            Reduction(red));
#endif
    }
}

} // namespace Nektar

#endif
