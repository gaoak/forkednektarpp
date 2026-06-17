//////////////////////////////////////////////////////////////////////////////
//
// File: ProfilerInterleaving.hpp
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
// Description: main content of profilers
//
///////////////////////////////////////////////////////////////////////////////

#include <cstdio>
#include <iomanip>
#include <iostream>

#include <Operators/Common/Operator.hpp>
#include <Operators/Field/Field.hpp>

#include <LibUtilities/BasicUtils/ErrorUtil.hpp>
#include <LibUtilities/BasicUtils/Timer.h>
#include <MultiRegions/ExpList.h>
#include <SpatialDomains/MeshGraphIO.h>

// Add likwid support
#ifdef LIKWID_PERFMON
#include <likwid.h>
#else
#define LIKWID_MARKER_INIT
#define LIKWID_MARKER_THREADINIT
#define LIKWID_MARKER_SWITCH
#define LIKWID_MARKER_REGISTER(regionTag)
#define LIKWID_MARKER_START(regionTag)
#define LIKWID_MARKER_STOP(regionTag)
#define LIKWID_MARKER_CLOSE
#define LIKWID_MARKER_GET(regionTag, nevents, events, time, count)
#endif

using namespace Nektar;
using namespace Nektar::Operators;
using namespace Nektar::LibUtilities;

/// Print the block information. If _verbose_=true, then print the block
/// information for each rank. If _verbose_=false, then only print the
/// total information for each rank. Caution: for many ranks and many
/// blocks, setting verbose may cause the display content too big to read.
template <FieldState TState>
void PrintBlockInfo(const MultiRegions::ExpListSharedPtr &expList,
                    const std::vector<BlockAttributes<TState>> &blockAttr)
{
    auto comm                 = expList->GetComm();
    unsigned int nrank        = comm->GetSize();
    auto rankBlockNum         = std::vector<unsigned int>(1, blockAttr.size());
    auto rankBlockNumDofs     = std::vector<size_t>(blockAttr.size(), 0);
    auto rankBlockNumData     = std::vector<unsigned int>(blockAttr.size(), 0);
    auto rankBlockNumElmts    = std::vector<size_t>(blockAttr.size(), 0);
    auto rankBlockNumPaddings = std::vector<size_t>(blockAttr.size(), 0);
    auto rankNumDofs          = std::vector<size_t>(1, 0);
    auto rankNumElmts         = std::vector<size_t>(1, 0);
    auto rankNumPaddings      = std::vector<size_t>(1, 0);
    auto rankGeomTypes        = std::vector<unsigned int>(1, 0);

    // Collect total information for each rank.
    for (size_t i = 0, expId = 0; i < blockAttr.size(); ++i)
    {
        rankBlockNumDofs[i]     = blockAttr[i].CompSize();
        rankBlockNumData[i]     = blockAttr[i].GetNumData();
        rankBlockNumElmts[i]    = blockAttr[i].GetNumElements();
        rankBlockNumPaddings[i] = blockAttr[i].GetNumElementsWithPadding() -
                                  blockAttr[i].GetNumElements();
        rankNumDofs[0] += blockAttr[i].CompSize();
        rankNumElmts[0] += blockAttr[i].GetNumElements();
        rankNumPaddings[0] += blockAttr[i].GetNumElementsWithPadding() -
                              blockAttr[i].GetNumElements();

        // Check the geometry type of the block: deformed or regular
        // if both types exist in the same rank, then it is labeled as mixed.
        auto gtype = expList->GetExp(expId)->GetGeomFactors()->GetGtype();
        if (gtype == SpatialDomains::eDeformed && rankGeomTypes[0] != 1)
        {
            rankGeomTypes[0] = 2; // Deformed
        }
        else if (gtype == SpatialDomains::eRegular && rankGeomTypes[0] != 2)
        {
            rankGeomTypes[0] = 1; // Regular
        }
        else
        {
            rankGeomTypes[0] = 3; // Mixed
        }
    }
    auto allRankBlockNum         = comm->Gather(0, rankBlockNum);
    auto allRankBlockNumDofs     = comm->Gather(0, rankBlockNumDofs);
    auto allRankBlockNumData     = comm->Gather(0, rankBlockNumData);
    auto allRankBlockNumElmts    = comm->Gather(0, rankBlockNumElmts);
    auto allRankBlockNumPaddings = comm->Gather(0, rankBlockNumPaddings);
    auto allRankNumDofs          = comm->Gather(0, rankNumDofs);
    auto allRankNumElmts         = comm->Gather(0, rankNumElmts);
    auto allRankNumPaddings      = comm->Gather(0, rankNumPaddings);
    auto allRankGeomTypes        = comm->Gather(0, rankGeomTypes);

    // Print summary information.
    if (comm->GetRank() == 0)
    {
        // Get the geometry type of the whole domain :
        // Calculate the average value, if average is 1, then it is regular
        // if average is 2, then it is deformed, otherwise it is mixed/
        auto GeomType = Vmath::Vsum(nrank, allRankGeomTypes.data(), 1) / nrank;

        std::string Gtype = "Null";
        if (abs(GeomType - 1.0) < 1e-8)
        {
            Gtype = "Regular";
        }
        else if (abs(GeomType - 2.0) < 1e-8)
        {
            Gtype = "Deformed";
        }
        else
        {
            Gtype = "Mixed";
        }

        // Check if verbose is set.
        auto verbose = expList->GetSession()->DefinesCmdLineArgument("verbose");

        if (verbose)
        {
            for (unsigned int rank = 0, cnt = 0; rank < nrank; rank++)
            {
                // Print block information in a table like format.
                // | BlockId | #Elements | #Paddings | #Points | BlockSize |
                std::cout << std::setw(5) << "Rank#" << std::setw(5) << rank
                          << std::endl;
                std::cout << std::endl;
                std::cout << std::setw(10) << "BlockId" << std::setw(12)
                          << "#Elements" << std::setw(12) << "#Paddings"
                          << std::setw(10) << "#Points" << std::setw(12)
                          << "BlockSize" << std::endl;
                // Print a dash line.
                std::cout << std::setw(10) << std::setfill('-') << ""
                          << std::setw(12) << "" << std::setw(12) << ""
                          << std::setw(10) << "" << std::setw(12) << ""
                          << std::setfill(' ') << std::endl;

                for (unsigned int i = 0; i < allRankBlockNum[rank]; ++i, ++cnt)
                {
                    std::cout << std::setw(10) << i << std::setw(12)
                              << allRankBlockNumElmts[cnt] << std::setw(12)
                              << allRankBlockNumPaddings[cnt] << std::setw(12)
                              << std::setw(10) << allRankBlockNumData[cnt]
                              << std::setw(12) << allRankBlockNumDofs[cnt]
                              << std::endl;
                }

                // Print a dash line.
                std::cout << std::setw(10) << std::setfill('-') << ""
                          << std::setw(12) << "" << std::setw(12) << ""
                          << std::setw(10) << "" << std::setw(12) << ""
                          << std::setfill(' ') << std::endl;
                // Final line is sum of all blocks.
                std::cout << std::setw(5) << "Total" << std::setw(5)
                          << allRankBlockNum[rank] << std::setw(12)
                          << allRankNumElmts[rank] << std::setw(12)
                          << allRankNumPaddings[rank] << std::setw(10) << "N/A"
                          << std::setw(12) << allRankNumDofs[rank]
                          << std::setw(16) << std::endl;
                std::cout << std::endl;
            }
        }

        std::cout << std::setw(10) << "Rank# " << std::setw(12) << "#Elements"
                  << std::setw(12) << "#Paddings" << std::setw(12) << "#DoFs"
                  << std::setw(12) << "Gtype" << std::setw(16) << "L1 error"
                  << std::endl;
        for (unsigned int rank = 0; rank < nrank; rank++)
        {
            std::cout << std::setw(10) << rank << std::setw(12)
                      << allRankNumElmts[rank] << std::setw(12)
                      << allRankNumPaddings[rank] << std::setw(12)
                      << allRankNumDofs[rank] << std::setw(12) << Gtype
                      << std::setw(16) << std::endl;
        }
    }
    comm->Block();
}

/// Print the profiler results, computed from the elapsed time and the total
/// number of dofs.
template <typename TData, FieldState TStateIn, FieldState TStateOut>
void PrintProfileResult(
    const CommSharedPtr comm, std::vector<double> &rankElapsed,
    const std::vector<BlockAttributes<TStateIn>> &inblockAttr,
    const std::vector<BlockAttributes<TStateOut>> &outblockAttr)
{
    // Collect elapsed time and compute the max, min, and average.
    unsigned int nrank  = comm->GetSize();
    auto rankNumInDofs  = std::vector<size_t>(1, 0);
    auto rankNumOutDofs = std::vector<size_t>(1, 0);

    // Collect total information for each rank.
    for (unsigned int i = 0; i < inblockAttr.size(); ++i)
    {
        rankNumInDofs[0] += inblockAttr[i].CompSize();
    }
    for (unsigned int i = 0; i < outblockAttr.size(); ++i)
    {
        rankNumOutDofs[0] += outblockAttr[i].CompSize();
    }

    auto allRankElapsed    = comm->Gather(0, rankElapsed);
    auto allRankNumInDofs  = comm->Gather(0, rankNumInDofs);
    auto allRankNumOutDofs = comm->Gather(0, rankNumOutDofs);

    if (comm->GetRank() == 0)
    {
        double maxElapsed = Vmath::Vmax(nrank, allRankElapsed.data(), 1);
        double minElapsed = Vmath::Vmin(nrank, allRankElapsed.data(), 1);
        double aveElapsed =
            Vmath::Vsum(nrank, allRankElapsed.data(), 1) / nrank;

        // Collect throughput for each rank:
        // The throughput for each rank is calucated by the total number of dofs
        // and the elapsed time in that rank. The total throughput is the sum of
        // all rank. So the total throughput by this way will not be identical
        // to total dofs divided by the total (min/ave/max) elapsed time.
        double inThroughput  = 0.0;
        double outThroughput = 0.0;
        size_t totInDofs     = 0;
        size_t totOutDofs    = 0;
        for (unsigned int rank = 0; rank < nrank; rank++)
        {
            inThroughput += allRankNumInDofs[rank] / allRankElapsed[rank];
            outThroughput += allRankNumOutDofs[rank] / allRankElapsed[rank];
            totInDofs += allRankNumInDofs[rank];
            totOutDofs += allRankNumOutDofs[rank];
        }

        // Print the summary :
        // Max/Min/Aver time can be used to judge the load-balance in parallel
        std::cout << "Max time per test (s): " << maxElapsed << std::endl;
        std::cout << "Min time per test (s): " << minElapsed << std::endl;
        std::cout << "Average time per test (s): " << aveElapsed << std::endl;
        std::cout << "Total ndof (input/output): " << totInDofs << " "
                  << totOutDofs << std::endl;
        std::cout << "Throughput (ndof/s): " << inThroughput << " "
                  << outThroughput << std::endl;
        std::cout << "Throughput (GB/s): " << inThroughput * sizeof(TData) / 1e9
                  << " " << outThroughput * sizeof(TData) / 1e9 << std::endl;
        std::cout << std::endl;
        std::cout << std::endl;
    }
    comm->Block();
}

template <FieldState TState, typename TData>
void LaunchProfiler(MultiRegions::ExpListSharedPtr &expList,
                    const unsigned int Ntest, const unsigned int nIn = 1,
                    const unsigned int nComp = 1, const unsigned int nHomo = 1)
{
    // Timer.
    Timer timer;

    auto session = expList->GetSession();

    // Get communicator.
    auto comm = expList->GetComm();

    std::string execName = Operator<TData>::GetOpExecSpace(session);
    std::string dataType = (std::is_same_v<TData, double>) ? "Double" : "Float";
    auto tag             = "Interleave" + execName + dataType;

    // Create block attributes.
    auto blockAttr = GetBlockAttributes<TData, TState>(expList);

    // Create fields.
    auto inout = Field<TData, TState>("f_in", blockAttr, nIn * nComp, nHomo);

    // Initialize the in field to random non-zeros: 1 2 3 4 ...
    for (size_t i = 0; i < inout.GetBlocks().size(); ++i)
    {
        auto inptr = inout.GetBlocks()[i]
                         .template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
        for (unsigned int n = 0; n < nIn * nComp; n++)
        {
            for (size_t j = 0; j < inout.GetBlocks()[i].CompSize(); ++j)
            {
                inptr[j] = (j + (n + 1.0)) / inout.GetBlocks()[i].CompSize();
            }
            inptr += inout.GetBlocks()[i].CompSize();
        }
    }

    // Warm-up : fill the cache and memory, and let core temperature/freq
    // stabilized.
    for (unsigned int i = 0; i < Ntest / 2; ++i)
    {
        inout.ReshapeStorage(NektarSpaces::GetVectorWidth<TData>(execName),
                             execName);
        inout.ReshapeStorage(1, execName);
    }

    comm->Block();

    // Benchmark
    nekDeviceSynchronize();
    timer.Start();
    LIKWID_MARKER_START(tag.c_str());

    for (unsigned int i = 0; i < Ntest; ++i)
    {
        inout.ReshapeStorage(NektarSpaces::GetVectorWidth<TData>(execName),
                             execName);
        inout.ReshapeStorage(1, execName);
    }

    nekDeviceSynchronize();
    LIKWID_MARKER_STOP(tag.c_str());
    timer.Stop();

    comm->Block();

    // Print block information and get the total number of dofs.
    if (comm->GetRank() == 0)
    {
        std::cout << "Input field: " << nComp * nIn << " components"
                  << std::endl;
    }
    PrintBlockInfo(expList, blockAttr);

    // Print block information and get the total number of dofs.
    if (comm->GetRank() == 0)
    {
        std::cout << "Output field: " << nComp * nIn << " components"
                  << std::endl;
    }
    PrintBlockInfo(expList, blockAttr);

    if (comm->GetRank() == 0)
    {
        std::cout << "---------------------------------" << std::endl;
        std::cout << "Interleaving Profiler : " << tag << std::endl;
        std::cout << "---------------------------------" << std::endl;
    }
    comm->Block();

    // Collect elapsed time and compute the max, min, and average.
    auto rankElapsed = std::vector<double>(1, timer.TimePerTest(Ntest));
    PrintProfileResult<TData>(comm, rankElapsed, blockAttr, blockAttr);
}
