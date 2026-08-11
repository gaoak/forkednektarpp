//////////////////////////////////////////////////////////////////////////////
//
// File: ProfilerBP.hpp
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

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "Operators/AssmbScatr/AssmbScatrOp.hpp"
#include "Operators/GlobalLinSysOps/LinearSolvers/ConjGrad/ConjGradOp.hpp"
#include "Operators/PreconOps/DiagPrecon/DiagPreconOp.hpp"
#include <Operators/ElmtOps/Helmholtz/HelmholtzOp.hpp>
#include <Operators/ElmtOps/Mass/MassOp.hpp>

#include <MultiRegions/Field/Field.hpp>

#include <LibUtilities/BasicUtils/ErrorUtil.hpp>
#include <LibUtilities/BasicUtils/Timer.h>
#include <MultiRegions/ContField.h>
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
using namespace Nektar::LibUtilities;
using namespace Nektar::MultiRegions;
using namespace Nektar::Operators;

/// Print the profiler results, computed from the elapsed time and the total
/// number of dofs.
template <typename TData, FieldState TStateIn, FieldState TStateOut>
void PrintProfileResult(
    const CommSharedPtr comm, const MultiRegions::ContFieldSharedPtr &expList,
    std::vector<double> &rankElapsed,
    const std::vector<BlockAttributes<TStateIn>> &inBlockAttr,
    const std::vector<BlockAttributes<TStateOut>> &outBlockAttr,
    const unsigned int nTests, const unsigned int totalIterations,
    const unsigned int numComponents)
{
    // Collect elapsed time and compute the max, min, and average.
    unsigned int nrank       = comm->GetSize();
    auto rankNumInDofs       = std::vector<size_t>(1, 0);
    auto rankNumOutDofs      = std::vector<size_t>(1, 0);
    auto rankTotalIterations = std::vector<unsigned int>(1, totalIterations);

    auto locToGloMap  = expList->GetLocalToGlobalMap();
    auto uniqueMap    = locToGloMap->GetGlobalToUniversalMapUnique();
    size_t globalDOFs = 0;
    for (int i = 0; i < uniqueMap.size(); ++i)
    {
        if (uniqueMap[i] > 0)
        {
            globalDOFs++;
        }
    }
    globalDOFs *= numComponents;
    comm->AllReduce(globalDOFs, Nektar::LibUtilities::ReduceSum);
    auto rankGlobalDOFs = std::vector<size_t>(1, globalDOFs);

    // Collect total information for each rank.
    for (unsigned int i = 0; i < inBlockAttr.size(); ++i)
    {
        rankNumInDofs[0] += inBlockAttr[i].CompSize() * numComponents;
    }
    for (unsigned int i = 0; i < outBlockAttr.size(); ++i)
    {
        rankNumOutDofs[0] += outBlockAttr[i].CompSize() * numComponents;
    }

    auto allRankElapsed     = comm->Gather(0, rankElapsed);
    auto allRankNumInDofs   = comm->Gather(0, rankNumInDofs);
    auto allRankNumOutDofs  = comm->Gather(0, rankNumOutDofs);
    auto allTotalIterations = comm->Gather(0, rankTotalIterations);

    if (comm->GetRank() == 0)
    {
        double maxElapsed = Vmath::Vmax(nrank, allRankElapsed.data(), 1);
        double minElapsed = Vmath::Vmin(nrank, allRankElapsed.data(), 1);
        double aveElapsed =
            Vmath::Vsum(nrank, allRankElapsed.data(), 1) / nrank;
        unsigned int globalTotalIterations =
            Vmath::Vmax(nrank, allTotalIterations.data(), 1);

        size_t totInDofs        = 0;
        size_t totOutDofs       = 0;
        double inThroughput     = 0.0;
        double outThroughput    = 0.0;
        double globalThroughput = 0.0;
        for (unsigned int rank = 0; rank < nrank; rank++)
        {
            totInDofs += allRankNumInDofs[rank];
            totOutDofs += allRankNumOutDofs[rank];
            inThroughput += allRankNumInDofs[rank] * allTotalIterations[rank];
            outThroughput += allRankNumOutDofs[rank] * allTotalIterations[rank];
        }

        // The throughput is calculated as the total number of dofs divided by
        // the time per iteration. The time per iteration is the maximum elapsed
        // time per solve divided by the average number of iterations per solve.
        // Formula: (totInDofs * globalTotalIterations) / (maxElapsedPerSolve *
        // Ntest)
        inThroughput     = (double)inThroughput / (maxElapsed * nTests);
        outThroughput    = (double)outThroughput / (maxElapsed * nTests);
        globalThroughput = (double)(rankGlobalDOFs[0] * globalTotalIterations) /
                           (maxElapsed * nTests);

        // Print the summary :
        std::cout << "Max time per test (s): " << maxElapsed << std::endl;
        std::cout << "Min time per test (s): " << minElapsed << std::endl;
        std::cout << "Average time per test (s): " << aveElapsed << std::endl;
        std::cout << "Total ndof (input/output): " << totInDofs << " "
                  << totOutDofs << std::endl;
        std::cout << "Global DOFs: " << globalDOFs << std::endl;
        std::cout << "Throughput (ndof/s): " << inThroughput << " "
                  << outThroughput << std::endl;
        std::cout << "Global Throughput (ndof/s): " << globalThroughput
                  << std::endl;
        std::cout << "Throughput (GB/s): " << inThroughput * sizeof(TData) / 1e9
                  << " " << outThroughput * sizeof(TData) / 1e9 << std::endl;
        std::cout << "Global Throughput (GB/s): "
                  << globalThroughput * sizeof(TData) / 1e9 << std::endl;
        std::cout << std::endl;
        std::cout << std::endl;
    }
    comm->Block();
}

template <typename TData, FieldState TState>
void FillFieldWithRandomValues(Field<TData, TState> &field)
{
    srand(0);

    auto &blocks           = field.GetBlocks();
    const auto numStorages = field.GetNumComponents() * field.GetNumHomoModes();
    for (size_t i = 0; i < blocks.size(); ++i)
    {
        auto ptr =
            blocks[i].template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
        const auto compSize = blocks[i].CompSize();
        for (unsigned int storage = 0; storage < numStorages; ++storage)
        {
            auto storagePtr = ptr + storage * compSize;
            for (size_t j = 0; j < compSize; ++j)
            {
                storagePtr[j] =
                    ((static_cast<double>(rand()) / RAND_MAX) * 2.0 - 1.0);
            }
        }
    }
}

template <typename TData, FieldState TState>
void PrintFieldDifference(const CommSharedPtr &comm,
                          Field<TData, TState> &field,
                          Field<TData, TState> &reference)
{
    auto &fieldBlocks     = field.GetBlocks();
    auto &referenceBlocks = reference.GetBlocks();
    ASSERTL0(fieldBlocks.size() == referenceBlocks.size(),
             "ProfilerBP verification requires matching block counts.");
    ASSERTL0(field.GetNumComponents() == reference.GetNumComponents(),
             "ProfilerBP verification requires matching component counts.");
    ASSERTL0(field.GetNumHomoModes() == reference.GetNumHomoModes(),
             "ProfilerBP verification requires matching homogeneous modes.");
    const auto numStorages = field.GetNumComponents() * field.GetNumHomoModes();
    double totalDiff       = 0.0;
    double totalMaxDiff    = 0.0;
    size_t count           = 0;

    for (size_t i = 0; i < fieldBlocks.size(); ++i)
    {
        auto fieldPtr =
            fieldBlocks[i].template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
        auto referencePtr =
            referenceBlocks[i]
                .template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
        const auto compSize = fieldBlocks[i].CompSize();

        for (unsigned int storage = 0; storage < numStorages; ++storage)
        {
            auto fieldStoragePtr     = fieldPtr + storage * compSize;
            auto referenceStoragePtr = referencePtr + storage * compSize;
            for (size_t j = 0; j < compSize; ++j)
            {
                double diff = std::abs((double)fieldStoragePtr[j] -
                                       (double)referenceStoragePtr[j]);
                totalDiff += diff;
                totalMaxDiff = std::max(totalMaxDiff, diff);
                count++;
            }
        }
    }

    std::vector<double> allDiff(1, totalDiff);
    std::vector<double> allMaxDiff(1, totalMaxDiff);
    std::vector<size_t> allCount(1, count);

    auto gatheredDiff    = comm->Gather(0, allDiff);
    auto gatheredMaxDiff = comm->Gather(0, allMaxDiff);
    auto gatheredCount   = comm->Gather(0, allCount);

    if (comm->GetRank() == 0)
    {
        double globalDiff =
            Vmath::Vsum(comm->GetSize(), gatheredDiff.data(), 1);
        double globalMaxDiff =
            Vmath::Vmax(comm->GetSize(), gatheredMaxDiff.data(), 1);
        size_t globalCount = 0;
        for (auto c : gatheredCount)
        {
            globalCount += c;
        }

        std::cout << "L1 error: " << globalDiff / globalCount << std::endl;
        std::cout << "Linf error: " << globalMaxDiff << std::endl;
    }
}

template <FieldState TStateIn, FieldState TStateOut, typename TData>
void LaunchProfiler(const MultiRegions::ContFieldSharedPtr &expList,

                    const unsigned int nTests, const int bp)
{
    // Timer.
    Timer timer;

    auto session = expList->GetSession();
    auto comm    = expList->GetComm();

    // Initialize operators.
    std::shared_ptr<ElmtOp<FieldState::Coeff, FieldState::Coeff, TData>> elmtOp;
    std::string opName;
    if (bp == 1)
    {
        elmtOp = MassOp<TData>::Create(expList, session->GetVariables());
        opName = "Mass";
    }
    else if (bp == 3)
    {
        std::vector<double> diffCoeff(6);
        diffCoeff[0] = 1.0; // D00
        diffCoeff[2] = 1.0; // D11
        diffCoeff[5] = 1.0; // D22
        elmtOp = HelmholtzOp<TData>::Create(expList, session->GetVariables());
        std::dynamic_pointer_cast<HelmholtzOp<TData>>(elmtOp)->SetDiffCoeff(
            diffCoeff);
        opName = "Helmholtz";
    }
    else
    {
        NEKERROR(ErrorUtil::efatal,
                 "ProfilerBP only supports BP=1 (Mass) and BP=3 (Helmholtz).");
    }

    auto assembOp =
        AssmbScatrOp<TData>::Create(expList, session->GetVariables());
    auto diagPreconOp =
        DiagPreconOp<TData>::Create(expList, session->GetVariables());
    auto conjGradOp =
        ConjGradOp<TData>::Create(expList, session->GetVariables());
    conjGradOp->SetLHS(elmtOp);
    conjGradOp->SetPrecon(diagPreconOp);
    conjGradOp->UpdatePrecon();

    // Set operator name tag.
    std::string execName = Operator<TData>::GetOpExecSpace(session);
    std::string implName =
        ElmtOp<FieldState::Coeff, FieldState::Coeff, TData>::GetOpImpl(
            opName, execName, session);
    std::string dataType = (std::is_same_v<TData, double>) ? "Double" : "Float";
    auto tag = "BP" + std::to_string(bp) + execName + implName + dataType;

    // Create block attributes.
    auto inBlockAttr  = GetBlockAttributes<TData, TStateIn>(expList);
    auto outBlockAttr = GetBlockAttributes<TData, TStateOut>(expList);

    // Create fields.
    auto fIn         = Field<TData, FieldState::Coeff>("f_in", inBlockAttr,
                                                       session->GetVariables(), 1);
    auto fOut        = Field<TData, FieldState::Coeff>("f_out", outBlockAttr,
                                                       session->GetVariables(), 1);
    auto fOutCorrect = Field<TData, FieldState::Coeff>(
        "f_out_correct", outBlockAttr, session->GetVariables(), 1);
    auto fOutCorrectAssemb = Field<TData, FieldState::Coeff>(
        "f_out_correct_assemb", outBlockAttr, session->GetVariables(), 1);

    // Set random output.
    FillFieldWithRandomValues(fOutCorrect);

    // Ensure C0 continuity.
    assembOp->Apply(fOutCorrect, fOutCorrectAssemb);

    // Compute expected solution.
    elmtOp->Apply(fOutCorrectAssemb, fIn);

    // Warm up solves.
    for (unsigned int i = 0; i < nTests / 2; ++i)
    {
        conjGradOp->Apply(fIn, fOut);
    }
    comm->Block();

    // Benchmark CG solve.
    nekDeviceSynchronize();
    timer.Start();
    LIKWID_MARKER_START(tag.c_str());

    unsigned int totalIterations = 0;
    for (unsigned int i = 0; i < nTests; ++i)
    {
        conjGradOp->Apply(fIn, fOut);
        totalIterations += conjGradOp->GetNiterations();
    }

    nekDeviceSynchronize();
    LIKWID_MARKER_STOP(tag.c_str());
    timer.Stop();

    comm->Block();

    // Verification
    PrintFieldDifference(comm, fOut, fOutCorrectAssemb);

    if (comm->GetRank() == 0)
    {
        std::cout << "---------------------------------" << std::endl;
        std::cout << "ProfilerBP : " << tag << std::endl;
        std::cout << "Iterations : " << totalIterations << std::endl;
        std::cout << "---------------------------------" << std::endl;
    }

    // Collect elapsed time per test (average duration of one solve).
    auto rankElapsed = std::vector<double>(1, timer.TimePerTest(nTests));

    PrintProfileResult<TData>(comm, expList, rankElapsed, inBlockAttr,
                              outBlockAttr, nTests, totalIterations,
                              fIn.GetNumComponents());
}
