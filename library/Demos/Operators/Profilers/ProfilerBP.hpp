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

#include <cstdio>
#include <iostream>

#include "Operators/AssmbScatr/AssmbScatrOp.hpp"
#include "Operators/GlobalLinSysOps/LinearSolvers/ConjGrad/ConjGradOp.hpp"
#include "Operators/PreconOps/DiagPrecon/DiagPreconOp.hpp"
#include <Operators/ElmtOps/Helmholtz/HelmholtzOp.hpp>
#include <Operators/ElmtOps/Mass/MassOp.hpp>

#include <Operators/Field/Field.hpp>
#include <Operators/LoopExecution/LoopExecution.hpp>

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
using namespace Nektar::Operators;
using namespace Nektar::LibUtilities;

/// Print the profiler results, computed from the elapsed time and the total
/// number of dofs.
template <typename TData, FieldState TStateIn, FieldState TStateOut>
void PrintProfileResult(
    const CommSharedPtr comm, const MultiRegions::ContFieldSharedPtr &expList,
    std::vector<double> &rankElapsed,
    const std::vector<BlockAttributes<TStateIn>> &inblockAttr,
    const std::vector<BlockAttributes<TStateOut>> &outblockAttr,
    const unsigned int Ntest, const unsigned int totalIterations)
{
    // Collect elapsed time and compute the max, min, and average.
    unsigned int nrank       = comm->GetSize();
    auto rankNumInDofs       = std::vector<size_t>(1, 0);
    auto rankNumOutDofs      = std::vector<size_t>(1, 0);
    auto rankTotalIterations = std::vector<unsigned int>(1, totalIterations);

    size_t globalDOFs   = expList->GetLocalToGlobalMap()->GetNumGlobalCoeffs();
    auto rankGlobalDOFs = std::vector<size_t>(1, globalDOFs);
    // Collect total information for each rank.
    for (unsigned int i = 0; i < inblockAttr.size(); ++i)
    {
        rankNumInDofs[0] += inblockAttr[i].CompSize();
    }
    for (unsigned int i = 0; i < outblockAttr.size(); ++i)
    {
        rankNumOutDofs[0] += outblockAttr[i].CompSize();
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

        size_t totInDofs        = 0;
        size_t totOutDofs       = 0;
        size_t totGlobalDOFs    = 0;
        double inThroughput     = 0.0;
        double outThroughput    = 0.0;
        double globalThroughput = 0.0;
        for (unsigned int rank = 0; rank < nrank; rank++)
        {
            totInDofs += allRankNumInDofs[rank];
            totOutDofs += allRankNumOutDofs[rank];
            totGlobalDOFs += rankGlobalDOFs[rank];
            inThroughput += allRankNumInDofs[rank] * allTotalIterations[rank];
            outThroughput += allRankNumOutDofs[rank] * allTotalIterations[rank];
            globalThroughput += rankGlobalDOFs[rank] * allTotalIterations[rank];
        }

        // The throughput is calculated as the total number of dofs divided by
        // the time per iteration. The time per iteration is the maximum elapsed
        // time per solve divided by the average number of iterations per solve.
        // Formula: (totInDofs * globalTotalIterations) / (maxElapsedPerSolve *
        // Ntest * nrank)
        inThroughput     = (double)inThroughput / (maxElapsed * Ntest);
        outThroughput    = (double)outThroughput / (maxElapsed * Ntest);
        globalThroughput = (double)globalThroughput / (maxElapsed * Ntest);

        // Print the summary :
        std::cout << "Max time per test (s): " << maxElapsed << std::endl;
        std::cout << "Min time per test (s): " << minElapsed << std::endl;
        std::cout << "Average time per test (s): " << aveElapsed << std::endl;
        std::cout << "Total ndof (input/output): " << totInDofs << " "
                  << totOutDofs << std::endl;
        std::cout << "Global DOFs: " << totGlobalDOFs << std::endl;
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

template <typename TData>
void LaunchProfiler(const MultiRegions::ContFieldSharedPtr &expList,
                    const unsigned int Ntest, const int BP)
{
    // Timer.
    Timer timer;

    auto session = expList->GetSession();
    auto comm    = expList->GetComm();

    // Initialize operators.
    std::shared_ptr<ElmtOp<FieldState::Coeff, FieldState::Coeff, TData>> elmtOp;
    std::string opName;
    if (BP == 1)
    {
        elmtOp = MassOp<TData>::Create(expList, session->GetVariables());
        opName = "Mass";
    }
    else if (BP == 3)
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

    auto assembOp =
        AssmbScatrOp<TData>::Create(expList, session->GetVariables());
    auto diagPreconOp =
        DiagPreconOp<TData>::Create(expList, session->GetVariables());
    auto conjGradOp =
        ConjGradOp<TData>::Create(expList, session->GetVariables());
    diagPreconOp->Configure(elmtOp);
    conjGradOp->SetLHS(elmtOp);
    conjGradOp->SetPrecon(diagPreconOp);

    // Set operator name tag.
    std::string execName = Operator<TData>::GetOpExecSpace(session);
    std::string implName =
        ElmtOp<FieldState::Coeff, FieldState::Coeff, TData>::GetOpImpl(
            opName, execName, session);
    std::string dataType = (std::is_same_v<TData, double>) ? "Double" : "Float";
    auto tag = "BP" + std::to_string(BP) + execName + implName + dataType;

    // Create block attributes.
    auto blockAttr = GetBlockAttributes<TData, FieldState::Coeff>(expList);

    // Create fields.
    auto fIn         = Field<TData, FieldState::Coeff>("f_in", blockAttr,
                                               session->GetVariables(), 1);
    auto fOut        = Field<TData, FieldState::Coeff>("f_out", blockAttr,
                                                session->GetVariables(), 1);
    auto fOutCorrect = Field<TData, FieldState::Coeff>(
        "f_out_correct", blockAttr, session->GetVariables(), 1);
    auto fOutCorrectAssemb = Field<TData, FieldState::Coeff>(
        "f_out_correct_assemb", blockAttr, session->GetVariables(), 1);

    // Set random output.
    srand(0);
    auto &blockOut = fOutCorrect.GetBlocks();
    for (size_t i = 0; i < blockOut.size(); ++i)
    {
        auto outPtr =
            blockOut[i].template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
        for (unsigned int n = 0; n < session->GetVariables().size(); n++)
        {
            for (size_t j = 0; j < blockOut[i].CompSize(); ++j)
            {
                outPtr[j] =
                    ((static_cast<double>(rand()) / RAND_MAX) * 2.0 - 1.0);
            }
        }
    }

    // Ensure C0 continuity.
    assembOp->Apply(fOutCorrect, fOutCorrectAssemb);

    // Compute expected solution.
    elmtOp->Apply(fOutCorrectAssemb, fIn);

    // Warm up solves.
    for (unsigned int i = 0; i < Ntest / 2; ++i)
    {
        conjGradOp->Apply(fIn, fOut);
    }
    comm->Block();

    // Benchmark CG solve.
    nekDeviceSynchronize();
    timer.Start();
    LIKWID_MARKER_START(tag.c_str());

    unsigned int totalIterations = 0;
    for (unsigned int i = 0; i < Ntest; ++i)
    {
        conjGradOp->Apply(fIn, fOut);
        totalIterations += conjGradOp->GetNiterations();
    }

    nekDeviceSynchronize();
    LIKWID_MARKER_STOP(tag.c_str());
    timer.Stop();

    comm->Block();

    // Verification
    {
        auto &blockOut        = fOut.GetBlocks();
        auto &blockOutCorrect = fOutCorrectAssemb.GetBlocks();
        double totalDiff      = 0.0;
        double totalMaxDiff   = 0.0;
        size_t count          = 0;

        for (size_t i = 0; i < blockOut.size(); ++i)
        {
            auto outPtr =
                blockOut[i]
                    .template GetPtr<NektarSpaces::HostSpace, ReadOnly>();
            auto correctPtr =
                blockOutCorrect[i]
                    .template GetPtr<NektarSpaces::HostSpace, ReadOnly>();

            for (size_t j = 0; j < blockOut[i].CompSize(); ++j)
            {
                double diff =
                    std::abs((double)outPtr[j] - (double)correctPtr[j]);
                totalDiff += diff;
                totalMaxDiff = std::max(totalMaxDiff, diff);
                count++;
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

    if (comm->GetRank() == 0)
    {
        std::cout << "---------------------------------" << std::endl;
        std::cout << "ProfilerBP : " << tag << std::endl;
        std::cout << "Iterations : " << totalIterations << std::endl;
        std::cout << "---------------------------------" << std::endl;
    }

    // Collect elapsed time per test (average duration of one solve).
    auto rankElapsed = std::vector<double>(1, timer.TimePerTest(Ntest));

    PrintProfileResult<TData>(comm, expList, rankElapsed, blockAttr, blockAttr,
                              Ntest, totalIterations);
}
