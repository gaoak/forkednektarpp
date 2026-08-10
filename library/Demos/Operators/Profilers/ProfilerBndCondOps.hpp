//////////////////////////////////////////////////////////////////////////////
//
// File: ProfilerBndCondOps.hpp
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

#include <Operators/BndCondOps/DirBndCond/DirBndCondOp.hpp>
#include <Operators/BndCondOps/NeuBndCond/NeuBndCondOp.hpp>
#include <Operators/Field/Field.hpp>
#include <Operators/Math/MathKernels.hpp>

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

/// Print the block information. If _verbose_=true, then print the block
/// information for each rank. If _verbose_=false, then only print the
/// total information for each rank. Caution: for many ranks and many
/// blocks, setting verbose may cause the display content too big to read.
template <FieldState TState>
void PrintBlockInfo(const MultiRegions::ContFieldSharedPtr &expList,
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
/// number of dofs. `numBndDofs` is this rank's number of boundary
/// coefficients actually touched by Apply() (see
/// DirBndCondOp/NeuBndCondOp::GetNumBndDofs()), reported alongside the
/// domain-wide figures since the operator only ever writes to that much
/// smaller subset of the field.
template <typename TData, FieldState TStateIn, FieldState TStateOut>
void PrintProfileResult(
    const CommSharedPtr comm, std::vector<double> &rankElapsed,
    const std::vector<BlockAttributes<TStateIn>> &inblockAttr,
    const std::vector<BlockAttributes<TStateOut>> &outblockAttr,
    size_t numBndDofs)
{
    // Collect elapsed time and compute the max, min, and average.
    unsigned int nrank  = comm->GetSize();
    auto rankNumInDofs  = std::vector<size_t>(1, 0);
    auto rankNumOutDofs = std::vector<size_t>(1, 0);
    auto rankNumBndDofs = std::vector<size_t>(1, numBndDofs);

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
    auto allRankNumBndDofs = comm->Gather(0, rankNumBndDofs);

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
        double bndThroughput = 0.0;
        size_t totInDofs     = 0;
        size_t totOutDofs    = 0;
        size_t totBndDofs    = 0;
        for (unsigned int rank = 0; rank < nrank; rank++)
        {
            inThroughput += allRankNumInDofs[rank] / allRankElapsed[rank];
            outThroughput += allRankNumOutDofs[rank] / allRankElapsed[rank];
            bndThroughput += allRankNumBndDofs[rank] / allRankElapsed[rank];
            totInDofs += allRankNumInDofs[rank];
            totOutDofs += allRankNumOutDofs[rank];
            totBndDofs += allRankNumBndDofs[rank];
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
        std::cout << "Total boundary ndof: " << totBndDofs << std::endl;
        std::cout << "Boundary throughput (ndof/s): " << bndThroughput
                  << std::endl;
        std::cout << "Boundary throughput (GB/s): "
                  << bndThroughput * sizeof(TData) / 1e9 << std::endl;
        std::cout << std::endl;
        std::cout << std::endl;
    }
    comm->Block();
}

// DirBndCondOp and NeuBndCondOp both operate in-place on a single
// Field<TData, FieldState::Coeff>, so unlike the ElmtOps profiler there is
// only one field/one FieldState to profile.
template <template <typename> typename Op, typename TData>
void LaunchProfiler(MultiRegions::ContFieldSharedPtr &expList,
                    const unsigned int Ntest, const double time = 0.0,
                    const bool timeDependent = false, const double dt = 1e-3,
                    const unsigned int nHomo = 1)
{
    // Timer.
    Timer timer;

    auto session = expList->GetSession();

    // Get communicator.
    auto comm = expList->GetComm();

    // Boundary condition operators are created against the list of session
    // variables/components, same as e.g. AssmbScatrZeroDirOp.
    auto variables     = session->GetVariables();
    unsigned int nComp = variables.size();

    // Create operator.
    auto oper = Op<TData>::Create(expList, variables);

    // Set operator name tag.
    std::string execName = Op<TData>::GetOpExecSpace(session);
    std::string opName   = oper->name;
    std::string dataType = (std::is_same_v<TData, double>) ? "Double" : "Float";
    auto tag = opName + execName + dataType + (timeDependent ? "TimeDep" : "");

    // Evaluate the boundary condition expressions and populate the
    // operator's internal boundary coefficients for this time level.
    //
    // If `timeDependent` is not requested, this happens once here, up
    // front, and the warm-up/timed loops below measure Apply() alone. If
    // `timeDependent` is requested, this call is instead made inside the
    // loops (see below), advancing `time` by `dt` on every iteration, to
    // profile the realistic per-timestep cost of re-evaluating
    // time-dependent boundary condition expressions together with Apply().
    // Note UpdateBndCoeffs() itself early-returns cheaply once the mesh's
    // boundary conditions are known not to be time dependent, so enabling
    // `timeDependent` on a mesh with only static BCs mostly just measures
    // that short-circuit check's overhead -- use a mesh with genuinely
    // time-dependent BCs (e.g. Helmholtz3D_Hex_AllBCs_P6_TimeDependentBC.xml)
    // to get a meaningful measurement.
    if (!timeDependent)
    {
        oper->UpdateBndCoeffs(static_cast<TData>(time));
    }

    // Create block attributes and the (single, in-place) coefficient field.
    auto blockAttr = GetBlockAttributes<TData, FieldState::Coeff>(expList);
    auto out =
        Field<TData, FieldState::Coeff>("f_out", blockAttr, nComp, nHomo);

    // Start from all-zero coefficients, matching how DirBndCondOp and
    // NeuBndCondOp are exercised in the unit tests (test_dirichlet.cpp,
    // test_neumann.cpp): this way a non-zero result after Apply() is a
    // meaningful signal that boundary conditions were actually imposed.
    out.template Initialize<NektarSpaces::HostSpace>(0.0);

    // Reshape.
    auto interleaveWidth = (execName == "AVX" || execName == "Device")
                               ? NektarSpaces::GetVectorWidth<TData>(execName)
                               : 1;
    out.ReshapeStorage(interleaveWidth, execName);

    // A single timestep of work: optionally refresh the boundary
    // coefficients at time level `stepTime` (time-dependent profiling),
    // then apply the boundary condition operator.
    double t  = time;
    auto step = [&](double stepTime) {
        if (timeDependent)
        {
            oper->UpdateBndCoeffs(static_cast<TData>(stepTime));
        }
        oper->Apply(out);
    };

    // Warm-up : fill the cache and memory, and let core temperature/freq
    // stabilized.
    //
    // Note: DirBndCondOp overwrites its boundary coefficients on every call
    // (idempotent), but NeuBndCondOp accumulates its weak boundary forcing
    // into `out` (atomic-add) rather than overwriting it, so repeated calls
    // here keep adding to the same buffer. That is intentional and harmless
    // for a throughput measurement: we are timing the operator's raw
    // per-call cost, not re-imposing a single boundary condition state.
    for (unsigned int i = 0; i < Ntest / 2; ++i)
    {
        step(t);
        t += dt;
    }
    comm->Block();

    // Benchmark
    // For CUDA, we synchronize the device before starting/stopping the timer
    // For MPI, since elmental operators are local we don't need to synchronize
    // after each operator call. Just add a block after the timer stops.

    nekDeviceSynchronize();
    timer.Start();
    LIKWID_MARKER_START(tag.c_str());

    for (unsigned int i = 0; i < Ntest; ++i)
    {
        step(t);
        t += dt;
    }

    nekDeviceSynchronize();
    LIKWID_MARKER_STOP(tag.c_str());
    timer.Stop();

    comm->Block();

    // Print block information and get the total number of dofs.
    if (comm->GetRank() == 0)
    {
        std::cout << "Field: " << nComp << " components" << std::endl;
    }
    PrintBlockInfo(expList, blockAttr);

    // Check if the boundary conditions actually changed the field. Since
    // `out` started at all-zero, this catches both a mesh with no
    // opName-type boundary conditions on the profiled variable(s), and (on
    // Device) a kernel that silently was not invoked.
    out.ReshapeStorage(1, execName);
    TData L2 = 0.0;
    l2norm<NektarSpaces::Serial>(out, &L2);
    if (L2 < 1e-9)
    {
        std::cout << "Warning: output is all zero! Either the mesh has no "
                  << opName
                  << "-type boundary conditions on the profiled variable(s), "
                  << "or the device may not be invoked." << std::endl;
    }

    if (comm->GetRank() == 0)
    {
        std::cout << "---------------------------------" << std::endl;
        std::cout << "BndCondOps Profiler : " << tag << std::endl;
        std::cout << "---------------------------------" << std::endl;
    }
    comm->Block();

    // Collect elapsed time and compute the max, min, and average.
    auto rankElapsed = std::vector<double>(1, timer.TimePerTest(Ntest));
    PrintProfileResult<TData>(comm, rankElapsed, blockAttr, blockAttr,
                              oper->GetNumBndDofs());
}
