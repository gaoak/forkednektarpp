//////////////////////////////////////////////////////////////////////////////
//
// File: ProfilerRiemannOps.hpp
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

#pragma once

#include <cstdio>
#include <iomanip>
#include <iostream>

#include <LibUtilities/BasicUtils/Field/Field.hpp>
#include <LibUtilities/BasicUtils/Math/Math.hpp>
#include <LibUtilities/BasicUtils/Utils/UtilsKernels.hpp>
#include <LibUtilities/LoopExecution/LoopExecution.hpp>

#include <LibUtilities/BasicUtils/ErrorUtil.hpp>
#include <LibUtilities/BasicUtils/Timer.h>
#include <MultiRegions/DisContField.h>
#include <MultiRegions/ExpList.h>
#include <SpatialDomains/MeshGraphIO.h>

#include "CompressibleSolver/CompressibleSolverOp.hpp"

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

// Helpers: flatten/unflatten component-major
template <typename TData>
static Array<OneD, TData> FlattenCompMajor(
    const Array<OneD, Array<OneD, TData>> &a, unsigned int ncomp, size_t npts)
{
    Array<OneD, TData> flat(ncomp * npts, 0.0);
    for (unsigned int c = 0; c < ncomp; ++c)
    {
        for (size_t i = 0; i < npts; ++i)
        {
            flat[c * npts + i] = a[c][i];
        }
    }
    return flat;
}

template <typename TData>
static void UnflattenCompMajor(const Array<OneD, TData> &flat,
                               Array<OneD, Array<OneD, TData>> &a,
                               unsigned int ncomp, size_t npts)
{
    for (unsigned int c = 0; c < ncomp; ++c)
    {
        for (size_t i = 0; i < npts; ++i)
        {
            a[c][i] = flat[c * npts + i];
        }
    }
}

/// Compute the expected results of certain operator from the expList
void GetExpectedResults(unsigned int spaceDim, size_t npts,
                        const Array<OneD, Array<OneD, double>> &normals,
                        Array<OneD, Array<OneD, double>> &fwd,
                        Array<OneD, Array<OneD, double>> &bwd,
                        Array<OneD, Array<OneD, double>> &flxRef,
                        double gamma = 1.4)
{
    const unsigned int nFields = spaceDim + 2;

    const double rho = 0.9;
    const double p   = 1.0;
    double u[3]      = {1.0, 2.0, 3.0};

    for (size_t i = 0; i < npts; ++i)
    {
        // density
        fwd[0][i] = rho;
        bwd[0][i] = rho;

        // momentum
        double mom[3] = {0, 0, 0};
        for (unsigned int d = 0; d < spaceDim; ++d)
        {
            mom[d]        = rho * u[d];
            fwd[1 + d][i] = mom[d];
            bwd[1 + d][i] = mom[d];
        }

        // energy
        const double rhoe = p / (gamma - 1.0);
        double kin        = 0.0;
        for (unsigned int d = 0; d < spaceDim; ++d)
        {
            kin += 0.5 * mom[d] * mom[d] / rho;
        }
        const double E      = rhoe + kin;
        fwd[nFields - 1][i] = E;
        bwd[nFields - 1][i] = E;

        // normal velocity u_n = u dot n
        double un = 0.0;
        for (unsigned int d = 0; d < spaceDim; ++d)
        {
            un += u[d] * normals[d][i];
        }

        // reference flux F·n
        // mass flux is rho * u_n
        flxRef[0][i] = rho * un; // mass

        // momentum flux is rho * u * u_n + p * n
        for (unsigned int k = 0; k < spaceDim; ++k)
        {
            flxRef[1 + k][i] = mom[k] * un + p * normals[k][i];
        }

        // energy flux is (E + p) * u_n
        flxRef[nFields - 1][i] = (E + p) * un;
    }
}

/// Print the block information. If _verbose_=true, then print the block
/// information for each rank. If _verbose_=false, then only print the
/// total information for each rank. Caution: for many ranks and many
/// blocks, setting verbose may cause the display content too big to read.
template <FieldState TState>
void PrintBlockInfo(
    const MultiRegions::ExpListSharedPtr &expList,
    const std::vector<LibUtilities::BlockAttributes<TState>> &blockAttr,
    std::vector<double> &rankL1Err)
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
    auto allRankL1Err            = comm->Gather(0, rankL1Err);

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
                      << std::setw(16)
                      << allRankL1Err[rank] / allRankNumDofs[rank] << std::endl;
        }
    }
    comm->Block();
}

/// Print the profiler results, computed from the elapsed time and the total
/// number of dofs.
template <typename TData, FieldState TState>
void PrintProfileResult(
    const CommSharedPtr comm, std::vector<double> &rankElapsed,
    const std::vector<LibUtilities::BlockAttributes<TState>> &traceblockAttr)
{
    // Collect elapsed time and compute the max, min, and average.
    unsigned int nrank  = comm->GetSize();
    auto rankNumInDofs  = std::vector<size_t>(1, 0);
    auto rankNumOutDofs = std::vector<size_t>(1, 0);

    // Collect total information for each rank.
    for (unsigned int i = 0; i < traceblockAttr.size(); ++i)
    {
        rankNumInDofs[0] += traceblockAttr[i].CompSize();
        rankNumOutDofs[0] += traceblockAttr[i].CompSize();
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

// Different operator may have different input/output attributes (FieldState,
// or number of components). We must provided all these information.
template <class Op, FieldState state, typename TData>
void LaunchProfiler(MultiRegions::ExpListSharedPtr const &expList,
                    const unsigned int Ntest, const std::string &method = "",
                    const unsigned int nHomo = 1)
{
    // Timer.
    Timer timer;

    auto session = expList->GetSession();

    // Get communicator.
    auto comm = expList->GetComm();

    const unsigned int spaceDim = expList->GetExp(0)->GetShapeDimension();
    const size_t npts           = expList->GetTrace()->GetTotPoints();
    const unsigned int nFields  = (spaceDim + 2) * nHomo;

    std::vector<std::string> nVariabels;
    for (unsigned int i = 0; i < nFields; ++i)
    {
        nVariabels.push_back("DefaultVar" + std::to_string(i));
    }

    // Create operator.
    auto oper =
        CompressibleSolverOp<TData>::Create(expList, nVariabels, method, "");

    // Set operator name tag.
    std::string execName = Op::GetOpExecSpace(session);
    std::string opName   = method;
    std::string dataType = (std::is_same_v<TData, double>) ? "Double" : "Float";
    auto tag             = opName + execName + dataType;

    // normals: unit along normalDir
    Array<OneD, Array<OneD, double>> normals(spaceDim);
    for (unsigned int d = 0; d < spaceDim; ++d)
    {
        normals[d] = Array<OneD, double>(npts, 0.0);
    }
    for (size_t i = 0; i < npts; ++i)
    {
        normals[0][i] = 1.0;
    }

    // arrays
    Array<OneD, Array<OneD, double>> fwd(nFields), bwd(nFields), flx(nFields),
        flxRef(nFields);
    for (unsigned int c = 0; c < nFields; ++c)
    {
        fwd[c]    = Array<OneD, double>(npts, 0.0);
        bwd[c]    = Array<OneD, double>(npts, 0.0);
        flx[c]    = Array<OneD, double>(npts, 0.0);
        flxRef[c] = Array<OneD, double>(npts, 0.0);
    }

    // Get expected results by constant Euler state and reference physical flux
    GetExpectedResults(spaceDim, npts, normals, fwd, bwd, flxRef);

    // Create block attributes.
    auto traceblockAttr = GetBlockAttributes<TData, state>(expList->GetTrace());

    // Create fields.
    auto fwdField = LibUtilities::Field<TData, state>("f_fwd", traceblockAttr,
                                                      nFields, nHomo);

    auto bwdField = LibUtilities::Field<TData, state>("f_bwd", traceblockAttr,
                                                      nFields, nHomo);

    auto fluxField = LibUtilities::Field<TData, state>("f_flux", traceblockAttr,
                                                       nFields, nHomo);

    auto normalsField = LibUtilities::Field<TData, state>(
        "traceNormals", traceblockAttr, spaceDim, nHomo);

    fwdField.template CopyArray<NektarSpaces::HostSpace>(
        FlattenCompMajor(fwd, nFields, npts));
    bwdField.template CopyArray<NektarSpaces::HostSpace>(
        FlattenCompMajor(bwd, nFields, npts));
    normalsField.template CopyArray<NektarSpaces::HostSpace>(
        FlattenCompMajor(normals, spaceDim, npts));

    oper->SetTraceNormals(normalsField);

    // Warm-up : fill the cache and memory, and let core temperature/freq
    // stabilized.
    for (unsigned int i = 0; i < Ntest / 2; ++i)
    {
        oper->Apply(fwdField, bwdField, fluxField);
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
        oper->Apply(fwdField, bwdField, fluxField);
    }

    nekDeviceSynchronize();
    LIKWID_MARKER_STOP(tag.c_str());
    timer.Stop();

    comm->Block();

    auto rankL1Error = std::vector<double>(1, 0.0);

    // Print block information and get the total number of dofs.
    if (comm->GetRank() == 0)
    {
        std::cout << "Input field: " << nFields << " components" << std::endl;
    }
    PrintBlockInfo(expList, traceblockAttr, rankL1Error);

    // First check if the output is all zeros.
    TData L2 = 0.0;
    Math::l2norm<NektarSpaces::Serial>(fluxField, &L2);
    if (L2 < 1e-9)
    {
        std::cout << "Warning: output does not change!"
                  << "Device may not be invoked!" << std::endl;
    }

    // Then check if results match with expected
    // If we compare float results with double results, then it is
    // reasonable to have some mismatched values (e.g., > 1e-4)
    Array<OneD, double> tmpArr  = fluxField.template ToArray<double>();
    Array<OneD, double> fluxArr = FlattenCompMajor(flxRef, nFields, npts);
    for (size_t i = 0, cnt = 0; i < tmpArr.size(); ++i)
    {
        // Print out first 100 mismatched values.
        if (abs(tmpArr[i] - fluxArr[i]) >
                1e-4 * std::sqrt(L2 / tmpArr.size()) &&
            cnt < 100)
        {
            std::cout << "i=" << i << " computed result = " << tmpArr[i]
                      << " expected result = " << fluxArr[i] << std::endl;
            ++cnt;
        }
        rankL1Error[0] += abs(tmpArr[i] - fluxArr[i]);
    }

    // Print block information and get the total number of dofs.
    if (comm->GetRank() == 0)
    {
        std::cout << "Output field: " << nFields << " components" << std::endl;
    }
    PrintBlockInfo(expList->GetTrace(), traceblockAttr, rankL1Error);

    if (comm->GetRank() == 0)
    {
        std::cout << "---------------------------------" << std::endl;
        std::cout << "ElmtOps Profiler : " << tag << std::endl;
        std::cout << "---------------------------------" << std::endl;
    }
    comm->Block();

    // Collect elapsed time and compute the max, min, and average.
    auto rankElapsed = std::vector<double>(1, timer.TimePerTest(Ntest));
    PrintProfileResult<TData>(comm, rankElapsed, traceblockAttr);
}
