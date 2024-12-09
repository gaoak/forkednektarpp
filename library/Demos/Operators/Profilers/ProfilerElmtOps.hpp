//////////////////////////////////////////////////////////////////////////////
//
// File: ProfilerElmtOps.hpp
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

#include <cstdio>
#include <iomanip>
#include <iostream>

#include <Operators/ElmtOps/OperatorBwdTrans.hpp>
#include <Operators/ElmtOps/OperatorHelmholtz.hpp>
#include <Operators/ElmtOps/OperatorIProductWRTBase.hpp>
#include <Operators/ElmtOps/OperatorIProductWRTDerivBase.hpp>
#include <Operators/ElmtOps/OperatorMass.hpp>
#include <Operators/ElmtOps/OperatorMultiplyByElmtInvMass.hpp>
#include <Operators/ElmtOps/OperatorPhysDeriv.hpp>
#include <Operators/Field/Field.hpp>
#include <Operators/LoopExecution/LoopExecution.hpp>
#include <Operators/MathKernels/MathKernels.hpp>
#include <Operators/Utils/UtilsKernels.hpp>

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
using FP_t = double;

static bool _verbose_ = false;

template <typename ExecSpace, typename TData> std::string GetExecutionName()
{
    std::string execName = "Unknown";
    if (std::is_same_v<ExecSpace, NektarSpaces::Serial>)
    {
        execName = "Serial";
    }
    else if (std::is_same_v<ExecSpace, NektarSpaces::AVX>)
    {
#if defined(__AVX512F__) && defined(NEKTAR_ENABLE_SIMD_AVX512)
        execName = "AVX512";
#elif defined(__AVX2__) && defined(NEKTAR_ENABLE_SIMD_AVX2)
        execName = "AVX2";
#else
        execName = "AVX";
#endif
    }
    else if (std::is_same_v<ExecSpace, NektarSpaces::CUDA>)
    {
        execName = "CUDA";
    }
    else if (std::is_same_v<ExecSpace, NektarSpaces::SYCL>)
    {
        execName = "SYCL";
    }
    else if (std::is_same_v<ExecSpace, NektarSpaces::KOKKOS>)
    {
        execName = "Kokkos";
    }
    return execName;
}

template <typename Impl> std::string GetImplementationName()
{
    std::string implName = "Unknown";
    if (std::is_same_v<Impl, SumFac>)
    {
        implName = "SumFac";
    }
    else if (std::is_same_v<Impl, SumFacQP>)
    {
        implName = "SumFacQP";
    }
    else if (std::is_same_v<Impl, StdMat>)
    {
        implName = "StdMat";
    }
    return implName;
}

template <class Op, typename TData> std::string GetOperatorName()
{
    std::string opName = "Unknown";
    if (std::is_same_v<Op, BwdTrans<TData>>)
    {
        opName = "BwdTrans";
    }
    else if (std::is_same_v<Op, IProductWRTBase<TData>>)
    {
        opName = "IProductWRTBase";
    }
    else if (std::is_same_v<Op, Mass<TData>>)
    {
        opName = "Mass";
    }
    else if (std::is_same_v<Op, MultiplyByElmtInvMass<TData>>)
    {
        opName = "MultiplyByElmtInvMass";
    }
    else if (std::is_same_v<Op, PhysDeriv<TData>>)
    {
        opName = "PhysDeriv";
    }
    else if (std::is_same_v<Op, IProductWRTDerivBase<TData>>)
    {
        opName = "IProductWRTDerivBase";
    }
    else if (std::is_same_v<Op, Helmholtz<TData>>)
    {
        opName = "Helmholtz";
    }
    return opName;
}

/// Print the block information. If _verbose_=true, then print the block
/// information for each rank on the screen. If _verbose_=false, then only print
/// the total information for each rank.
size_t PrintBlockInfo(const MultiRegions::ExpListSharedPtr &expList,
                      const std::vector<BlockAttributes> &blocks,
                      Array<OneD, int> &ranksNumElmts,
                      Array<OneD, int> &ranksNumPaddings,
                      Array<OneD, int> &ranksNumDofs)
{
    auto comm           = expList->GetComm();
    auto myrank         = comm->GetRank();
    ranksNumElmts       = Array<OneD, int>(comm->GetSize(), 0);
    ranksNumDofs        = Array<OneD, int>(comm->GetSize(), 0);
    ranksNumPaddings    = Array<OneD, int>(comm->GetSize(), 0);
    auto ranksGeomTypes = Array<OneD, int>(comm->GetSize(), 0);

    // Collect total information for each rank
    for (size_t i = 0, expId = 0; i < blocks.size(); ++i)
    {
        ranksNumElmts[myrank] += blocks[i].GetNumElements();
        ranksNumPaddings[myrank] +=
            blocks[i].GetNumElementsWithPadding() - blocks[i].GetNumElements();
        ranksNumDofs[myrank] += blocks[i].size();

        auto gtype = expList->GetExp(expId)->GetMetricInfo()->GetGtype();
        if (gtype == SpatialDomains::eDeformed && ranksGeomTypes[myrank] != 1)
        {
            ranksGeomTypes[myrank] = 2; // Deformed
        }
        else if (gtype == SpatialDomains::eRegular &&
                 ranksGeomTypes[myrank] != 2)
        {
            ranksGeomTypes[myrank] = 1; // Regular
        }
        else
        {
            ranksGeomTypes[myrank] = 3; // Mixed
        }
    }
    comm->AllReduce(ranksNumElmts, LibUtilities::ReduceSum);
    comm->AllReduce(ranksNumPaddings, LibUtilities::ReduceSum);
    comm->AllReduce(ranksNumDofs, LibUtilities::ReduceSum);
    comm->AllReduce(ranksGeomTypes, LibUtilities::ReduceMax);
    size_t ndofs = Vmath::Vsum(ranksNumDofs.size(), ranksNumDofs, 1);
    auto GeomType =
        Vmath::Vsum(ranksGeomTypes.size(), ranksGeomTypes, 1) / comm->GetSize();

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

    if (_verbose_) // print each block information in each rank
    {
        for (int rank = 0; rank < comm->GetSize(); rank++)
        {
            if (myrank == rank)
            {
                // Print block information in a table like format
                // | BlockId | #Elements | #Paddings | #Points | BlockSize |
                std::cout << std::endl;
                std::cout << std::setw(10) << "BlockId" << std::setw(12)
                          << "#Elements" << std::setw(12) << "#Paddings"
                          << std::setw(10) << "#Points" << std::setw(12)
                          << "BlockSize" << std::endl;
                // print a dash line
                std::cout << std::setw(10) << std::setfill('-') << ""
                          << std::setw(12) << "" << std::setw(12) << ""
                          << std::setw(10) << "" << std::setw(12) << ""
                          << std::setfill(' ') << std::endl;

                for (size_t i = 0; i < blocks.size(); ++i)
                {
                    if (_verbose_)
                    {
                        std::cout << std::setw(10) << i << std::setw(12)
                                  << blocks[i].GetNumElements() << std::setw(12)
                                  << blocks[i].GetNumElementsWithPadding() -
                                         blocks[i].GetNumElements()
                                  << std::setw(12) << std::setw(10)
                                  << blocks[i].GetNumData() << std::setw(12)
                                  << blocks[i].size() << std::endl;
                    }
                }

                // print a dash line
                std::cout << std::setw(10) << std::setfill('-') << ""
                          << std::setw(12) << "" << std::setw(12) << ""
                          << std::setw(10) << "" << std::setw(12) << ""
                          << std::setfill(' ') << std::endl;
                // Final line is sum of all blocks
                std::cout << std::setw(5) << "Rank#" << std::setw(5) << myrank
                          << std::setw(12) << ranksNumElmts[myrank]
                          << std::setw(12) << ranksNumPaddings[myrank]
                          << std::setw(10) << "N/A" << std::setw(12)
                          << ranksNumDofs[myrank] << std::endl;
                std::cout << std::endl;
            }
        }
    }

    // print summary information:
    if (comm->GetRank() == 0) // print the summary
    {
        std::cout << std::setw(10) << "Rank# " << std::setw(12) << "#Elements"
                  << std::setw(12) << "#Paddings" << std::setw(12) << "#DoFs"
                  << std::setw(12) << "Gtype" << std::endl;
        for (int rank = 0; rank < comm->GetSize(); rank++)
        {
            std::cout << std::setw(10) << rank << std::setw(12)
                      << ranksNumElmts[rank] << std::setw(12)
                      << ranksNumPaddings[rank] << std::setw(12)
                      << ranksNumDofs[rank] << std::setw(12) << Gtype
                      << std::endl;
        }
    }
    comm->Block();
    return ndofs;
}

/// Print the profile results, computed from the elapsed time and the total
/// number of dofs.
template <typename TData>
void PrintProfileResult(const CommSharedPtr &comm, const NekDouble timePerTest,
                        const Array<OneD, int> indofs,
                        const Array<OneD, int> outdofs)
{
    // collect elapsed time and compute the max, min, and average
    Array<OneD, NekDouble> rankElapsed(comm->GetSize(), 0.0);
    rankElapsed[comm->GetRank()] = timePerTest;
    comm->AllReduce(rankElapsed, LibUtilities::ReduceSum);
    NekDouble maxElapsed = Vmath::Vmax(rankElapsed.size(), rankElapsed, 1);
    NekDouble minElapsed = Vmath::Vmin(rankElapsed.size(), rankElapsed, 1);
    NekDouble aveElapsed =
        Vmath::Vsum(rankElapsed.size(), rankElapsed, 1) / comm->GetSize();
    // collect throughput for each rank:
    NekDouble inThroughput  = 0.0;
    NekDouble outThroughput = 0.0;
    size_t totInDofs        = 0.0;
    size_t totOutDofs       = 0.0;
    for (int i = 0; i < comm->GetSize(); i++)
    {
        inThroughput += indofs[i] / rankElapsed[i];
        outThroughput += outdofs[i] / rankElapsed[i];
        totInDofs += indofs[i];
        totOutDofs += outdofs[i];
    }

    if (comm->GetRank() == 0) // print the summary
    {
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
    }
    comm->Block();
}

template <class Op, FieldState stateIn, FieldState stateOut, typename ExecSpace,
          typename Impl, typename TData>
void Profiler(MultiRegions::ExpListSharedPtr &expList, const int Ntest,
              const int nIn = 1, const int nOut = 1)
{
    Timer timer;
    std::string execName = GetExecutionName<ExecSpace, TData>();
    std::string implName = GetImplementationName<Impl>();
    std::string OpName   = GetOperatorName<Op, TData>();
    auto tag             = OpName + execName + implName;

    if (std::is_same_v<TData, double>)
    {
        tag += "Double";
    }
    else if (std::is_same_v<TData, float>)
    {
        tag += "Float";
    }

    auto comm = expList->GetComm();

    if (comm->GetRank() == 0)
    {
        std::cout << "---------------------------------" << std::endl;
        std::cout << "ElmtOps Profiler : " << tag << std::endl;
        std::cout << "---------------------------------" << std::endl;
    }
    comm->Block();

    using MemSpace = typename ExecSpace::memory_space;

    // Create blocks
    auto blocks_in  = GetBlockAttributes<TData>(stateIn, expList);
    auto blocks_out = GetBlockAttributes<TData>(stateOut, expList);
    // Create fields
    auto in = Field<TData, stateIn>::template Create<MemSpace>(
        "f_in", blocks_in, nIn, ExecSpace::alignment);
    auto out = Field<TData, stateOut>::template Create<MemSpace>(
        "f_out", blocks_out, nOut, ExecSpace::alignment);
    // Create operator
    auto oper = Op::template Create<ExecSpace, Impl>(expList);

    // initialize the in field to random non-zeros: 1 2 3 4 ...
    // initialize the out field to zeros
    auto inblk = in.GetBlocks();
    for (size_t i = 0; i < inblk.size(); ++i)
    {
        auto inptr = inblk[i].template GetPtr<MemSpace, WriteOnly>();
        Nektar::parallel_for<ExecSpace>(
            0, inblk[i].size(),
            NEKTAR_LAMBDA(unsigned int j) { inptr[j] = j + 1.0; });
    }
    auto outblk = out.GetBlocks();
    for (size_t i = 0; i < outblk.size(); ++i)
    {
        auto outptr = outblk[i].template GetPtr<MemSpace, WriteOnly>();
        Nektar::parallel_for<ExecSpace>(
            0, outblk[i].size(),
            NEKTAR_LAMBDA(unsigned int j) { outptr[j] = 0.0; });
    }

    // Warm-up
    for (int i = 0; i < Ntest / 2; ++i)
    {
        oper->apply(in, out);
    }
    comm->Block();

    // Benchmark
    // For CUDA, we synchronize the device before starting/stopping the timer
    // For MPI, since elmental operators are local we don't need to synchronize
    // after each operator call. Just add a block after the timer stops.

#if defined(NEKTAR_ENABLE_CUDA)
    cudaDeviceSynchronize();
#elif defined(NEKTAR_ENABLE_KOKKOS)
    Kokkos::fence();
#endif
    timer.Start();
    LIKWID_MARKER_START(tag.c_str());

    for (int i = 0; i < Ntest; ++i)
    {
        oper->apply(in, out);
    }

#if defined(NEKTAR_ENABLE_CUDA)
    cudaDeviceSynchronize();
#elif defined(NEKTAR_ENABLE_KOKKOS)
    Kokkos::fence();
#endif
    LIKWID_MARKER_STOP(tag.c_str());
    timer.Stop();

    comm->Block();

    // check if the output is all zeros
    TData L2 = 0.0;
    l2norm<ExecSpace, TData, stateOut>(out, &L2);
    if (L2 < 1e-9)
    {
        std::cout << "Warning: output does not change!"
                  << "Device may not be invoked!" << std::endl;
    }

    Array<OneD, int> ranksNumElmts, ranksNumPaddings;
    Array<OneD, int> ranksNumInDofs, ranksNumOutDofs;

    // print block information and get the total number of dofs
    if (comm->GetRank() == 0)
    {
        std::cout << "Input field: " << std::endl;
    }
    PrintBlockInfo(expList, blocks_in, ranksNumElmts, ranksNumPaddings,
                   ranksNumInDofs);

    // print block information and get the total number of dofs
    if (comm->GetRank() == 0)
    {
        std::cout << "Output field: " << std::endl;
    }
    PrintBlockInfo(expList, blocks_out, ranksNumElmts, ranksNumPaddings,
                   ranksNumOutDofs);

    // collect elapsed time and compute the max, min, and average
    PrintProfileResult<TData>(comm, timer.TimePerTest(Ntest), ranksNumInDofs,
                              ranksNumOutDofs);
}

template <class Op, FieldState stateIn, FieldState stateOut, typename TData>
void LaunchProfiler(MultiRegions::ExpListSharedPtr &expList, const int Ntest,
                    const int nIn = 1, const int nOut = 1)
{
#if defined(NEKTAR_ENABLE_CUDA)
    Profiler<Op, stateIn, stateOut, NektarSpaces::CUDA, SumFac, TData>(
        expList, Ntest, nIn, nOut);
    Profiler<Op, stateIn, stateOut, NektarSpaces::CUDA, SumFacQP, TData>(
        expList, Ntest, nIn, nOut);
#elif defined(NEKTAR_ENABLE_SYCL)
    Profiler<Op, stateIn, stateOut, NektarSpaces::SYCL, SumFac, TData>(
        expList, Ntest, nIn, nOut);
    Profiler<Op, stateIn, stateOut, NektarSpaces::SYCL, SumFacQP, TData>(
        expList, Ntest, nIn, nOut);
#elif defined(NEKTAR_ENABLE_KOKKOS)
    Profiler<Op, stateIn, stateOut, NektarSpaces::KOKKOS, SumFac, TData>(
        expList, Ntest, nIn, nOut);
    Profiler<Op, stateIn, stateOut, NektarSpaces::KOKKOS, SumFacQP, TData>(
        expList, Ntest, nIn, nOut);
#else
#if defined(NEKTAR_ENABLE_SIMD_AVX2) || defined(NEKTAR_ENABLE_SIMD_AVX512)
    Profiler<Op, stateIn, stateOut, NektarSpaces::AVX, SumFac, TData>(
        expList, Ntest, nIn, nOut);
#else
    Profiler<Op, stateIn, stateOut, NektarSpaces::Serial, SumFac, TData>(
        expList, Ntest, nIn, nOut);
#endif
    // default: since it is extremely slow, we only execute 1/10 times
    Profiler<Op, stateIn, stateOut, NektarSpaces::Serial, StdMat, TData>(
        expList, Ntest / 10, nIn, nOut);
#endif
}
