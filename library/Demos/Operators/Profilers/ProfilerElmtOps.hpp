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
// Description: main content of profilers
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

/// Compute the expected results of certain operator from the expList
template <class Op, typename TData>
void GetExpectedResults(const MultiRegions::ExpListSharedPtr &expList,
                        const Array<OneD, NekDouble> &inArr,
                        const Array<OneD, Array<OneD, NekDouble>> &inArrays,
                        Array<OneD, NekDouble> &outArr,
                        Array<OneD, Array<OneD, NekDouble>> &outArrays)
{
    std::string opName = "Unknown";
    if (std::is_same<Op, BwdTrans<TData>>::value)
    {
        expList->BwdTrans(inArr, outArr);
    }
    else if (std::is_same<Op, IProductWRTBase<TData>>::value)
    {
        expList->IProductWRTBase(inArr, outArr);
    }
    else if (std::is_same<Op, Mass<TData>>::value)
    {
        expList->GeneralMatrixOp(
            MultiRegions::GlobalMatrixKey(StdRegions::eMass), inArr, outArr);
    }
    else if (std::is_same<Op, MultiplyByElmtInvMass<TData>>::value)
    {
        expList->GeneralMatrixOp(
            MultiRegions::GlobalMatrixKey(StdRegions::eInvMass), inArr, outArr);
    }
    else if (std::is_same<Op, PhysDeriv<TData>>::value)
    {
        for (int d = 0; d < outArrays.size(); d++)
        {
            expList->PhysDeriv(d, inArr, outArrays[d]);
        }
    }
    else if (std::is_same<Op, IProductWRTDerivBase<TData>>::value)
    {
        expList->IProductWRTDerivBase(inArrays, outArr);
    }
    else if (std::is_same<Op, Helmholtz<TData>>::value)
    {
        StdRegions::ConstFactorMap factors;
        factors[StdRegions::eFactorLambda] = 1.0;
        MultiRegions::GlobalMatrixKey gkey(
            StdRegions::eHelmholtz, MultiRegions::NullAssemblyMapSharedPtr,
            factors);
        expList->GeneralMatrixOp(gkey, inArr, outArr);
    }
}

/// Reshape the storage of the field, to match the layout of legacy Nektar
/// Array, for result comparison.
template <typename TData, FieldState stateOut>
void ReshapeToScalar(Field<TData, stateOut> &in)
{
    for (unsigned int blk = 0; blk < in.GetBlocks().size(); ++blk)
    {
        auto &block = in.GetBlocks()[blk];
        TData *inptr =
            block.template GetPtr<NektarSpaces::HostSpace, ReadWrite>();
        unsigned int numElmtsPad =
            block.GetNumElements() + block.GetNumPaddingElements();
        for (unsigned int component = 0; component < in.GetNumComponents();
             component++)
        {
            ReshapeStorage<NektarSpaces::Serial, 1>(
                block.GetInterleaveWidth(), numElmtsPad, block.GetNumData(),
                inptr + component * block.size());
        }

        block.template SetInterleaveWidth<TData>(1);
    }
}

/// Print the block information. If _verbose_=true, then print the block
/// information for each rank. If _verbose_=false, then only print the
/// total information for each rank. Caution: for many ranks and many
/// blocks, setting verbose may cause the display content too big to read.
size_t PrintBlockInfo(const MultiRegions::ExpListSharedPtr &expList,
                      const std::vector<BlockAttributes> &blocks,
                      Array<OneD, int> &ranksNumElmts,
                      Array<OneD, int> &ranksNumPaddings,
                      Array<OneD, int> &ranksNumDofs,
                      Array<OneD, NekDouble> &rankL1Err)
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

        // check the geometry type of the block: deformed or regular
        // if both types exist in the same rank, then it is labeled as mixed
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

    // Get the geometry type of the whole domain :
    // Calculate the average value, if average is 1, then it is regular
    // if average is 2, then it is deformed, otherwise it is mixed/
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

    comm->AllReduce(rankL1Err, LibUtilities::ReduceSum);

    // print summary information:
    if (comm->GetRank() == 0) // print the summary
    {
        std::cout << std::setw(10) << "Rank# " << std::setw(12) << "#Elements"
                  << std::setw(12) << "#Paddings" << std::setw(12) << "#DoFs"
                  << std::setw(12) << "Gtype" << std::setw(16) << "L1 error"
                  << std::endl;
        for (int rank = 0; rank < comm->GetSize(); rank++)
        {
            std::cout << std::setw(10) << rank << std::setw(12)
                      << ranksNumElmts[rank] << std::setw(12)
                      << ranksNumPaddings[rank] << std::setw(12)
                      << ranksNumDofs[rank] << std::setw(12) << Gtype
                      << std::setw(16) << rankL1Err[rank] << std::endl;
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
    // The throughput for each rank is calucated by the total number of dofs
    // and the elapsed time in that rank. The total throughput is the sum of
    // all ranks. So the total throughput by this way will not be identical
    // to total dofs divided by the total (min/ave/max) elapsed time.
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

    // print the summary :
    // Max/Min/Aver time can be used to judge the load-balance in parallel
    if (comm->GetRank() == 0)
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
        std::cout << std::endl;
    }
    comm->Block();
}

// Different operator may have different input/output attributes (FieldState,
// or number of components). We must provided all these information.
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

    // identify the data type
    if (std::is_same_v<TData, double>)
    {
        tag += "Double";
    }
    else if (std::is_same_v<TData, float>)
    {
        tag += "Float";
    }

    auto comm = expList->GetComm();

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
    auto inblk    = in.GetBlocks();
    TData blksize = inblk.size();
    for (size_t i = 0; i < inblk.size(); ++i)
    {
        auto inptr = inblk[i].template GetPtr<MemSpace, WriteOnly>();
        Nektar::parallel_for<ExecSpace>(
            0, inblk[i].size(),
            NEKTAR_LAMBDA(unsigned int j) { inptr[j] = (j + 1.0) / blksize; });
    }
    auto outblk = out.GetBlocks();
    for (size_t i = 0; i < outblk.size(); ++i)
    {
        auto outptr = outblk[i].template GetPtr<MemSpace, WriteOnly>();
        Nektar::parallel_for<ExecSpace>(
            0, outblk[i].size(),
            NEKTAR_LAMBDA(unsigned int j) { outptr[j] = 0.0; });
    }

    // Create input and output Array for explist
    Array<OneD, NekDouble> inArr  = in.template ToArray<NekDouble>();
    Array<OneD, NekDouble> outArr = out.template ToArray<NekDouble>();
    Array<OneD, Array<OneD, NekDouble>> inArrays(nIn);
    Array<OneD, Array<OneD, NekDouble>> outArrays(nOut);
    for (int d = 0; d < nIn; d++)
    {
        inArrays[d] = inArr + d * inArr.size() / nIn;
    }
    for (int d = 0; d < nOut; d++)
    {
        outArrays[d] = outArr + d * outArr.size() / nOut;
    }
    // Get expected results from expList
    GetExpectedResults<Op, TData>(expList, inArr, inArrays, outArr, outArrays);

    // Warm-up : fill the cache and memory, and let core temperature/freq
    // stabilized
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

    // collect block information for each rank:
    // number of elements for each rank;
    // number of paddings for each rank, etc.
    Array<OneD, NekDouble> rankL1Error(comm->GetSize(), 0.0);
    Array<OneD, int> ranksNumElmts, ranksNumPaddings;
    Array<OneD, int> ranksNumInDofs, ranksNumOutDofs;

    // print block information and get the total number of dofs
    if (comm->GetRank() == 0)
    {
        std::cout << "Input field: " << std::endl;
    }
    PrintBlockInfo(expList, blocks_in, ranksNumElmts, ranksNumPaddings,
                   ranksNumInDofs, rankL1Error);

    // Additional check on the results, you can disable it if not used
    {
        // First check if the output is all zeros
        TData L2 = 0.0;
        l2norm<ExecSpace, TData, stateOut>(out, &L2);
        if (L2 < 1e-9)
        {
            std::cout << "Warning: output does not change!"
                      << "Device may not be invoked!" << std::endl;
        }

        // Then check if results match with expected
        // If we compare float results with double results, then it is
        // reasonable to have some mismatched values (e.g., > 1e-4)
        ReshapeToScalar<TData, stateOut>(out);
        Array<OneD, NekDouble> tmpArr = out.template ToArray<NekDouble>();
        for (int i = 0, cnt = 0; i < tmpArr.size(); ++i)
        {
            // print out first 100 mismatched values
            if (abs(tmpArr[i] - outArr[i]) > 1e-4 && cnt < 100)
            {
                std::cout << "i=" << i << " computed result = " << tmpArr[i]
                          << " expected result = " << outArr[i] << std::endl;
                ++cnt;
            }
            rankL1Error[comm->GetRank()] += abs(tmpArr[i] - outArr[i]);
        }
    }

    // print block information and get the total number of dofs
    if (comm->GetRank() == 0)
    {
        std::cout << "Output field: " << std::endl;
    }
    PrintBlockInfo(expList, blocks_out, ranksNumElmts, ranksNumPaddings,
                   ranksNumOutDofs, rankL1Error);

    if (comm->GetRank() == 0)
    {
        std::cout << "---------------------------------" << std::endl;
        std::cout << "ElmtOps Profiler : " << tag << std::endl;
        std::cout << "---------------------------------" << std::endl;
    }
    comm->Block();

    // collect elapsed time and compute the max, min, and average
    PrintProfileResult<TData>(comm, timer.TimePerTest(Ntest), ranksNumInDofs,
                              ranksNumOutDofs);
}

// You can add/remove the impl or exec to be profiled together as you like.
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
    // Since StdMat on CPU is really slow, we disable it by default
    // Profiler<Op, stateIn, stateOut, NektarSpaces::Serial, StdMat, TData>(
    //     expList, Ntest / 20, nIn, nOut);
#endif
}
