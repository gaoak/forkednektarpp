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

#include <Operators/ElmtOps/BwdTrans/BwdTransOp.hpp>
#include <Operators/ElmtOps/Helmholtz/HelmholtzOp.hpp>
#include <Operators/ElmtOps/IProductWRTBase/IProductWRTBaseOp.hpp>
#include <Operators/ElmtOps/IProductWRTDerivBase/IProductWRTDerivBaseOp.hpp>
#include <Operators/ElmtOps/LinAdvDiffReaction/LinAdvDiffReactionOp.hpp>
#include <Operators/ElmtOps/Mass/MassOp.hpp>
#include <Operators/ElmtOps/MultiplyByElmtInvMass/MultiplyByElmtInvMassOp.hpp>
#include <Operators/ElmtOps/PhysDeriv/PhysDerivOp.hpp>
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

/// Compute the expected results of certain operator from the expList
void GetExpectedResults(const std::string &opName,
                        const MultiRegions::ExpListSharedPtr &expList,
                        const Array<OneD, NekDouble> &inArr,
                        const Array<OneD, Array<OneD, NekDouble>> &inArrays,
                        Array<OneD, NekDouble> &outArr,
                        Array<OneD, Array<OneD, NekDouble>> &outArrays)
{
    if (opName == "BwdTrans")
    {
        expList->BwdTrans(inArr, outArr);
    }
    else if (opName == "IProductWRTBase")
    {
        expList->IProductWRTBase(inArr, outArr);
    }
    else if (opName == "Mass")
    {
        expList->GeneralMatrixOp(
            MultiRegions::GlobalMatrixKey(StdRegions::eMass), inArr, outArr);
    }
    else if (opName == "MultiplyByElmtInvMass")
    {
        expList->GeneralMatrixOp(
            MultiRegions::GlobalMatrixKey(StdRegions::eInvMass), inArr, outArr);
    }
    else if (opName == "PhysDeriv")
    {
        for (unsigned int d = 0; d < outArrays.size(); d++)
        {
            expList->PhysDeriv(d, inArr, outArrays[d]);
        }
    }
    else if (opName == "IProductWRTDerivBase")
    {
        expList->IProductWRTDerivBase(inArrays, outArr);
    }
    else if (opName == "Helmholtz")
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
        auto numElmtsPad = block.GetNumElementsWithPadding();
        for (unsigned int component = 0; component < in.GetNumComponents();
             component++)
        {
            ReshapeStorage<NektarSpaces::Serial>(
                1, block.GetInterleaveWidth(),
                numElmtsPad * in.GetNumHomoModes(), block.GetNumData(),
                inptr + component * block.size() * in.GetNumHomoModes());
        }

        block.template SetInterleaveWidth<TData>(1);
    }
}

/// Print the block information. If _verbose_=true, then print the block
/// information for each rank. If _verbose_=false, then only print the
/// total information for each rank. Caution: for many ranks and many
/// blocks, setting verbose may cause the display content too big to read.
void PrintBlockInfo(const MultiRegions::ExpListSharedPtr &expList,
                    const std::vector<BlockAttributes> &blocks,
                    std::vector<NekDouble> &rankL1Err)
{
    auto comm                 = expList->GetComm();
    unsigned int nrank        = comm->GetSize();
    auto rankBlockNum         = std::vector<unsigned int>(1, blocks.size());
    auto rankBlockNumDofs     = std::vector<size_t>(blocks.size(), 0);
    auto rankBlockNumData     = std::vector<unsigned int>(blocks.size(), 0);
    auto rankBlockNumElmts    = std::vector<size_t>(blocks.size(), 0);
    auto rankBlockNumPaddings = std::vector<size_t>(blocks.size(), 0);
    auto rankNumDofs          = std::vector<size_t>(1, 0);
    auto rankNumElmts         = std::vector<size_t>(1, 0);
    auto rankNumPaddings      = std::vector<size_t>(1, 0);
    auto rankGeomTypes        = std::vector<unsigned int>(1, 0);

    // Collect total information for each rank.
    for (size_t i = 0, expId = 0; i < blocks.size(); ++i)
    {
        rankBlockNumDofs[i]  = blocks[i].size();
        rankBlockNumData[i]  = blocks[i].GetNumData();
        rankBlockNumElmts[i] = blocks[i].GetNumElements();
        rankBlockNumPaddings[i] =
            blocks[i].GetNumElementsWithPadding() - blocks[i].GetNumElements();
        rankNumDofs[0] += blocks[i].size();
        rankNumElmts[0] += blocks[i].GetNumElements();
        rankNumPaddings[0] +=
            blocks[i].GetNumElementsWithPadding() - blocks[i].GetNumElements();

        // Check the geometry type of the block: deformed or regular
        // if both types exist in the same rank, then it is labeled as mixed.
        auto gtype = expList->GetExp(expId)->GetMetricInfo()->GetGtype();
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
template <typename TData>
void PrintProfileResult(const CommSharedPtr comm,
                        std::vector<NekDouble> &rankElapsed,
                        const std::vector<BlockAttributes> &inblocks,
                        const std::vector<BlockAttributes> &outblocks)
{
    // Collect elapsed time and compute the max, min, and average.
    unsigned int nrank  = comm->GetSize();
    auto rankNumInDofs  = std::vector<size_t>(1, 0);
    auto rankNumOutDofs = std::vector<size_t>(1, 0);

    // Collect total information for each rank.
    for (unsigned int i = 0; i < inblocks.size(); ++i)
    {
        rankNumInDofs[0] += inblocks[i].size();
    }
    for (unsigned int i = 0; i < outblocks.size(); ++i)
    {
        rankNumOutDofs[0] += outblocks[i].size();
    }

    auto allRankElapsed    = comm->Gather(0, rankElapsed);
    auto allRankNumInDofs  = comm->Gather(0, rankNumInDofs);
    auto allRankNumOutDofs = comm->Gather(0, rankNumOutDofs);

    if (comm->GetRank() == 0)
    {
        NekDouble maxElapsed = Vmath::Vmax(nrank, allRankElapsed.data(), 1);
        NekDouble minElapsed = Vmath::Vmin(nrank, allRankElapsed.data(), 1);
        NekDouble aveElapsed =
            Vmath::Vsum(nrank, allRankElapsed.data(), 1) / nrank;

        // Collect throughput for each rank:
        // The throughput for each rank is calucated by the total number of dofs
        // and the elapsed time in that rank. The total throughput is the sum of
        // all rank. So the total throughput by this way will not be identical
        // to total dofs divided by the total (min/ave/max) elapsed time.
        NekDouble inThroughput  = 0.0;
        NekDouble outThroughput = 0.0;
        size_t totInDofs        = 0;
        size_t totOutDofs       = 0;
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
template <class Op, FieldState stateIn, FieldState stateOut, typename TData>
void LaunchProfiler(MultiRegions::ExpListSharedPtr &expList,
                    const unsigned int Ntest, const unsigned int nIn = 1,
                    const unsigned int nOut = 1, const unsigned int nComp = 1,
                    const unsigned int nHomo = 1)
{
    // Timer.
    Timer timer;

    // Get communicator.
    auto comm = expList->GetComm();

    // Create operator.
    auto oper = Op::Create(expList);

    // Set operator name tag.
    auto session = expList->GetSession();
    std::string execName =
        session->GetCmdLineArgument<std::string>("opExecSpace");
    std::string implName = session->GetCmdLineArgument<std::string>("opImpl");
    std::string opName   = oper->name;
    std::string dataType = (std::is_same_v<TData, double>) ? "Double" : "Float";
    auto tag             = opName + execName + implName + dataType;

    // Check if addition configure is required.
    if (opName == "Helmholtz")
    {
        std::dynamic_pointer_cast<HelmholtzOp<TData>>(oper)->SetLambda(1.0);
    }
    if (opName == "LinAdvDiffReaction")
    {
        Array<OneD, NekDouble> vel(expList->GetCoordim(0) *
                                       (size_t)expList->GetNpoints(),
                                   1.0); // prevent overflow
        std::dynamic_pointer_cast<LinAdvDiffReactionOp<TData>>(oper)->SetLambda(
            -1.0);
        std::dynamic_pointer_cast<LinAdvDiffReactionOp<TData>>(oper)->SetAdvVel(
            expList->GetCoordim(0), vel);
    }

    // Set alignment.
    size_t alignment = Nektar::GetExecSpaceAlignment(execName);

    // Create blocks.
    auto blocks_in  = GetBlockAttributes<TData>(stateIn, expList);
    auto blocks_out = GetBlockAttributes<TData>(stateOut, expList);

    // Create fields.
    auto in =
        Field<TData, stateIn>("f_in", blocks_in, nIn * nComp, nHomo, alignment);
    auto out = Field<TData, stateOut>("f_out", blocks_out, nOut * nComp, nHomo,
                                      alignment);

    // Initialize the in field to random non-zeros: 1 2 3 4 ...
    auto &inblk = in.GetBlocks();
    for (size_t i = 0; i < inblk.size(); ++i)
    {
        auto inptr =
            inblk[i].template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
        for (unsigned int n = 0; n < nIn * nComp; n++)
        {
            for (size_t j = 0; j < inblk[i].size(); ++j)
            {
                inptr[j] = (j + (n + 1.0)) / inblk.size();
            }
            inptr += inblk.size();
        }
    }

    // Create input and output Array for explist.
    Array<OneD, NekDouble> inArr = in.template ToArray<NekDouble>();
    Array<OneD, NekDouble> outArr(out.size());
    Array<OneD, Array<OneD, NekDouble>> inArrays(nIn);
    Array<OneD, Array<OneD, NekDouble>> outArrays(nOut);
    for (unsigned int d = 0; d < nIn; d++)
    {
        inArrays[d] = inArr + d * inArr.size() / nIn / nComp;
    }
    for (unsigned int d = 0; d < nOut; d++)
    {
        outArrays[d] = outArr + d * outArr.size() / nOut / nComp;
    }

    // Get expected results from expList.
    GetExpectedResults(opName, expList, inArr, inArrays, outArr, outArrays);

    // Warm-up : fill the cache and memory, and let core temperature/freq
    // stabilized.
    for (unsigned int i = 0; i < Ntest / 2; ++i)
    {
        oper->Apply(in, out);
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
        oper->Apply(in, out);
    }

    nekDeviceSynchronize();
    LIKWID_MARKER_STOP(tag.c_str());
    timer.Stop();

    comm->Block();

    auto rankL1Error = std::vector<NekDouble>(1, 0.0);

    // Print block information and get the total number of dofs.
    if (comm->GetRank() == 0)
    {
        std::cout << "Input field: " << nComp * nIn << " components"
                  << std::endl;
    }
    PrintBlockInfo(expList, blocks_in, rankL1Error);

    // First check if the output is all zeros.
    TData L2;
    l2norm<NektarSpaces::Serial>(out, &L2);
    if (L2 < 1e-9)
    {
        std::cout << "Warning: output does not change!"
                  << "Device may not be invoked!" << std::endl;
    }

    // Then check if results match with expected
    // If we compare float results with double results, then it is
    // reasonable to have some mismatched values (e.g., > 1e-4)
    ReshapeToScalar(out);
    Array<OneD, NekDouble> tmpArr = out.template ToArray<NekDouble>();
    for (size_t i = 0, cnt = 0; i < tmpArr.size(); ++i)
    {
        if (opName == "LinAdvDiffReaction")
        {
            break;
        }

        // Print out first 100 mismatched values.
        if (abs(tmpArr[i] - outArr[i]) > 1e-4 && cnt < 100)
        {
            std::cout << "i=" << i << " computed result = " << tmpArr[i]
                      << " expected result = " << outArr[i] << std::endl;
            ++cnt;
        }
        rankL1Error[0] += abs(tmpArr[i] - outArr[i]);
    }

    // Print block information and get the total number of dofs.
    if (comm->GetRank() == 0)
    {
        std::cout << "Output field: " << nComp * nOut << " components"
                  << std::endl;
    }
    PrintBlockInfo(expList, blocks_out, rankL1Error);

    if (comm->GetRank() == 0)
    {
        std::cout << "---------------------------------" << std::endl;
        std::cout << "ElmtOps Profiler : " << tag << std::endl;
        std::cout << "---------------------------------" << std::endl;
    }
    comm->Block();

    // Collect elapsed time and compute the max, min, and average.
    auto rankElapsed = std::vector<NekDouble>(1, timer.TimePerTest(Ntest));
    PrintProfileResult<TData>(comm, rankElapsed, blocks_in, blocks_out);
}
