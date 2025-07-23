//////////////////////////////////////////////////////////////////////////////
//
// File: BP1.cpp
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
// Description: executable source file
//
///////////////////////////////////////////////////////////////////////////////

#include "Demos/Operators/Profilers/ProfilerElmtOps.hpp"
#include "Operators/AssmbScatr/AssmbScatrOp.hpp"
#include "Operators/GlobalLinSysOps/ConjGrad/ConjGradOp.hpp"
#include "Operators/GlobalLinSysOps/FwdTrans/FwdTransOp.hpp"
#include "Operators/PreconOps/DiagPrecon/DiagPreconOp.hpp"

#include <LibUtilities/BasicUtils/Timer.h>
#include <MultiRegions/ContField.h>
#include <MultiRegions/ExpList.h>
#include <SpatialDomains/MeshGraphIO.h>
using namespace Nektar::Operators;
using namespace Nektar::LibUtilities;
using namespace Nektar;

int main(int argc, char *argv[])
{
    typedef double tData;
    constexpr FieldState stateIn  = FieldState::Coeff;
    constexpr FieldState stateOut = FieldState::Coeff;
    int nIn                       = 1;
    int nOut                      = 1;
    int nComp                     = 1;
    // Initialise a session, graph and explist.
    auto session = LibUtilities::SessionReader::CreateInstance(argc, argv);
    auto graph   = SpatialDomains::MeshGraphIO::Read(session);

    // Load parameters (from the command lines).
    int nTest, order;
    session->LoadParameter("Ntest", nTest, 100);
    session->LoadParameter("order", order, 0);
    session->LoadParameter("Ncomp", nComp, 1);

    // Set the order of the polynomial expansion if provided
    // we keep the point distribution the same and make
    // the number of points the same difference from the number
    // of modes as the original expansion definition.
    if (order > 0)
    {
        graph->SetExpansionInfoToNumModes(order + 1);
    }

    // Create a ExpList from the graph(mesh).
    auto expList = MemoryManager<MultiRegions::ContField>::AllocateSharedPtr(
        session, graph, "DefaultVar", true, false, Collections::eNoCollection);

    expList->SetDataWarehouse();

    auto nDim = expList->GetGraph()->GetSpaceDimension();

    auto massOp       = MassOp<double>::Create(expList);
    auto diagPreconOp = Operators::DiagPreconOp<double>::Create(expList);
    diagPreconOp->Configure(massOp);
    auto conjGradOp = Operators::ConjGradOp<double>::Create(expList);
    conjGradOp->SetLHS(massOp);
    conjGradOp->SetPrecon(diagPreconOp);
    // Timer.
    Timer timer;

    // Create operator.
    auto oper = conjGradOp;

    // Set operator name tag.
    std::string execName =
        session->GetCmdLineArgument<std::string>("opExecSpace");
    std::string implName = session->GetCmdLineArgument<std::string>("opImpl");
    std::string opName   = oper->name;
    std::string dataType = (std::is_same_v<tData, double>) ? "Double" : "Float";
    auto tag             = opName + execName + implName + dataType;

    // Set alignment.
    size_t alignment = Nektar::GetExecSpaceAlignment(execName);

    // Create blocks.
    auto blocksIn               = GetBlockAttributes<tData>(stateIn, expList);
    auto blocksOut              = GetBlockAttributes<tData>(stateOut, expList);
    auto blocksOutCorrect       = GetBlockAttributes<tData>(stateOut, expList);
    auto blocksOutCorrectAssemb = GetBlockAttributes<tData>(stateOut, expList);

    // Create fields.
    auto fIn  = Field<tData, stateIn>::Create("f_in", blocksIn, nIn * nComp, 1,
                                              alignment);
    auto fOut = Field<tData, stateOut>::Create("f_out", blocksOut, nOut * nComp,
                                               1, alignment);
    auto fOutCorrect = Field<tData, stateOut>::Create(
        "f_out_correct", blocksOutCorrect, nOut * nComp, 1, alignment);
    auto fOutCorrectAssemb = Field<tData, stateOut>::Create(
        "f_out_correct_assemb", blocksOutCorrectAssemb, nOut * nComp, 1,
        alignment);

    // Set random output
    srand(0);
    auto blockOut = fOutCorrect.GetBlocks();
    for (size_t i = 0; i < blockOut.size(); ++i)
    {
        auto outPtr =
            blockOut[i].template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
        for (unsigned int n = 0; n < nOut * nComp; n++)
        {
            for (size_t j = 0; j < blockOut[i].size(); ++j)
            {
                outPtr[j] =
                    ((static_cast<double>(rand()) / RAND_MAX) * 2.0 - 1.0);
            }
        }
    }

    // Ensure C0 continuity
    auto assembOp = AssmbScatrOp<tData>::Create(expList);
    assembOp->Apply(fOutCorrect, fOutCorrectAssemb);

    // Compute expected solution
    massOp->Apply(fOutCorrectAssemb, fIn);

    // Warm up solves
    for (unsigned int i = 0; i < nTest / 2; ++i)
    {
        oper->Apply(fIn, fOut);
    }

    // Benchmark CG solve
#if defined(NEKTAR_ENABLE_CUDA)
    CHECK_HIPCUDA_ERROR(cudaDeviceSynchronize());
#elif defined(NEKTAR_ENABLE_HIP)
    CHECK_HIPCUDA_ERROR(hipDeviceSynchronize());
#elif defined(NEKTAR_ENABLE_SYCL)
    SYCLQueue::GetInstance().wait();
#endif
    timer.Start();
    for (unsigned int i = 0; i < nTest; ++i)
    {
        oper->Apply(fIn, fOut);
    }
#if defined(NEKTAR_ENABLE_CUDA)
    CHECK_HIPCUDA_ERROR(cudaDeviceSynchronize());
#elif defined(NEKTAR_ENABLE_HIP)
    CHECK_HIPCUDA_ERROR(hipDeviceSynchronize());
#elif defined(NEKTAR_ENABLE_SYCL)
    SYCLQueue::GetInstance().wait();
#endif
    timer.Stop();

    // output results
    long numElmts      = expList->GetNumElmts();
    long numDofs       = expList->GetNcoeffs();
    auto expOrder      = (*expList->GetExp())[0]->GetBase()[0]->GetNumModes();
    double elmtsPerDim = std::pow(numElmts, 1.0 / nDim);
    long globalDOFs =
        std::pow((elmtsPerDim - 1) * (expOrder - 1) + expOrder, 3);
    double timerS   = timer.Elapsed().count();
    double sPerIter = timerS / (nTest * 5000);

    std::cout << expOrder << " " << numElmts << " " << globalDOFs << " "
              << numDofs << " " << sPerIter << std::endl;
}
