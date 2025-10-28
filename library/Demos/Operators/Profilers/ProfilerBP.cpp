//////////////////////////////////////////////////////////////////////////////
//
// File: ProfilerBP.cpp
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

#include "Operators/AssmbScatr/AssmbScatrOp.hpp"
#include "Operators/GlobalLinSysOps/ConjGrad/ConjGradOp.hpp"
#include "Operators/GlobalLinSysOps/FwdTrans/FwdTransOp.hpp"
#include "Operators/PreconOps/DiagPrecon/DiagPreconOp.hpp"
#include <Operators/ElmtOps/Helmholtz/HelmholtzOp.hpp>
#include <Operators/ElmtOps/Mass/MassOp.hpp>

#include <LibUtilities/BasicUtils/Timer.h>
#include <MultiRegions/ContField.h>
#include <MultiRegions/ExpList.h>
#include <SpatialDomains/MeshGraphIO.h>

using namespace Nektar::Operators;
using namespace Nektar::LibUtilities;
using namespace Nektar;

int main(int argc, char *argv[])
{
    typedef double TData;
    int nIn   = 1;
    int nOut  = 1;
    int nComp = 1;

    // Initialise a session, graph and explist.
    auto session = LibUtilities::SessionReader::CreateInstance(argc, argv);
    auto graph   = SpatialDomains::MeshGraphIO::Read(session);

    // Load parameters (from the command lines).
    int BP, nTest, order;
    session->LoadParameter("BP", BP, 1);
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

    // Initialize operators.
    std::shared_ptr<ElmtOp<FieldState::Coeff, FieldState::Coeff, TData>> elmtOp;
    if (BP == 1)
    {
        elmtOp = MassOp<TData>::Create(expList);
    }
    else if (BP == 3)
    {
        elmtOp = HelmholtzOp<TData>::Create(expList);
    }
    auto assembOp     = AssmbScatrOp<TData>::Create(expList);
    auto diagPreconOp = DiagPreconOp<TData>::Create(expList);
    auto conjGradOp   = ConjGradOp<TData>::Create(expList);
    diagPreconOp->Configure(elmtOp);
    conjGradOp->SetLHS(elmtOp);
    conjGradOp->SetPrecon(diagPreconOp);

    // Timer.
    Timer timer;

    // Set alignment.
    std::string execName =
        session->GetCmdLineArgument<std::string>("opExecSpace");
    size_t alignment = Nektar::GetExecSpaceAlignment(execName);

    // Create blocks.
    auto blocksIn  = GetBlockAttributes<TData>(FieldState::Coeff, expList);
    auto blocksOut = GetBlockAttributes<TData>(FieldState::Coeff, expList);
    auto blocksOutCorrect =
        GetBlockAttributes<TData>(FieldState::Coeff, expList);
    auto blocksOutCorrectAssemb =
        GetBlockAttributes<TData>(FieldState::Coeff, expList);

    // Create fields.
    auto fIn = Field<TData, FieldState::Coeff>("f_in", blocksIn, nIn * nComp, 1,
                                               alignment);
    auto fOut        = Field<TData, FieldState::Coeff>("f_out", blocksOut,
                                                nOut * nComp, 1, alignment);
    auto fOutCorrect = Field<TData, FieldState::Coeff>(
        "f_out_correct", blocksOutCorrect, nOut * nComp, 1, alignment);
    auto fOutCorrectAssemb = Field<TData, FieldState::Coeff>(
        "f_out_correct_assemb", blocksOutCorrectAssemb, nOut * nComp, 1,
        alignment);

    // Set random output.
    srand(0);
    auto &blockOut = fOutCorrect.GetBlocks();
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

    // Ensure C0 continuity.
    assembOp->Apply(fOutCorrect, fOutCorrectAssemb);

    // Compute expected solution.
    elmtOp->Apply(fOutCorrectAssemb, fIn);

    // Warm up solves.
    for (unsigned int i = 0; i < nTest / 2; ++i)
    {
        conjGradOp->Apply(fIn, fOut);
    }

    // Benchmark CG solve.
    nekDeviceSynchronize();
    timer.Start();
    for (unsigned int i = 0; i < nTest; ++i)
    {
        conjGradOp->Apply(fIn, fOut);
    }
    nekDeviceSynchronize();

    timer.Stop();

    // Output results.
    unsigned int expOrder =
        (*expList->GetExp())[0]->GetBase()[0]->GetNumModes();
    size_t numElmts    = expList->GetNumElmts();
    size_t numDofs     = expList->GetNcoeffs();
    size_t elmtsPerDim = std::pow(numElmts, 1.0 / nDim);
    size_t globalDOFs =
        std::pow((elmtsPerDim - 1) * (expOrder - 1) + expOrder, 3);
    double time        = timer.Elapsed().count();
    double timePerIter = time / (nTest * 5000);

    std::cout << expOrder << " " << numElmts << " " << globalDOFs << " "
              << numDofs << " " << timePerIter << std::endl;
}
