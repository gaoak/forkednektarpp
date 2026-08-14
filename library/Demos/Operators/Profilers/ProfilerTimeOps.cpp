//////////////////////////////////////////////////////////////////////////////
//
// File: ProfilerTimeOps.cpp
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

#include "LibUtilities/BasicUtils/Math/Math.hpp"
#include <SolverCore/TimeOps/TimeOp.hpp>

#include <LibUtilities/BasicUtils/Timer.h>
#include <MultiRegions/ExpList.h>
#include <SpatialDomains/MeshGraphIO.h>

using namespace Nektar;
using namespace Nektar::LibUtilities;
using namespace Nektar::MultiRegions;
using namespace Nektar::Operators;
using namespace Nektar::SolverCore;

class DummySolver
{
public:
    DummySolver(const std::string execName)
    {
        math = Math::MathHelper(execName);
    }

    void DoLHS(Field<double, FieldState::Phys> &in,
               Field<double, FieldState::Phys> &out,
               [[maybe_unused]] const double &time, const double &lambda)
    {
        // Factor for implicit/stiff part of analytic test problem
        auto factor = 1.0 / (1.0 - lambda * m_beta);

        // Multiply extrapolated rhs
        math.mul(factor, in, out);
    }

    void DoExplicitRHS(Field<double, FieldState::Phys> &in,
                       Field<double, FieldState::Phys> &out,
                       [[maybe_unused]] const double &time,
                       const double &factor)
    {
        // Multiply solution by factor
        math.mul(m_alpha * factor, in, out);
    }

    void DoImplicitRHS(Field<double, FieldState::Phys> &in,
                       Field<double, FieldState::Phys> &out,
                       [[maybe_unused]] const double &time,
                       const double &factor)
    {
        // Multiply solution by factor
        math.mul(m_beta * factor, in, out);
    }

    void DoProjection(Field<double, FieldState::Phys> &in,
                      Field<double, FieldState::Phys> &out,
                      [[maybe_unused]] const double &time)
    {
        // Multiply solution by factor
        math.mul(1.0, in, out);
    }

protected:
    double m_alpha = 1.0;
    double m_beta  = -1.0;

    Math::MathHelper math;
};

int main(int argc, char *argv[])
{
    typedef double TData;

    // Initialise a session, graph and explist.
    auto session = LibUtilities::SessionReader::CreateInstance(argc, argv);
    auto graph   = SpatialDomains::MeshGraphIO::Read(session);

    // Load parameters (from the command lines).
    int nTimeStep, timeOrder, order;
    std::string method;
    session->LoadSolverInfo("method", method, "BDFImplicit");
    session->LoadParameter("timeOrder", timeOrder, 1);
    session->LoadParameter("nTimeStep", nTimeStep, 100);
    session->LoadParameter("order", order, 0);

    // Set the order of the polynomial expansion if provided
    // we keep the point distribution the same and make
    // the number of points the same difference from the number
    // of modes as the original expansion definition.
    if (order > 0)
    {
        graph->SetExpansionInfoToNumModes(order + 1);
    }

    // Create a ExpList from the graph(mesh).
    auto expList = MemoryManager<MultiRegions::ExpList>::AllocateSharedPtr(
        session, graph, true, "u", Collections::eNoCollection);

    expList->SetDataWarehouse();

    // Timer.
    Timer timer;

    // Create block attributes.
    auto blockAttr = GetBlockAttributes<TData, FieldState::Phys>(expList);

    // Create fields.
    auto fIn = Field<TData, FieldState::Phys>("f_in", blockAttr,
                                              session->GetVariables(), 1);

    // Set random output.
    srand(0);
    auto &blockIn = fIn.GetBlocks();
    for (size_t i = 0; i < blockIn.size(); ++i)
    {
        auto inPtr =
            blockIn[i].template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
        for (unsigned int n = 0; n < session->GetVariables().size(); n++)
        {
            for (size_t j = 0; j < blockIn[i].CompSize(); ++j)
            {
                inPtr[j] =
                    ((static_cast<double>(rand()) / RAND_MAX) * 2.0 - 1.0);
            }
        }
    }

    double timestep = 0.0001;
    session->SetParameter("TimeStep", timestep);
    std::string execName = Operator<double>::GetOpExecSpace(session);
    auto solver          = DummySolver(execName);

    auto timeOp = TimeOp<TData>::Create(expList, {"u"}, method, timeOrder);
    timeOp->DefineExplicitRhs(&DummySolver::DoExplicitRHS, &solver);
    timeOp->DefineImplicitRhs(&DummySolver::DoImplicitRHS, &solver);
    timeOp->DefineImplicit(&DummySolver::DoLHS, &solver);

    // Warm up solves.
    for (unsigned int i = 0; i < nTimeStep / 2; ++i)
    {
        timeOp->Apply(fIn);
    }

    // Benchmark CG solve.
    nekDeviceSynchronize();
    timer.Start();
    for (unsigned int i = 0; i < nTimeStep; ++i)
    {
        timeOp->Apply(fIn);
    }
    nekDeviceSynchronize();

    timer.Stop();

    // Output results.
    unsigned int expOrder =
        (*expList->GetExp())[0]->GetBase()[0]->GetNumModes();
    size_t numDofs = expList->GetNcoeffs();
    double time    = timer.Elapsed().count() / nTimeStep;

    std::cout << "Order: " << expOrder << std::endl;
    std::cout << "Total ndof: " << numDofs << std::endl;
    std::cout << "Throughput (ndof/s): " << numDofs / time << std::endl;
    std::cout << "Throughput (GB/s): " << numDofs / time * sizeof(TData) / 1e9
              << std::endl;
    std::cout << std::endl;
}
