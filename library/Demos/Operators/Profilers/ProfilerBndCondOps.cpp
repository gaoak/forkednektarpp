//////////////////////////////////////////////////////////////////////////////
//
// File: ProfilerBndCondOps.cpp
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

#include "ProfilerBndCondOps.hpp"

#include <sstream>

#if defined(NEKTAR_USE_MAGMA)
#include "magma_v2.h"
#endif

/**
 * @brief main program of the profiler
 *
 * Functions:
 *
 *  Benchmark/profile the DirBndCondOp (Dirichlet lifting) and NeuBndCondOp
 *  (weak Neumann forcing) continuous Galerkin boundary condition operators,
 *  and print out the performance statistics, e.g. elapsed time, throughput,
 *  ndofs. Users can customize which operators and data types (float,
 *  double) to be profiled. Optionally (timeDependent=1), the per-timestep
 *  cost of re-evaluating time-dependent boundary conditions can be
 *  profiled by calling UpdateBndCoeffs() inside the timed loop instead of
 *  once up front.
 *
 * Usage:
 *
 *  1.  Build the Nektar++ project with following configurations to get the
 *      profiler executable:
 *          NEKTAR_BUILD_DEMOS=ON
 *          NEKTAR_ENABLE_DEVICE_SUPPORT=ON
 *      You may need other configurations to enable certain features, e.g. CUDA,
 *      AVX, MPI, etc.
 *
 *  2.  Prepare a mesh file for the profiling, e.g. mesh.xml. Make sure it has
 *      enough elements to get meaningful profiling results, and that it
 *      defines Dirichlet and/or Neumann boundary conditions (with non-zero
 *      expressions) on the session variable(s) so the operators being
 *      profiled actually have boundary coefficients to process. The
 *      Helmholtz test meshes under library/UnitTests/Operators/run (e.g.
 *      Helmholtz3D_Hex_AllBCs_P6.xml) are good starting points.
 *
 *  3.  Run the profiler executable with the mesh file and other parameters.
 *      --opExecSpace=Serial
 *              specify the execution space. Possible values are: Serial, AVX,
 *              and Device
 *      --opImpl=StdMat
 *              DirBndCondOp/NeuBndCondOp internally evaluate the boundary
 *              condition expressions with an ExpressionOp (an element
 *              operator), which requires an implementation to be selected
 *              the same way as in ProfilerElmtOps. Possible values are:
 *              StdMat, SumFac, and SumFacTOP.
 *      -P Ntest=100
 *              number of repeated runs for each operator. Usually a operator
 *              takes very short time to finish, so we need to repeat it many
 *              times to get accurate timing, and also let CPU/GPU running at
 *              a stable frequency.
 *      -P order=5
 *              the order of the polynomial expansions. If provided, it will
 *              override the expansion definition in the mesh file. This gives
 *              a convenient way to profile different orders by same mesh file.
 *      -P time=0.0
 *              starting time level at which to evaluate boundary condition
 *              expressions.
 *      -P timeDependent=0
 *              if non-zero, profile the realistic per-timestep cost of a
 *              time-dependent boundary condition: instead of calling
 *              UpdateBndCoeffs() once up front and then timing Apply()
 *              alone, UpdateBndCoeffs(t) is called on every warm-up/timed
 *              iteration, with t advancing by `dt` each call, immediately
 *              followed by Apply(). This requires a mesh whose boundary
 *              conditions are actually time dependent (e.g.
 *              Helmholtz3D_Hex_AllBCs_P6_TimeDependentBC.xml) -- on a
 *              mesh with only static boundary conditions,
 *              UpdateBndCoeffs() early-returns cheaply regardless of `t`,
 *              so this mostly measures that short-circuit check's
 *              overhead instead.
 *      -P dt=0.001
 *              time step used to advance `time` between iterations when
 *              timeDependent=1 is set. Unused otherwise.
 *      -I operators=DirBndCond,NeuBndCond
 *              specify the operators for the benchmark. If not specified,
 *              both DirBndCond and NeuBndCond will be benchmarked.
 *      -verbose
 *              print out more information. Not recommended if you launch many
 *              processes.
 *
 *      Examples:
 *
 *      # So far We can only launch one GPU:
 *          ./ProfilerBndCondOps mesh.xml --opExecSpace=Device --opImpl=StdMat
 *          -P Ntest=100 -P order=5 -verbose
 *
 *      # To squeeze all the performance of CPU, typically we launch as many
 *      # processes as the number of cores on a machine:
 *          mpirun -np 12 ./ProfilerBndCondOps mesh.xml --opExecSpace=Serial
 * --opImpl=StdMat -P Ntest=200 -P order=3
 *
 *      # Profile the per-timestep cost of re-evaluating time-dependent
 *      # boundary conditions, on a mesh that actually has them:
 *          ./ProfilerBndCondOps Helmholtz3D_Hex_AllBCs_P6_TimeDependentBC.xml
 *          --opExecSpace=Serial --opImpl=StdMat -P Ntest=200 -P timeDependent=1
 *          -P dt=0.001
 *
 *      # Launch likwid and use 18 processes per socket(CPU package), MEM_DP
 *      # tells likwid to measure memory and flops performance, This is
 *      # usually for a roofline analysis:
 *          likwid-mpirun -nperdomain S:18 -m -g MEM_DP ./ProfilerBndCondOps
 *          --opExecSpace=Serial --opImpl=StdMat mesh.xml
 *
 *  4.  likwid is a powerful tool to measure the performance of the CPU/GPI,
 *      including memory bandwidth, flops, cache misses and stalls, etc. The
 *      profiler already has the likwid API integrated. To use likwid. You need
 *      to first ensure likwid is properly installed and then build Nektar with
 *          NEKTAR_USE_LIKWID=ON.
 *      See likwid documentation for more information.
 */
int main(int argc, char *argv[])
{
#ifdef NEKTAR_USE_MAGMA
    magma_init();
#endif

    LIKWID_MARKER_INIT;
    LIKWID_MARKER_THREADINIT;

    // Initialise a session, graph and explist.
    auto session = LibUtilities::SessionReader::CreateInstance(argc, argv);
    auto graph   = SpatialDomains::MeshGraphIO::Read(session);

    // Load parameters (from the command lines).
    int Ntest, order, timeDependentFlag;
    double time, dt;
    std::string Operators;
    session->LoadParameter("Ntest", Ntest, 100);
    session->LoadParameter("order", order, 0);
    session->LoadParameter("time", time, 0.0);
    session->LoadParameter("timeDependent", timeDependentFlag, 0);
    session->LoadParameter("dt", dt, 1e-3);
    session->LoadSolverInfo("operators", Operators, "");
    const bool timeDependent = timeDependentFlag != 0;

    // Set the order of the polynomial expansion if provided
    // we keep the point distribution the same and make
    // the number of points the same difference from the number
    // of modes as the original expansion definition.
    if (order > 0)
    {
        graph->SetExpansionInfoToNumModes(order + 1);
    }

    const auto &variables = session->GetVariables();
    ASSERTL0(!variables.empty(), "No variables defined in session file.");

    // DirBndCondOp and NeuBndCondOp are continuous Galerkin operators: the
    // Dirichlet lifting resolves duplicated/parallel boundary dofs through
    // the assembly map's universal gather, which requires a ContField.
    auto explist = MemoryManager<MultiRegions::ContField>::AllocateSharedPtr(
        session, graph, variables[0], true, false, Collections::eNoCollection);

    explist->SetDataWarehouse();

    // Print GPU properties.
    if (session->GetComm()->GetRank() == 0 &&
        Operator<double>::GetOpExecSpace(session) == "Device")
    {
        PrintDeviceProperties();
    }

    std::vector<std::string> Operators0;

    if (Operators.empty())
    {
        Operators0 = {
            "DirBndCond",
            "NeuBndCond",
        };
    }
    else
    {
        std::stringstream ss(Operators);
        std::string item;
        while (std::getline(ss, item, ','))
        {
            // trim spaces
            item.erase(0, item.find_first_not_of(" \t"));
            item.erase(item.find_last_not_of(" \t") + 1);
            if (!item.empty())
            {
                Operators0.push_back(item);
            }
        }
    }

    // Benchmark-double
#if defined(NEKTAR_ENABLE_DOUBLE_PRECISION)
    for (auto &Operator : Operators0)
    {
        if (Operator == "DirBndCond")
        {
            LaunchProfiler<DirBndCondOp, double>(explist, Ntest, time,
                                                 timeDependent, dt);
        }
        else if (Operator == "NeuBndCond")
        {
            LaunchProfiler<NeuBndCondOp, double>(explist, Ntest, time,
                                                 timeDependent, dt);
        }
    }
#endif
#if defined(NEKTAR_ENABLE_SINGLE_PRECISION)
    // Benchmark-float
    for (auto &Operator : Operators0)
    {
        if (Operator == "DirBndCond")
        {
            LaunchProfiler<DirBndCondOp, float>(explist, Ntest, time,
                                                timeDependent, dt);
        }
        else if (Operator == "NeuBndCond")
        {
            LaunchProfiler<NeuBndCondOp, float>(explist, Ntest, time,
                                                timeDependent, dt);
        }
    }
#endif

    LIKWID_MARKER_CLOSE;

    session->Finalise();

#ifdef NEKTAR_USE_MAGMA
    magma_finalize();
#endif
}
