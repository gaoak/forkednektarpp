//////////////////////////////////////////////////////////////////////////////
//
// File: ProfilerElmtOps.cpp
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

#include "ProfilerElmtOps.hpp"

/**
 * @brief main program of the profiler
 *
 * Functions:
 *
 *  Benchmark/profile the elemental operators and print out the
 *  performance statistics, e.g. elapsed time, throughput, ndofs. Users can
 *  customize which operators, implementations(StdMat, SumFac), execution
 *  spaces(AVX, CUDA, SYCL), and data types(float, doubule) to be profiled.
 *
 * Usage:
 *
 *  1.  Build the Nektar++ project with following configurations to get the
 *      profiler executable:
 *          NEKTAR_BUILD_DEMOS=ON
 *          NEKTAR_BUILD_REDESIGN=ON
 *      You may need other configurations to enable certain features, e.g. CUDA,
 *      AVX, MPI, etc.
 *
 *  2.  Prepare a mesh file for the profiling, e.g. mesh.xml. Make sure it has
 *      enough elements to get meaningful profiling results. Make sure the file
 *      contains correct EXPANSIONS definitions.
 *
 *  3.  Run the profiler executable with the mesh file and other parameters.
 *      -P Ntest=100
 *              number of repeated runs for each operator. Usually a operator
 *              takes very short time to finish, so we need to repeat it many
 *              times to get accurate timing, and also let CPU/GPU running at
 *              a stable frequency.
 *      -P order=5
 *              the order of the polynomial expansions. If provided, it will
 *              override the expansion definition in the mesh file. This gives
 *              a convenient way to profile different orders by same mesh file.
 *      -verbose
 *              print out more information. Not recommended if you launch many
 *              processes.
 *
 *      Examples:
 *
 *      # So far We can only launch serial run on GPUs:
 *          ./ProfilerElmtOps mesh.xml -P Ntest=100 -P order=5 -verbose
 *
 *      # To squeeze all the performance of CPU, typically we launch as many
 *      # processes as the number of cores on a machine:
 *          mpirun -np 12 ./ProfilerElmtOps mesh.xml -P Ntest=200 -P order=3
 *
 *      # Launch likwid and use 18 processes per socket(CPU package), MEM_DP
 *      # tells likwid to measure memory and flops performance, This is
 *      # usually for a roofline analysis:
 *          likwid-mpirun -nperdomain S:18 -m -g MEM_DP ./ProfilerElmtOps
 * mesh.xml
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
#if defined(NEKTAR_ENABLE_KOKKOS)
    Kokkos::initialize();
#endif
    LIKWID_MARKER_INIT;
    LIKWID_MARKER_THREADINIT;

    // Initialise a session, graph and explist
    LibUtilities::SessionReaderSharedPtr session;
    SpatialDomains::MeshGraphSharedPtr graph;
    MultiRegions::ExpListSharedPtr explist;

    session = LibUtilities::SessionReader::CreateInstance(argc, argv);
    graph   = SpatialDomains::MeshGraphIO::Read(session);

    // Load parameters (from the command lines)
    int Ntest, order;
    session->LoadParameter("Ntest", Ntest, 100);
    session->LoadParameter("order", order, 0);

    // check if verbose is set
    _verbose_ = session->DefinesCmdLineArgument("verbose");

    // Set the order of the polynomial expansion if provided
    // we keep the point distribution the same and make
    // the number of points the same difference from the number
    // of modes as the original expansion definition
    if (order > 0)
    {
        graph->SetExpansionInfoToNumModes(order + 1);
    }

    // create a ExpList from the graph(mesh)
    explist = MemoryManager<MultiRegions::ExpList>::AllocateSharedPtr(
        session, graph, true, "DefaultVar", Collections::eNoCollection);

    explist->SetDataWarehouse();

    auto nDim = explist->GetGraph()->GetSpaceDimension();

    // You can add/remove the operators to be profiled as you like.
    // Benchmark-double
    LaunchProfiler<BwdTrans<double>, FieldState::Coeff, FieldState::Phys,
                   double>(explist, Ntest, 1, 1);
    LaunchProfiler<IProductWRTBase<double>, FieldState::Phys, FieldState::Coeff,
                   double>(explist, Ntest, 1, 1);
    LaunchProfiler<PhysDeriv<double>, FieldState::Phys, FieldState::Phys,
                   double>(explist, Ntest, 1, nDim);
    LaunchProfiler<IProductWRTDerivBase<double>, FieldState::Phys,
                   FieldState::Coeff, double>(explist, Ntest, nDim, 1);
    LaunchProfiler<Helmholtz<double>, FieldState::Coeff, FieldState::Coeff,
                   double>(explist, Ntest, 1, 1);
    LaunchProfiler<Mass<double>, FieldState::Coeff, FieldState::Coeff, double>(
        explist, Ntest, 1, 1);

#if defined(NEKTAR_ENABLE_SINGLE_PRECISION)
    // Benchmark-float
    LaunchProfiler<BwdTrans<float>, FieldState::Coeff, FieldState::Phys, float>(
        explist, Ntest, 1, 1);
    LaunchProfiler<IProductWRTBase<float>, FieldState::Phys, FieldState::Coeff,
                   float>(explist, Ntest, 1, 1);
    LaunchProfiler<PhysDeriv<float>, FieldState::Phys, FieldState::Phys, float>(
        explist, Ntest, 1, nDim);
    LaunchProfiler<IProductWRTDerivBase<float>, FieldState::Phys,
                   FieldState::Coeff, float>(explist, Ntest, nDim, 1);
    LaunchProfiler<Helmholtz<float>, FieldState::Coeff, FieldState::Coeff,
                   float>(explist, Ntest, 1, 1);
    LaunchProfiler<Mass<float>, FieldState::Coeff, FieldState::Coeff, float>(
        explist, Ntest, 1, 1);
#endif

    LIKWID_MARKER_CLOSE;

    session->Finalise();

#if defined(NEKTAR_ENABLE_KOKKOS)
    Kokkos::finalize();
#endif
}
