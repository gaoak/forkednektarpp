//////////////////////////////////////////////////////////////////////////////
//
// File: ProfilerElmtOps.cu
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

#include "ProfilerElmtOps.hpp"

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
    MultiRegions::ExpListSharedPtr contfield;

    session = LibUtilities::SessionReader::CreateInstance(argc, argv);
    graph   = SpatialDomains::MeshGraphIO::Read(session);

    // Load parameters (from the command lines)
    int Ntest, order, deformed_p;
    session->LoadParameter("Ntest", Ntest, 100);
    session->LoadParameter("order", order, 3);
    session->LoadParameter("deformed", deformed_p, 0);

    // check if verbose is set
    _verbose_ = session->DefinesCmdLineArgument("verbose");

    // Set the order of the polynomial expansion
    // we keep the point distribution the same and make
    // the number of points the same difference from the number
    // of modes as the original expansion definition
    graph->SetExpansionInfoToNumModes(order + 1);

    // create a contfield
    contfield = MemoryManager<MultiRegions::ContField>::AllocateSharedPtr(
        session, graph, "DefaultVar", true, false, Collections::eNoCollection);

    auto nDim = contfield->GetGraph()->GetSpaceDimension();

    // Print GPU properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "--------------------------------" << std::endl;
    std::cout << "Device Properties " << std::endl;
    std::cout << "--------------------------------" << std::endl;
    printf("  Device name: %s\n", prop.name);
    printf("  Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
    printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
    printf("  Peak Memory Bandwidth (GB/s): %f\n",
           2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
    printf("  Number of multiprocessors: %d\n", prop.multiProcessorCount);

    // Benchmark
    LaunchProfiler<BwdTrans<FP_t>, FieldState::Coeff, FieldState::Phys, FP_t>(
        contfield, Ntest, 1, 1);
    LaunchProfiler<IProductWRTBase<FP_t>, FieldState::Phys, FieldState::Coeff,
                   FP_t>(contfield, Ntest, 1, 1);
    LaunchProfiler<PhysDeriv<FP_t>, FieldState::Phys, FieldState::Phys, FP_t>(
        contfield, Ntest, 1, nDim);
    LaunchProfiler<IProductWRTDerivBase<FP_t>, FieldState::Phys,
                   FieldState::Coeff, FP_t>(contfield, Ntest, nDim, 1);
    LaunchProfiler<Helmholtz<FP_t>, FieldState::Coeff, FieldState::Coeff, FP_t>(
        contfield, Ntest, 1, 1);
    LaunchProfiler<Mass<FP_t>, FieldState::Coeff, FieldState::Coeff, FP_t>(
        contfield, Ntest, 1, 1);

    LIKWID_MARKER_CLOSE;

    session->Finalise();

#if defined(NEKTAR_ENABLE_KOKKOS)
    Kokkos::finalize();
#endif
}
