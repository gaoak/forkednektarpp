///////////////////////////////////////////////////////////////////////////////
//
// File: Operator.cpp
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

#include "Operators/Operator.hpp"

#include <LibUtilities/BasicUtils/SessionReader.h>

namespace Nektar::Operators
{

#if defined(NEKTAR_ENABLE_SIMD_AVX2) || defined(NEKTAR_ENABLE_SIMD_AVX512)
    std::string g_ExecSpace = "AVX ";
#elif defined(NEKTAR_ENABLE_CUDA)
    std::string g_ExecSpace = "CUDA ";
#elif defined(NEKTAR_ENABLE_HIP)
    std::string g_ExecSpace = "Hip";
#elif defined(NEKTAR_ENABLE_KOKKOS)
    std::string g_ExecSpace = "Kokkos";
#else
    std::string g_ExecSpace = "Serial ";
#endif


#if defined(NEKTAR_ENABLE_CUDA) || defined(NEKTAR_ENABLE_HIP) ||  \
    defined(NEKTAR_ENABLE_SIMD_AVX2) || defined(NEKTAR_ENABLE_SIMD_AVX512)
    std::string g_Impl = {"SumFac "};
#else
    std::string g_Impl = {"StdMat "};
#endif


  
std::string operatorExecSpace =
    LibUtilities::SessionReader::RegisterCmdLineArgument(
        "opExecSpace", "",
        "Operator default ExecSpace - "
        "Serial "
#if !defined(NEKTAR_ENABLE_CUDA) && !defined(NEKTAR_ENABLE_HIP) &&  \
    !defined(NEKTAR_ENABLE_KOKKOS) &&  \
    !defined(NEKTAR_ENABLE_SIMD_AVX2) && !defined(NEKTAR_ENABLE_SIMD_AVX512)
        "(default)"
#endif
#if defined(NEKTAR_ENABLE_SIMD_AVX2) || defined(NEKTAR_ENABLE_SIMD_AVX512)
        ", AVX "
        "(default)"
#endif
#if defined(NEKTAR_ENABLE_CUDA)
        ", CUDA "
        "(default)"
#endif
#if defined(NEKTAR_ENABLE_HIP)
        ", Hip"
        "(default)"
#endif
#if defined(NEKTAR_ENABLE_KOKKOS)
        ", Kokkos"
        "(default)"
#endif
                                                         );

std::string operatorImpl =
    LibUtilities::SessionReader::RegisterCmdLineArgument(
        "opImpl", "",
        "Operator default Implementation - "
        "StdMat "
#if !defined(NEKTAR_ENABLE_CUDA) && !defined(NEKTAR_ENABLE_HIP) &&  \
    !defined(NEKTAR_ENABLE_SIMD_AVX2) && !defined(NEKTAR_ENABLE_SIMD_AVX512)
        "(default)"
#endif
        "SumFac "
#if defined(NEKTAR_ENABLE_CUDA) || defined(NEKTAR_ENABLE_CUDA) ||  \
    defined(NEKTAR_ENABLE_SIMD_AVX2) || defined(NEKTAR_ENABLE_SIMD_AVX512)
        "(default)"
#endif
                                                         );

#if defined(NEKTAR_ENABLE_KOKKOS)
std::string kokkosCmdPolicy =
    LibUtilities::SessionReader::RegisterCmdLineArgument(
        "kokkos_policy", "",
        "Kokkos Execution Policy - range (default), mdrange, team");

std::string kokkosCmdLeaguesPerLoop =
    LibUtilities::SessionReader::RegisterCmdLineArgument(
        "kokkos_leagues_per_loop", "",
        "Kokkos TeamPolicy number of leagues (work items)"
        "per loop (default 1).");

std::string kokkosCmdTeamsPerLeague =
    LibUtilities::SessionReader::RegisterCmdLineArgument(
        "kokkos_teams_per_league", "",
        "Kokkos TeamPolicy number of teams (threads) "
        "per Kokkos TeamPolicy league (default 256/16).");

std::string kokkosCmdChunkSize =
    LibUtilities::SessionReader::RegisterCmdLineArgument(
        "kokkos_chunk_size", "",
        "Kokkos TeamPolicy and RangePolicy chunk size.");

std::string kokkosCmdTileSize =
    LibUtilities::SessionReader::RegisterCmdLineArgument(
        "kokkos_tile_size", "",
        "Kokkos MDRangePolicy tile size.");
#endif

template <typename TData> OperatorFactory<TData> &GetOperatorFactory()
{
    static OperatorFactory<TData> instance;
    return instance;
}

template OperatorFactory<double> &GetOperatorFactory();

} // namespace Nektar::Operators
