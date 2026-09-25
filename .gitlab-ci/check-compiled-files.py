###############################################################################
##
## File: check-compiled-files.py
##
## For more information, please see: http://www.nektar.info
##
## The MIT License
##
## Copyright (c) 2006 Division of Applied Mathematics, Brown University (USA),
## Department of Aeronautics, Imperial College London (UK), and Scientific
## Computing and Imaging Institute, University of Utah (USA).
##
## Permission is hereby granted, free of charge, to any person obtaining a
## copy of this software and associated documentation files (the "Software"),
## to deal in the Software without restriction, including without limitation
## the rights to use, copy, modify, merge, publish, distribute, sublicense,
## and/or sell copies of the Software, and to permit persons to whom the
## Software is furnished to do so, subject to the following conditions:
##
## The above copyright notice and this permission notice shall be included
## in all copies or substantial portions of the Software.
##
## THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
## OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
## FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
## THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
## LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
## FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
## DEALINGS IN THE SOFTWARE.
##
## Description: Check for compiled files using compile_commands.json
##
###############################################################################

import json, sys, glob, os

# Should be run from root source directory.
cwd = os.getcwd()

# Folders and extensions to check
folders = ['library', 'solvers', 'tests', 'utilities']
exts = ['cpp', 'c']

# Presently some stuff is not in the CI.
ignore_sources = [
    # CADfix API. The CAD classes moved from NekMesh to SpatialDomains; the
    # two NekMesh modules are additionally not yet ported to build against
    # SpatialDomains geometry, and NEKTAR_USE_CFI refuses to configure until
    # they are.
    "library/SpatialDomains/CADSystem/CFI/CADSystemCFI.cpp",
    "library/SpatialDomains/CADSystem/CFI/CADSurfCFI.cpp",
    "library/SpatialDomains/CADSystem/CFI/CADCurveCFI.cpp",
    "library/SpatialDomains/CADSystem/CFI/CADVertCFI.cpp",
    "library/NekMesh/Module/InputModules/InputCADfix.cpp",
    "library/NekMesh/Module/OutputModules/OutputCADfix.cpp",
    # Likwid
    "solvers/CompressibleFlowSolver/Utilities/TimeRiemann.cpp",
    "solvers/CompressibleFlowSolver/Utilities/TimeRoeKernel.cpp",
    # Template for PWS
    "solvers/PulseWaveSolver/EquationSystems/TemplatePressureArea.cpp",
    # CardiacEPSolver CellMLToNektar template file
    "solvers/CardiacEPSolver/Utilities/CellMLToNektar/nektar/template/model.cpp",
    # FFT
    "library/LibUtilities/FFT/NekDeviceFFT.cpp",
    "library/LibUtilities/FFT/NekDeviceFFT.cu",
    "library/LibUtilities/FFT/NekDeviceFFT.hip",
    "library/LibUtilities/FFT/DerivZDeviceFFT.cpp",
    "library/LibUtilities/FFT/DerivZDeviceFFT.cu",
    "library/LibUtilities/FFT/DerivZDeviceFFT.hip",
    "library/UnitTests/LibUtilities/TestDeviceFFT.cpp",
    # Profiler
    # NekBlas
    "library/LibUtilities/LinearAlgebra/NekBlas/magma.cpp",
    "library/LibUtilities/LinearAlgebra/NekBlas/magmaHandle.cpp",
    "library/LibUtilities/LinearAlgebra/NekBlas/xsmm.cpp",
    "library/LibUtilities/LinearAlgebra/NekBlas/xsmmHandle.cpp",
    "library/LibUtilities/LinearAlgebra/NekBlas/cuBlas.cpp",
    "library/LibUtilities/LinearAlgebra/NekBlas/cuBlasHandle.cpp",
    "library/LibUtilities/LinearAlgebra/NekBlas/hipBlas.cpp",
    "library/LibUtilities/LinearAlgebra/NekBlas/hipBlasHandle.cpp",
    "library/LibUtilities/LinearAlgebra/NekBlas/oneMath.cpp",
    "library/LibUtilities/LinearAlgebra/NekBlas/oneMathHandle.cpp",
    # CompressibleFlowSolverRedesign, built off by default while its advection
    # path is the one this branch replaces. The flux-framework rewrite turns
    # the solver back on and takes these five entries out again.
    "solvers/CompressibleFlowSolverRedesign/CompressibleFlowSolverRedesign.cpp",
    "solvers/CompressibleFlowSolverRedesign/EquationSystems/EulerCFE.cpp",
    "solvers/CompressibleFlowSolverRedesign/EquationSystems/NavierStokesCFE.cpp",
    "solvers/CompressibleFlowSolverRedesign/RiemannSolvers/UnitTests/TestRiemann.cpp",
    "solvers/CompressibleFlowSolverRedesign/RiemannSolvers/Profilers/ProfilerRiemannOps.cpp",
]

ignore_sources = [ os.path.join(cwd, os.path.normpath(p)) for p in ignore_sources ]

with open(sys.argv[1], 'r') as f:
    compilation_data = json.load(f)
    compiled_files = [ entry['file'] for entry in compilation_data ]

    # Search for all c/cpp files in library, solvers and utilities folder.
    found_files = []
    for folder in folders:
        for ext in exts:
            found_files += glob.glob(os.path.join(cwd, folder, '**', '*.{:s}'.format(ext)), recursive=True)

    # Compare the lists of files.
    all_good = True
    for f in found_files:
        if f in ignore_sources or "XSMM" in f or "CUDA" in f or "HIP" in f or "SYCL" in f or "AVX" in f or "UnitTests/Operators/Test" in f:
            continue

        if f not in compiled_files:
            print('Uncompiled file: ' + f)
            all_good = False

    if not all_good:
        exit(1)
