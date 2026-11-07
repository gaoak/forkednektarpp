#!/usr/bin/env bash
# Source this file AFTER loading the Nektar++ module:
#   source solvers/IncNavierStokesSolver/Examples/CoupledMHDThickWall2D/scripts/env.sh
# These variables name the current build-tree tools, including Debug suffixes.
mhd_case_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
mhd_repo_dir="$(cd -- "$mhd_case_dir/../../../.." && pwd)"
mhd_build_dir="${NEKTAR_BUILD_DIR:-$mhd_repo_dir/build}"
mhd_suffix=""
if [[ -f "$mhd_build_dir/CMakeCache.txt" ]] &&
   [[ "$(sed -n 's/^CMAKE_BUILD_TYPE:STRING=//p' "$mhd_build_dir/CMakeCache.txt")" == "Debug" ]]; then
    mhd_suffix="-g"
fi
mhd_fluid_solver="$mhd_build_dir/solvers/IncNavierStokesSolver/IncNavierStokesSolver$mhd_suffix"
mhd_electric_solver="$mhd_build_dir/solvers/ADRSolver/ADRSolver$mhd_suffix"
mhd_partitioner="$mhd_build_dir/utilities/Extras/MHDPartition$mhd_suffix"
mhd_nekmesh="$mhd_build_dir/utilities/NekMesh/NekMesh$mhd_suffix"
mhd_fieldconvert="$mhd_build_dir/utilities/FieldConvert/FieldConvert$mhd_suffix"
