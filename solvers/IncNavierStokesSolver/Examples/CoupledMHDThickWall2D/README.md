# Coupled Hunt flow with finite conducting Hartmann walls (Ha=1)

[中文学生教程](TUTORIAL_zh.md) covers Mesh2D `square` meshing, configuration,
partitioning, MPI launch and postprocessing.

This example uses the **unchanged** four-fluid-quad/four-solid-quad mesh as a
square duct cross-section. Velocity is along **z** (`w`), the magnetic field
is along **y**, and the top/bottom Hartmann walls conduct electricity. The
left/right side walls are insulating. All outer electrical surfaces are
insulated; there is no imposed electrode voltage. Conductivity-weighted CG
assembly couples fluid and solid potential/current across the internal interfaces.

The geometry is `0<=x,y<=1` in the fluid, with solid slabs at `-0.25<=y<=0`
and `1<=y<=1.25`. This is a **finite-conductivity, thick-wall Hunt configuration**,
not the ideal perfectly conducting-wall limit. Conducting Hartmann walls and
insulating side walls follow the generalized Hunt configuration discussed in
[Bluck et al., JFM 771 (2015)](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/an-analytical-solution-to-electromagnetically-coupled-duct-flow-in-mhd/902FADFCFAB35273CA5E4D25BCDE4920).

## Equations and quasi-3D setting

The fluid session follows the 3DH1D arrangement in
`/StorageLBM/yujiez/Shercliff_Ha1000/N5/Duct.xml`:

- `Homogeneous=1D`, `HomModesZ=2`, `LZ=1`;
- variables `u,v,w,phi,p`, with flow-rate forcing along z;
- no-slip velocity and high-order pressure conditions on all four x/y walls;
- `By=MagneticFieldStrength`, with implicit Lorentz damping enabled.

Only the fully developed **real zero Fourier mode** is retained. The second
Fourier plane is the null imaginary mean mode; this is not a simulation of
z-dependent disturbances. Homogeneous-direction MPI decomposition is unsupported
for this coupling: keep `--npz=1` (the default). Partition the x/y elements only.

IncNS evolves the quasi-3D velocity. ADR solves the **2D cross-sectional**
conductivity-weighted electric problem, since `d(phi)/dz=0`. It receives all
three velocity components, so the source includes `w` through
`u x B = (-w*By,0,u*By)`. The pairing validates the x/y layout once, transfers
the mean-plane local coefficients, and inserts the returned potential into the
real zero mode while clearing its imaginary mean mode. No spatial interpolation
is used. Higher Fourier-mode configurations are rejected instead of silently
being reduced to a cross-sectional solve.

The flow has five modes/seven quadrature points per x/y direction for velocity
and phi; pressure has four modes with the same quadrature. Unlike the supplied
reference's lower-order phi, this direct-transfer example keeps the velocity and
potential bases identical on paired fluid elements.

## Parameters

The density-normalized definition is

```text
Ha = B*a*sqrt(sigma_fluid/Kinvis)
a  = HalfWidth = 0.5
B  = Ha/a*sqrt(Kinvis/sigma_fluid)
```

Defaults: `Ha=1`, `Kinvis=0.1`, `sigma_fluid=1`, `sigma_wall=10`, wall thickness
`0.25`, `Flowrate=0.1`, `TimeStep=0.02`, `NumSteps=500`. Thus `By≈0.632455532`
and the wall conductance ratio `sigma_wall*h/(sigma_fluid*a)=5`.
`Flowrate` sets mean axial velocity (section area is one here). The default
run reaches time 10 and performs 501 electric solves. It is a small teaching
and coupling check, not a spatially converged high-Ha benchmark.

The electric boundary problem is pure Neumann. Nektar++ fixes the additive
potential reference; absolute phi depends on the gauge. Currents and gradients
are physical quantities. Boundary conditions in IncNS are not reapplied to the
potential returned by ADR.

## Build and run

From the repository root:

```bash
module load nektar++/latest
cmake -S . -B build -DNEKTAR_USE_MPI=ON -DNEKTAR_USE_CWIPI=ON \
    -DNEKTAR_USE_SCOTCH=ON -DNEKTAR_UTILITY_EXTRAS=ON \
    -DNEKTAR_UTILITY_FIELDCONVERT=ON -DCMAKE_BUILD_WITH_INSTALL_RPATH=OFF
cmake --build build --target MHDPartition ADRSolver IncNavierStokesSolver FieldConvert -j4
solvers/IncNavierStokesSolver/Examples/CoupledMHDThickWall2D/Allrun
```

CWIPI and Nektar++ must use the same MPI installation. Existing builds can
usually use just the build command. The script selects current build-tree
executables, including the Debug `-g` suffix.

Default M=2, N=1 means **2 IncNS ranks + 3 ADR ranks = 5 total ranks**. An
optional first script argument selects the result directory; the default remains
`build/coupled-mhd-thick-wall-2d`. `FLUID_RANKS=M SOLID_RANKS=N` selects other
positive counts; use a new output directory when changing the counts.

`MHD_CASE_DIR` selects a directory containing the four input XML files.
`MHD_CHECK_RESULTS=0` skips the default Hunt physical checker for customized or
short transient runs. `MHD_TIMEOUT` defaults to 120 seconds. `FIELD_CONVERT`
can override the postprocessing binary. This quasi-3D example requires
`MHD_TRANSFER=Coefficients`; the old point-transfer path is tested separately.

## Manual partitioning and launch

After loading the module, source `scripts/env.sh` from the example directory
(or its full path) to set the `$mhd_*` executable paths. In a separate work
directory containing the four input files:

```bash
"$mhd_partitioner" 2 1 partitions \
    fluid-mesh.xml fluid.xml electric-mesh.xml electric.xml
mpiexec -np 2 "$mhd_fluid_solver" --cwipi fluid "$PWD/partitions/fluid_xml" \
    : -np 3 "$mhd_electric_solver" --cwipi electric "$PWD/partitions/electric_xml"
```

Scotch partitions fluid elements once and reuses those partitions in both
applications. It partitions the extra solid elements separately. IncNS rank i
pairs with ADR rank i for `0<=i<M`; ADR ranks `M..M+N-1` contain only solid.
Total MPI ranks are `2M+N`. The CSV manifest lists every element's owner.

The generated coupling blocks use `TYPE=Coefficients`, `Role=Fluid/Electric`
and `FluidRanks=M`. Fluid sends `u,v,w,mhd_time,mhd_stop`, receives `phi`; ADR
reverses the lists. Time/stop values are protocol data, not solution variables.
CWIPI creates the application-local communicators; private MPI messages carry
the paired coefficient payloads. Exactly two applications are supported.

ADR caches the imposed fixed-field load at initialization and updates the
motional load only on fluid elements. Pure solid ranks allocate no velocity
receive or motional-load work arrays. They still participate in ADR's control
reductions and global electrical solve. Current output retains the velocity
from the final force evaluation, including after the shutdown handshake.

## Output and checks

The result directory contains:

- `log.partition`, `log.coupled-mhd`;
- `partitions/partition-map.csv`, `fluid_xml/`, `electric_xml/`;
- `partitions/fluid.fld/`: `u,v,w,phi,p` in 3DH1D;
- `partitions/electric.fld/`: `phi,Jx,Jy,Jz,sigma` in the 2D section;
- `partitions/fluid.prs`: the axial flow-rate controller's driving gradient;
- `fluid.csv`, `electric.csv`, and `log.convert-*`, generated by the checker.

With no exact Hunt solution supplied, the solver's `L 2 error` lines compare
the solution to zero; they are **not discretization errors**. Allrun instead
checks the axial flow rate, no-slip and zero transverse velocity, agreement of
returned potential, nonzero wall currents, late-time driving-gradient stability,
and the dissipative identity

```text
integral_fluid w*(J x B)_z + integral_fluid+solid |J|^2/sigma = 0.
```

Typical default results: mean w≈0.1, max sampled w≈0.206953, driving gradient
≈0.309465712516. CSV rounding limits the energy check (threshold `1e-4` relative).
These consistency checks do not establish an exact-profile or grid-convergence
error. The eight-quad grid should not be used to resolve Ha=1000 wall layers.

For a 2D mean-section visualization, use:

```bash
"$mhd_fieldconvert" -f -m homplane:planeid=0 \
    fluid-mesh.xml fluid.xml partitions/fluid.fld fluid-mean.vtu
"$mhd_fieldconvert" -f electric-mesh.xml electric.xml \
    partitions/electric.fld electric.vtu
```

Use `homplane:planeid=0:wavespace` to inspect the real Fourier coefficient, or
planeid=1:wavespace to inspect the null imaginary mean. Raw CSV export of the
unextracted 3DH1D field is not used by this example.

## Mesh generation and regression coverage

`generate_case.py` regenerates the same geometry and the current Hunt sessions.
The `scripts` helpers follow Mesh2D's `square` workflow; see the Chinese tutorial.
`MHDPartition` preserves each solver's boundary definitions and the composite
ordering needed for parallel CG assembly.

The old planar, electrode-driven analytical case is retained under
`../../Tests/CoupledMHDChannel/`. CTests `CoupledMHDThickWall2D` and
`MHDCoefficients` use these fixtures to retain point/coefficients, fixed-field
and geometry-mismatch regression coverage. `IncNavierStokesSolver_CoupledHunt`
checks the new Hunt example with M/N=1/1, 2/1 and 2/2, partition-independent axial
velocity, and rejection of unsupported Fourier layouts. Reconfigure CMake after
updating the source to register the new test.
