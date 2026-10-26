# MHD with conducting thick walls and unified partitions

This small 2D planar channel couples `IncNavierStokesSolver` on the fluid mesh
to `ADRSolver` on a conforming fluid/solid mesh. The two executables retain
their existing single-mesh field layout. No field-specific domain or boundary
framework is needed.

There are **4 fluid quads and 4 solid quads**, four modes and five quadrature
points per direction. The default run takes only **5 time steps**.

## Build and run

From the repository root, with a CWIPI-enabled build in `./build`:

```bash
module load nektar++/latest
cmake -S . -B build -DNEKTAR_USE_MPI=ON -DNEKTAR_USE_CWIPI=ON \
    -DNEKTAR_USE_SCOTCH=ON -DNEKTAR_UTILITY_EXTRAS=ON \
    -DCMAKE_BUILD_WITH_INSTALL_RPATH=OFF
cmake --build build --target MHDPartition ADRSolver IncNavierStokesSolver -j 4
solvers/IncNavierStokesSolver/Examples/CoupledMHDThickWall2D/Allrun
```

CWIPI 0.11.1 can be built using the existing ThirdParty configuration. It must
be built against the same OpenMPI installation as Nektar++. If it is already
installed, provide `CWIPI_LIBRARY`, `CWIPI_INCLUDE_DIR`, and set
`THIRDPARTY_BUILD_CWIPI=OFF` when configuring. PETSc is not required.

`Allrun` selects the build-tree executable names, including the Debug `-g`
suffix, rather than an older installation in `build/dist` or `/apps`. By default
it uses coefficient transfer with **M=2 fluid partitions and N=1 solid partition**:
IncNS has 2 ranks, ADR has 3 ranks, and the total is 5 ranks. Both solutions
are checked against the analytical fields. An optional first argument selects
the result directory. The default is `build/coupled-mhd-thick-wall-2d`, containing:

- `log.coupled-mhd`: solver output and analytical errors;
- `log.partition`: preprocessing output;
- `partitions/partition-map.csv`: element ID, material and owning ranks;
- `partitions/fluid_xml`, `partitions/electric_xml`: self-contained rank sessions;
- `partitions/fluid.fld`: `u,v,phi,p` on the fluid mesh;
- `partitions/electric.fld`: `phi,Jx,Jy,Jz,sigma` on the fluid/solid mesh.

For M=2, N=2 (6 total ranks):

```bash
FLUID_RANKS=2 SOLID_RANKS=2 \
    solvers/IncNavierStokesSolver/Examples/CoupledMHDThickWall2D/Allrun \
    build/coupled-mhd-m2-n2
```

M and N must be positive, with at least one element per partition. Use a new
result directory when changing the rank counts. To run the original CWIPI
point-interpolation mode, use `MHD_TRANSFER=Cwipi`; then `FLUID_RANKS` and
`ELECTRIC_RANKS` independently select the solver sizes (both default to 1),
and field outputs are `fluid-mesh.fld` and `electric-mesh.fld`.

The `IncNavierStokesSolver_MHDCoefficients` CTest entry checks M/N=1/1, 2/1 and
2/2, asymmetric polynomial transfers in both directions, reversed application
launch order, and startup rejection of incompatible geometry. The existing
`IncNavierStokesSolver_CoupledMHDThickWall2D` entry retains the point-mode tests
at 1+1 and 2+2 ranks. `generate_case.py` regenerates the four checked-in XML
files; generation is not required to run the example.

## Unified preprocessing and coefficient transfer

The fluid mesh must be a subset of the electrical mesh, preserving global
element, edge/face and vertex IDs, element orientations and geometry. The ADR
`FLUID` composites must contain exactly the fluid mesh domain. Both solvers
must use the same fluid-element bases, orders and quadrature for velocity and
potential. Their physical boundary conditions can differ.

The preprocessing command, run with a single process, is:

```bash
MHDPartition 2 1 partitions fluid-mesh.xml fluid.xml electric-mesh.xml electric.xml
mpiexec -np 2 IncNavierStokesSolver --cwipi fluid partitions/fluid_xml \
    : -np 3 ADRSolver --cwipi electric partitions/electric_xml
```

Use the build-tree executable paths (and `-g` suffix for Debug) if these
commands are not on `PATH`. `Allrun` does this automatically.

`MHDPartition` uses the existing Scotch partitioner once for the fluid region
and once for the solid region. The fluid partition sets are reused unchanged
in both applications. IncNS rank `i` and ADR rank `i`, for `0 <= i < M`, own
the same fluid elements; ADR ranks `M ... M+N-1` own only solid elements.
The existing XML partition writer preserves each session's boundary conditions
and the composite ordering required by the global continuous Galerkin solve.
ADR still solves one globally assembled fluid/solid electric problem over all
M+N ranks, including communication across the material interfaces.

The generated coupling blocks use `TYPE="Coefficients"`, `Role="Fluid"` or
`Role="Electric"`, and `FluidRanks=M`. For example, the fluid block is:

```xml
<COUPLING NAME="MHD" TYPE="Coefficients">
    <I PROPERTY="Role" VALUE="Fluid" />
    <I PROPERTY="FluidRanks" VALUE="2" />
    <I PROPERTY="SendSteps" VALUE="1" />
    <I PROPERTY="ReceiveSteps" VALUE="1" />
    <I PROPERTY="SendVariables" VALUE="u,v,mhd_time,mhd_stop" />
    <I PROPERTY="ReceiveVariables" VALUE="phi" />
</COUPLING>
```

The electric block reverses the field lists and uses `Role="Electric"`.
The preprocessor converts the source CWIPI blocks automatically. CWIPI still
creates application-local communicators through distinct `--cwipi` names.
A private MPI intercommunicator carries direct paired messages, identifying
the two applications by role; `RemoteName` is accepted but unused in this mode.
Only one fluid and one electric application are supported per MPI job.

At initialization, each pair checks element IDs, ordered vertices, physical
quadrature coordinates, basis types and orders. It caches a coefficient
packing map sorted by global element ID. A mismatch aborts before stepping;
later exchanges reuse the map without geometric location or interpolation.
Messages contain element-local expansion coefficients, not globally assembled
CG degrees of freedom. Solid ranks receive no field payload from IncNS.

Velocity is transformed locally from the current RHS-stage physical arrays
before sending, because the stored velocity coefficients can belong to another
integration stage. ADR reconstructs the velocity at its quadrature points for
the motional source. Solved potential coefficients are copied directly back
into the fluid potential field. Packing and payload size scale with the local
coefficient count; initialization also exchanges geometry metadata once.
This removes point location and interpolation, but is not a measured speedup
claim. Static partitions and expansion layouts are required throughout a run.

## Geometry and analytical solution

Fluid occupies `0 <= x <= 1`, `0 <= y <= 1`. The lower wall occupies
`-0.25 <= y <= 0`; the upper wall occupies `1 <= y <= 1.25`. The fluid
conductivity is 1 and the wall conductivity is 0.1. Coordinates, conductivity,
and force follow the existing IncNS normalization, with density normalized
to one.

The outer wall surfaces are electrodes with potentials 0 and `Voltage=0.1`.
The left and right electrical boundaries are insulating. These electrode
conditions give a nonzero, analytically known current through both walls.
Fluid velocity at the inlet/outlet is `u=4*U0*y*(1-y)`, `v=0`; the fluid/wall
interfaces are no-slip. `U0=0.1`, `Bz=B0=0.1`, and `Kinvis=0.1`.

For wall thickness `h`, define:

```text
R  = 2*h/sigma_wall + 1/sigma_fluid
Jy = -(Voltage + 2*U0*B0/3)/R = -0.0177777777777778
```

Let `s=clamp(y,0,1)`. The potential throughout fluid and walls is:

```text
phi(y) = -Jy*((y+h)/sigma_wall + (1/sigma_fluid-1/sigma_wall)*s)
         -U0*B0*(2*s^2 - 4*s^3/3)
```

The fluid body force `8*Kinvis*U0 - Jy*B0` balances viscous diffusion and the
Lorentz force. The parabolic initial flow is therefore an exact steady
solution of the coupled equations. The test checks both returned fluid
potential and full fluid/solid potential, as well as velocity and pressure.
The solution is polynomial on each element, so this coarse mesh is sufficient
to verify the coupling without a resolution study.

## Input configuration

In the fluid session, retain `phi` immediately before pressure and select:

```xml
<I PROPERTY="ElectricPotential" VALUE="Coupled" />
```

The default is `Internal`, preserving existing MHD cases. `ForcingLorentz`
and its `ElectricMagneticFields` function remain required. The coupled path
receives `phi`, computes its gradient on the fluid mesh, and uses the existing
Lorentz-force calculation. Its local electrical boundary conditions are not
applied to the received potential.

The ADR session declares **only `phi`** as a solution variable:

```xml
<I PROPERTY="EQType" VALUE="CoupledElectricPotential" />
<I PROPERTY="Projection" VALUE="Continuous" />
```

At the `NEKTAR` level, define material volume composites:

```xml
<ELECTRICPOTENTIAL>
    <FLUID COMPOSITE="C[0]" CONDUCTIVITY="ElectricConductivity" />
    <SOLID COMPOSITE="C[1]" CONDUCTIVITY="WallConductivity" />
</ELECTRICPOTENTIAL>
```

`CONDUCTIVITY` is a constant expression using session parameters. Regions must
not overlap and must cover every electrical element; conductivities must be
positive. Additional `SOLID` entries can specify other materials. Interface
vertices/edges/faces must be shared by the conforming electrical mesh.
Electrical boundary conditions apply to the exterior only. With the weak
current-conservation formulation, homogeneous Neumann means zero **total
normal current**. A nonzero Neumann value is minus the outward current density,
not simply the normal potential derivative. Pure-Neumann problems use the
existing singular-system handling to fix the potential reference.

Both sessions must use the same constant `ElectricMagneticFields`. In the
fluid session `ElectricConductivity` must match the ADR fluid material.

The checked-in source `COUPLING` blocks use `TYPE="Cwipi"`, the same coupling
name, distinct `--cwipi` application names, and the following point-mode options.
The default preprocessing replaces these with the coefficient configuration
described above:

```xml
<I PROPERTY="SendSteps" VALUE="1" />
<I PROPERTY="ReceiveSteps" VALUE="1" />
<I PROPERTY="ReceiveProjection" VALUE="Local" />
<I PROPERTY="FilterWidth" VALUE="0" />
<I PROPERTY="NotLocMethod" VALUE="Keep" />
<I PROPERTY="GeomTol" VALUE="1e-10" />
```

Fluid sends `u,v,mhd_time,mhd_stop` and receives `phi`. ADR does the reverse.
For full 3D meshes, insert `w` after `v`. Received velocities are auxiliary
arrays. Solid velocity is explicitly excluded from the motional source. In point
mode, unlocated points inside solid elements are expected. The local receive
projection prevents material-interface smoothing of these arrays. The PDE
solution itself still uses a continuous Galerkin space.

## Exchange protocol and scope

Each Lorentz-force evaluation sends that evaluation's velocity and time,
waits for the ADR solution, and then computes the force. The exchange counter
is independent of the time-step number, supporting initialization and multiple
RHS stages. A normal request carries `mhd_stop=-1`; termination uses `+1` and
receives a final acknowledgement before either side finalizes CWIPI. Both
control fields are transported internally; they are not solution variables.
Coefficient mode carries time and the shutdown marker in message metadata.

`mhd_time` and `mhd_stop` must be spatial constants on every received fluid
element. ADR checks them to detect missing overlap or inadequate receive
resolution. The fluid mesh must lie within the electrical fluid region.
There is no electrical time integration; ADR solves the instantaneous
conductivity-weighted Poisson problem for each request. Output potential and
current correspond to the last force evaluation.

The implementation targets stationary conducting solids, static full 2D/3D
meshes, and constant imposed electromagnetic fields. The supplied validation
is 2D. Homogeneous Fourier meshes and moving electrical meshes are not
supported by this coupled mode. Coefficient mode requires matching fluid
submeshes and expansion layouts. Point mode also permits independent meshes;
interpolation then introduces spatial transfer error that requires its own
accuracy study.
