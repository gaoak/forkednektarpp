# CWIPI MHD with conducting thick walls

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
    -DCMAKE_BUILD_WITH_INSTALL_RPATH=OFF
cmake --build build --target ADRSolver IncNavierStokesSolver -j 4
solvers/IncNavierStokesSolver/Examples/CoupledMHDThickWall2D/Allrun
```

CWIPI 0.11.1 can be built using the existing ThirdParty configuration. It must
be built against the same OpenMPI installation as Nektar++. If it is already
installed, provide `CWIPI_LIBRARY`, `CWIPI_INCLUDE_DIR`, and set
`THIRDPARTY_BUILD_CWIPI=OFF` when configuring. PETSc is not required.

`Allrun` selects the build-tree executable names, including the Debug `-g`
suffix, rather than an older installation in `build/dist` or `/apps`. It uses
one MPI rank per solver and checks both solutions against the analytical
fields. An optional first argument selects the result directory. The default
is `build/coupled-mhd-thick-wall-2d`, containing:

- `log.coupled-mhd`: solver output and analytical errors;
- `fluid-mesh.fld`: `u,v,phi,p` on the fluid mesh;
- `electric-mesh.fld`: `phi,Jx,Jy,Jz,sigma` on the fluid/solid mesh.

For two ranks per solver:

```bash
FLUID_RANKS=2 ELECTRIC_RANKS=2 \
    solvers/IncNavierStokesSolver/Examples/CoupledMHDThickWall2D/Allrun \
    build/coupled-mhd-thick-wall-2d-parallel
```

The `IncNavierStokesSolver_CoupledMHDThickWall2D` CTest entry covers both rank
counts. `generate_case.py` reproducibly regenerates the four checked-in XML
files; generation is not required to run the example.

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

Both `COUPLING` blocks use `TYPE="Cwipi"`, the same coupling name, distinct
`--cwipi` application names, and these options:

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
arrays. Solid velocity is explicitly excluded from the motional source;
unlocated points inside solid elements are expected. The local receive
projection prevents material-interface smoothing of these arrays. The PDE
solution itself still uses a continuous Galerkin space.

## Exchange protocol and scope

Each Lorentz-force evaluation sends that evaluation's velocity and time,
waits for the ADR solution, and then computes the force. The exchange counter
is independent of the time-step number, supporting initialization and multiple
RHS stages. A normal request carries `mhd_stop=-1`; termination uses `+1` and
receives a final acknowledgement before either side finalizes CWIPI. Both
control fields are transported internally; they are not solution variables.

`mhd_time` and `mhd_stop` must be spatial constants on every received fluid
element. ADR checks them to detect missing overlap or inadequate receive
resolution. The fluid mesh must lie within the electrical fluid region.
There is no electrical time integration; ADR solves the instantaneous
conductivity-weighted Poisson problem for each request. Output potential and
current correspond to the last force evaluation.

The implementation targets stationary conducting solids, static full 2D/3D
meshes, and constant imposed electromagnetic fields. The supplied validation
is 2D. Homogeneous Fourier meshes and moving electrical meshes are not
supported by this coupled mode. Start with matching fluid submeshes and
quadrature; interpolation on independent meshes also introduces spatial
transfer error that requires its own accuracy study.
