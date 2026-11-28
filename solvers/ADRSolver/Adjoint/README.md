# Temperature adjoint solvers (`AdjointTemperature`, `UnsteadyAdjointTemperature`)

Continuous adjoint of the temperature advection-diffusion equation, implemented
on top of the stock `SteadyAdvectionDiffusion` / `UnsteadyAdvectionDiffusion`
assembly in `ADRSolver`.  The `IncNavierStokes*` modifications on the same
branch belong to the separate Hartmann-braking feature, not to the adjoint
solver.

| File | Contents |
|---|---|
| `solvers/ADRSolver/Adjoint/AdjointBaseFlow.{h,cpp}` | reads a `.fld` base flow or a `%d` time series, interpolates in time, hands out `V = -u` (and the interpolation state) |
| `solvers/ADRSolver/Adjoint/AdjointBoundaryCheck.h` | shared guard that disables Xxt when non-Dirichlet BCs meet a base flow (see "Known pitfalls") |
| `solvers/ADRSolver/EquationSystems/AdjointTemperature.{h,cpp}` | steady adjoint (`EQTYPE="AdjointTemperature"`) |
| `solvers/ADRSolver/EquationSystems/UnsteadyAdjointTemperature.{h,cpp}` | pseudo-time adjoint (`EQTYPE="UnsteadyAdjointTemperature"`) |
| `solvers/ADRSolver/CMakeLists.txt` | the sources above are added to the `ADRSolver` target |

## Equations

Forward problem (`ν = k/(ρ c_p)`, `q = Q_v/(ρ c_p)`):

```
∂_t T + u·∇T − ν∇²T = q
k ∂_n T + β (T − T_b) = g          (Robin on ∂Ω)
```

Adjoint problem:

```
−∂_t ψ − u·∇ψ − ν∇²ψ = w_T/(ρ c_p)         ψ(x, t1) = 0
∂_n ψ + a ψ = b,   a = (β + ρ c_p u_n)/k,  b = β w_q/k
```

**`u_n` in `a` is the forward normal velocity (`u·n`), never `−u·n`.**
The steady solver drops the time derivatives.  The unsteady solver marches in
pseudo-time `τ = t1 − t` with velocity `V = −u(x, t1−τ)` and source
`w_T(x, t1−τ)/(ρ c_p)`; the session boundary values and the source are
evaluated at the physical time `t = t1 − τ` automatically.

Nektar's Robin convention is `∂_n u + PRIMCOEFF·u = VALUE`, so for an adjoint
Robin boundary you write `PRIMCOEFF = a` and `VALUE = b` (both computed with
`u_n = u·n` of the **forward** flow, e.g. `(beta + rho*cp*Uadv)/k`).
Note that Nektar evaluates `PRIMCOEFF` once, at `t = 0`; a time-dependent
Robin coefficient is frozen at its `t = 0` value.  Keep `a` constant.

## Steady solver

```
<I PROPERTY="EQTYPE" VALUE="AdjointTemperature"/>
```

Solves `−u·∇ψ − ν∇²ψ + λψ = s` with `s = w_T/(ρ c_p)` supplied by the session
function `AdjointSource` (the name `Forcing` is accepted as a fallback).
It reuses `LinearAdvectionDiffusionReactionSolve` with `c = −u/ε`,
`λ_factor = −λ/ε` and `f = −s/ε`; `ε = ν`.

Parameters (`<PARAMETERS>`):

| Name | Default | Meaning |
|---|---|---|
| `Rho`, `Cp`, `Conductivity` | 1 | `ν = Conductivity/(Rho·Cp)` |
| `epsilon` | — | overrides `ν` directly |
| `lambda` | 0 | linear reaction coefficient (sign checked by `run_steady_lambda`) |
| `BaseFlowTime` | first slice | physical time used when a time series is given |

Base flow: an analytic session function `BaseFlow` (`Vx`, `Vy`, `Vz`) or the
SolverInfo property `BaseFlowFile` (a `.fld` or a printf pattern with
`N_slices > 1`).  Series parameters: `N_slices`, `N_start`, `N_skip`,
`TimeSeriesStart`, `TimeSeriesDt`, `BaseFlow_interporder` (1–4),
`BaseFlowPeriodic`, `Period`.  The velocity is handed to the assembly as
`V = −u` and is never modified in place, so repeated `v_DoSolve` calls do not
double-scale it; the source is cached at initialisation, so a second solve
also reuses the same RHS.

## Unsteady solver

```
<I PROPERTY="EQTYPE" VALUE="UnsteadyAdjointTemperature"/>
```

Requires a placeholder session function `AdvectionVelocity` (zero components;
it is overwritten by the base flow) and a time-dependent `AdjointSource`.
Window parameters: `AdjointStartTime` (default 0), `AdjointEndTime` (or
Nektar's `FinTime`).  The pseudo-time march runs from 0 to
`AdjointEndTime − AdjointStartTime`; checkpoints store the pseudo-time in
`Time` and the physical time in the metadata fields `PhysicalTime` /
`AdjointTimeShift`.  Restart resumes at the pseudo-time stored in the
checkpoint.  Sub-stepping and the steady-state stop condition are rejected.

The base flow is re-evaluated at every stage (`t = t1 − τ`), so both a
time-series base flow and a time-dependent source follow the reversed time.

## Verification

All cases live in `/storage/debian2home3/MHD/yujiez/adjoint-tests/` and are
driven by `run_regression.sh` (run it on the compute node; it writes
`regression.log`).  Latest full run: everything below reproduced exactly.

| Directory | Case | L2 error |
|---|---|---|
| `run_steady_mms`, `_p5`, `_p7`, `_p9` | steady MMS, P4/P5/P7/P9 | 2.66511e-06 / 6.86236e-08 / 2.96561e-11 / 7.92e-15 |
| `run_steady_mms_u15` | steady MMS, u=1.5 | 2.66331e-06 |
| `run_steady_file` | `.fld` base flow / interpolated series / periodic series / MPI 2 ranks | 2.66511e-06 / 2.66331e-06 / 2.66331e-06 / 2.66511e-06 |
| `run_steady_robin` | all-Robin MMS | 8.37053e-08 |
| `run_steady_robin` | Robin + Dirichlet | 8.36979e-08 |
| `run_steady_robin` | Dirichlet + Neumann MMS | 1.73908e-07 |
| `run_steady_linbc` | linear `psi=x`, Dirichlet+Neumann / all-Robin | 2.58e-14 / 5.40e-15 |
| `run_steady_lambda` | lambda = 0.3 manufactured solution (pins the `−λ/ε` sign) | 8.35088e-08 |
| `run_unsteady_mms` | time-varying MMS / MPI 2 ranks | 2.68659e-06 |
| `run_unsteady_mms` | time-dependent Neumann wall (exercises the unsteady guard) | 3.19583e-06 |
| `run_unsteady_file` | time-varying `.fld` series | 2.48581e-06 |
| `run_unsteady_restart` | restart from checkpoint | 3.39919e-06 |
| `run_unsteady_steadylimit` | steady limit (falls back to 2.66511e-06) | 2.66511e-06 |
| `run_unsteady_nudiff` | `ν` from `Conductivity/(Rho·Cp)` | 4.84171e-06 |
| `run_unsteady_u0` | zero base flow | 1.68534e-06 |
| `run_unsteady_equilibrium` | steady-state base flow | 2.66820e-06 |

Convergence (steady MMS): P4→P9 gives 2.7e-6 → 7.9e-15, i.e. spectral
convergence to round-off.  The P9 value varies in the last two digits between
runs (7.92255e-15 vs 7.92469e-15) — BLAS-level round-off, not a solver
difference.

The plan's M5 items are covered by the channel demo below: the adjoint
identity `∫ w_T T dV = ∫ ψ q dV` and a localized-source finite-difference
check.

## Sensitivity maps: demo and plotting recipe

The adjoint field **is** the sensitivity map: for the plan's
`J = ∫ w_T T dV + ∫ w_q b dS` (b the forward Robin right-hand side), the
pointwise derivatives are `δJ/δQ_v(x) = ψ(x)` and
`δJ/δb(x) = ψ(x) − w_q(x)` (steady; unsteady: one `ψ` per time level).

`adjoint-tests/run_steady_channel_sensitivity/` is a self-contained demo on
the embedded 10x10 `[-1,1]^2` mesh: Poiseuille flow `u_x = U0(1-y^2)`,
`ν = 0.05`, cold walls and inlet, zero-gradient outflow, and an objective
`J = ∫ w_T T dV` with `w_T` a normalised Gaussian at `(0.5, 0)`.

| File | Contents |
|---|---|
| `tools/make_sensitivity_session.py` | generates `channel_sensitivity.xml` (adjoint), `channel_forward.xml` (unit source), `channel_forward_bump.xml` (localized source) |
| `tools/plot_sensitivity.py` | `.dat` → PNG (linear + log panels) |
| `tools/check_adjoint_identity.py` | quadrature, adjoint identity, FD sensitivity check |

```
ssh yujiez@192.168.211.106
cd /home3/MHD/yujiez/adjoint-tests/run_steady_channel_sensitivity
ADRSolver channel_sensitivity.xml
FieldConvert -f channel_sensitivity.xml channel_sensitivity.fld \
             channel_sensitivity.dat
cd .. && /apps/anaconda3/bin/python3 -I tools/plot_sensitivity.py \
    run_steady_channel_sensitivity/channel_sensitivity.dat \
    run_steady_channel_sensitivity/channel_sensitivity.png
```

Result: `channel_sensitivity.png` (2.04 is `ψ_max`).  Quantitative checks
against the stock forward solver (same mesh):

| Check | Value |
|---|---|
| `∫ w_T T dV` vs `∫ ψ q dV` (q = 1) | 1.628668 vs 1.628678, rel. 6.6e-06 |
| localized source `A g` at (−0.4, 0.5) vs `A∫ ψ g dV` | rel. 2.3e-04 |

For an **unsteady** objective, run `UnsteadyAdjointTemperature` with
checkpoints: frame *k* is `ψ` at pseudo-time `τ_k`, i.e. the sensitivity at
physical time `t1 − τ_k` (the metadata stores `PhysicalTime`).  Convert each
frame with `FieldConvert -f` and animate the sequence exactly like the
`w`/`teta` animations of the heat runs; `FieldConvert` can also write `.vtu`
for ParaView (`/apps/ParaView-5.13.0/bin/pvpython` is available).

Consistency rules used by the demo (they matter for the identity to close):

- **Outflow:** the adjoint BC must match the forward one.  For
  `dT/dn = 0` impose `dψ/dn + (u_n/ν)ψ = 0` (Robin with `β = 0`;
  `PRIMECOEFF` may be an expression in `x,y`).  A plain Neumann condition
  leaves a boundary term `∮ ψ T u_n`, changes `ψ_max` by ≈ 6 % and breaks the
  identity.
- **Stock forward solver sign:** `SteadyAdvectionDiffusion` solves
  `c·∇T − ε∇²T = −Forcing`, so a physical source `q = +1` is entered as
  `Forcing = -1`.
- **`FieldConvert` overwrite:** without `-f` it prompts and, with no TTY,
  keeps the old `.dat` while still exiting 0 — stale-field trap.
- **Post-processing quadrature:** the embedded mesh elements carry
  *equispaced* points, so integrate with Newton-Cotes (Boole's 5-point rule
  is exact for the degree-4 representation), not GLL weights.
- **Dirichlet is where `T` is fixed.  Robin/Neumann coefficients use the
  forward `u_n`.**

## Known pitfalls (library-level, verified with the stock solvers too)

1. **Xxt is a sparse Cholesky** (`sparse_cholesky` in `Xxt.hpp`) and needs a
   symmetric matrix.  With Neumann/Robin BCs and a base flow the CG boundary
   (static-condensation) system is non-symmetric, and Xxt silently returns a
   wrong answer (e.g. L2 1.17 instead of 1.7e-07).  `AdjointBoundaryCheck.h`
   detects an `Xxt*` request together with any non-Dirichlet BC and switches
   the session to `DirectStaticCond` with a warning.  The default Nektar
   solver (`DirectMultiLevelStaticCond`, registered in `GlobalLinSys.cpp`) is
   unaffected.
2. **`IterativeStaticCond` (GMRES) is also wrong for mixed Dirichlet+Neumann
   boundaries**: it returns the exact solution minus a uniform constant in
   the interior (0.084 for the mixed-DN test).  All-Robin and
   Robin+Dirichlet are fine.  For this reason the automatic fallback is
   `DirectStaticCond`, not the iterative solver.
3. **A session without `<BOUNDARYREGIONS>` silently ignores every boundary
   condition** (`DisContField` creates no boundary expansions).  The probes
   in `run_steady_linbc` did not test anything until this block was added.
4. A steady pure-Neumann problem with this operator is singular (constant
   null space); `r_neu`/`r_diffneu` are expected to fail and are not part of
   the regression suite.
5. On `debain-91`, the system Open MPI 5 (`/usr/bin/mpirun`) aborts in
   `opal_shmem_base_select`; use `/apps/openmpi/4.1.5/bin/mpiexec` (the
   regression script defaults to it).
6. `FieldConvert` needs the session to convert `.fld` → `.dat`
   (`FieldConvert session.xml field.fld out.dat`); without it, it segfaults.
7. **A BC whose `VALUE` depends on `t` must carry
   `USERDEFINEDTYPE="TimeDependent"`.**  Nektar infers time dependence only
   from that attribute (`Conditions.cpp`), not from the expression; otherwise
   the boundary value is evaluated once (for the unsteady adjoint: at the
   physical end time `t1`) and frozen.  Symptom: `unsteady_adjoint_mms_neu`
   gives L2 0.0406 instead of 3.2e-06.  The steady solver is unaffected
   (single evaluation), and constant BC values never need the attribute.

## Build

```
ssh yujiez@192.168.211.106
cd /home3/MHD/yujiez/build-forkednektarpp-HB
/apps/cmake/3.28.0/bin/cmake --build . --target ADRSolver -j 8
# binary: solvers/ADRSolver/ADRSolver
```
