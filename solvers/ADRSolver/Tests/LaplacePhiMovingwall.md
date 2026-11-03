# Movingwall velocity potential

`LaplacePhi` supports a prescribed wall velocity on 2D edge boundaries:

```xml
<N VAR="phi0" USERDEFINEDTYPE="Movingwall" VALUE="0" />
```

This imposes `d(phi0)/dn = n_x*u + n_y*v`, where `n` is the outward
normal of the **fluid domain**. Unlike the existing rigid-body `MOVE`
condition, no extra minus sign is applied. `VALUE` is a placeholder;
the solver replaces the Neumann coefficients with the computed flux.
Use `phi0` for a single potential; if several fields use `Movingwall`,
they receive the same prescribed velocity. The rigid-body added-mass
matrix is not computed when any field uses `Movingwall`.

The default input is `Movingwall.dat` in the current working directory.
To select another file, add this to `CONDITIONS/SOLVERINFO`:

```xml
<I PROPERTY="MovingWallFile" VALUE="my-wall-velocities.dat" />
```

Each nonempty line contains exactly `x y z u v w`, separated by whitespace.
Blank lines and `#` comments are accepted. Supply each distinct wall vertex
once, in any order. Coordinates must match mesh vertices to an absolute
tolerance of `1e-8` in mesh units. Missing or ambiguously duplicated
endpoints cause an error. For a planar 2D mesh, supply `z=0` and normally
`w=0`; the third velocity component does not contribute to the flux.

For each edge, velocity at reference coordinate `xi` is
`U(xi) = (1-xi)/2 * U0 + (1+xi)/2 * U1`.
Thus the same rule applies to curved edges, using their local coordinate,
not a projection onto the straight chord. The normal is evaluated at each
boundary quadrature point. Velocities are read once and are time independent.
3D face boundaries are not supported by this endpoint format.

## Airfoil example

`Movingwall.dat` contains all 48 distinct endpoints of composite `C[0]`
in `NACA0012_Re400.xml`, with a uniform velocity `(1,0,0)`.
`LaplacePhiMovingwall.xml` uses this file for `phi0`, retaining zero
Dirichlet conditions on the outer boundary. From this directory run:

```sh
/path/to/ADRSolver NACA0012_Re400.xml LaplacePhiMovingwall.xml
```

The original `LaplacePhi.xml` and its `MOVE` regression remain unchanged.

## Interpolation regression

`LaplacePhiMovingwallLinear.xml` uses `MovingwallLinear.dat`, with endpoint
velocities `(y,x,0)` in deliberately shuffled order. On the unit square,
three sides have `Movingwall` conditions and the bottom has zero Dirichlet
data. The exact solution is `phi0=x*y`. This checks varying endpoint
velocities, edge orientation, the sign of the normal flux and the filename
override. Its `.tst` file requires L2 and Linf errors below `1e-10`.
