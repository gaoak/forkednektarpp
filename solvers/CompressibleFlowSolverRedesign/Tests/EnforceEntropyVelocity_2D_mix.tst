<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>
    Subsonic Riemann inflow holding the entropy and the velocity, on a mixed
    quadrilateral/triangle mesh.

    Ported from the legacy pair square_mix.xml + session_enforceEntropyVelocity.xml,
    with two changes and one addition.

    The equation type moves from NavierStokesImplicitCFE to NavierStokesCFE,
    because the redesign has no implicit path. The timestep comes down with it,
    from the implicit 0.05 to 0.001, which is the explicit stability limit here
    rather than a choice.

    The initial condition is perturbed by a density and momentum bump, and that
    addition is the point of this file. With the freestream initial state the
    legacy case uses, this condition and a plain Dirichlet at the same values
    give byte-identical answers - the uniform state is preserved either way, so
    nothing distinguishes the Riemann and entropy algebra from simply imposing
    the session values. Legacy's own test has that weakness. The bump makes the
    interior trace at the inflow differ from the prescribed state, and the two
    then separate clearly: rho L2 2.01186 against 2.01447, E 6.56585 against
    6.58032. Reverting the operator therefore changes this test, which is what a
    test of a boundary condition should do.

    Verified against legacy on this same file, which both solvers accept
    unchanged: every value agrees to all printed digits except rhov, which
    differs in the sixth significant figure (0.00628557 against 0.00628555).
    This file carries legacy's values with tolerances set to accept that.

    The unperturbed variant is worth keeping in mind as a separate property:
    with it, the redesign reproduces legacy's published metrics exactly - rho
    and rhou Linf 1.0, E 3.29018, L2 2.0 and 6.58036, rhov at round-off - which
    says the condition leaves a uniform state undisturbed.

    The mesh is legacy's square_mix, two quadrilaterals and four triangles, so
    the trace carries more than one block. The session also puts an
    EnforceEntropyVelocity region and a PressureOutflow region side by side, and
    the two read the energy slot differently: the inflow takes a genuine
    conserved state there, the outflow a static pressure. Only the tag says
    which.
    </description>
    <executable>CompressibleFlowSolverRedesign</executable>
    <parameters>EnforceEntropyVelocity_2D_mix.xml</parameters>
    <files>
        <file description="Session File">EnforceEntropyVelocity_2D_mix.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="rho"  tolerance="1e-7">2.01186</value>
            <value variable="rhou" tolerance="1e-7">2.0089</value>
            <value variable="rhov" tolerance="1e-7">0.00628555</value>
            <value variable="E"    tolerance="1e-6">6.56585</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="rho"  tolerance="1e-7">1.19768</value>
            <value variable="rhou" tolerance="1e-7">1.21074</value>
            <value variable="rhov" tolerance="1e-6">0.0168045</value>
            <value variable="E"    tolerance="1e-6">3.58069</value>
        </metric>
    </metrics>
</test>
