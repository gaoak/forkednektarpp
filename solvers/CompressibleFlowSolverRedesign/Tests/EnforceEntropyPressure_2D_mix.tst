<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>
    Subsonic Riemann boundary holding the entropy and the pressure, on a mixed
    quadrilateral/triangle mesh.

    This is the combination the reference singles out: of the ways to fill the
    degree of freedom a subsonic boundary leaves, enforcing the entropy with the
    pressure is the one a linearised analysis shows to be stable for imposing a
    known pressure. Unlike the entropy-velocity condition it also has a genuine
    subsonic outflow branch, which imposes the prescribed pressure and takes the
    density from the interior along an isentrope, so this case exercises both
    ends of the domain rather than only the inflow.

    Built on legacy's square_mix.xml with the conditions derived from
    session_enforceEntropyVelocity.xml by swapping the tag, not copied from
    session_enforceEntropyPressure.xml, which differs: legacy's own
    entropy-pressure case puts a plain Dirichlet at the outlet and comments the
    PressureOutflow region out. This file keeps the outflow, so it covers an
    entropy inflow and a pressure outflow together. The comparison below is
    therefore against legacy running *this* file rather than against legacy's
    published numbers for a different case.

    The other three changes are shared with its entropy-velocity sibling: the
    equation type moves from NavierStokesImplicitCFE to NavierStokesCFE because
    the redesign has no implicit path, the timestep drops to the explicit limit
    with it, and the initial condition carries a bump.

    That bump is what makes this a test of the condition rather than of the
    session values. With the freestream initial state legacy uses, these
    conditions and a plain Dirichlet at the same values give byte-identical
    answers - a uniform state is preserved either way. Perturbed, the three
    entropy conditions separate from each other and from a plain Dirichlet: rho
    L2 is 2.01186, 2.01714 and 2.0147 for the velocity, pressure and total
    enthalpy variants against 2.01447 for a plain Dirichlet, so each of these
    files pins its own condition.

    Verified against legacy on this same file, which both solvers accept
    unchanged: every L2 and Linf value agrees to all printed digits.
    </description>
    <executable>CompressibleFlowSolverRedesign</executable>
    <parameters>EnforceEntropyPressure_2D_mix.xml</parameters>
    <files>
        <file description="Session File">EnforceEntropyPressure_2D_mix.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="rho"  tolerance="1e-7">2.01714</value>
            <value variable="rhou" tolerance="1e-7">2.02085</value>
            <value variable="rhov" tolerance="1e-8">0.00717563</value>
            <value variable="E"    tolerance="1e-6">6.59667</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="rho"  tolerance="1e-7">1.18419</value>
            <value variable="rhou" tolerance="1e-7">1.29682</value>
            <value variable="rhov" tolerance="1e-7">0.0220049</value>
            <value variable="E"    tolerance="1e-6">3.85262</value>
        </metric>
    </metrics>
</test>
