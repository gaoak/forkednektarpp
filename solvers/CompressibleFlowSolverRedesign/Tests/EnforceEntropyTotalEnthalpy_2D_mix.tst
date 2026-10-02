<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>
    Subsonic Riemann boundary holding the entropy and the total enthalpy, on a mixed
    quadrilateral/triangle mesh.

    The stagnation variant, for when the upstream reservoir is what is known
    rather than a static quantity. The sound speed is no longer available
    directly and comes from a quadratic, which is the one piece of algebra here
    not shared with its two siblings; its subsonic outflow is the same
    isentropic pressure branch the entropy-pressure condition uses.

    Built on legacy's square_mix.xml with the conditions derived from
    session_enforceEntropyVelocity.xml by swapping the tag, which for this
    condition is exactly session_enforceEntropyTotalEnthalpy.xml - the two
    legacy sessions differ in nothing else. The same three changes as its
    entropy-velocity sibling apply: the equation type moves from
    NavierStokesImplicitCFE to NavierStokesCFE because the redesign has no
    implicit path, the timestep drops to the explicit limit with it, and the
    initial condition carries a bump.

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
    <parameters>EnforceEntropyTotalEnthalpy_2D_mix.xml</parameters>
    <files>
        <file description="Session File">EnforceEntropyTotalEnthalpy_2D_mix.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="rho"  tolerance="1e-7">2.0147</value>
            <value variable="rhou" tolerance="1e-7">2.01496</value>
            <value variable="rhov" tolerance="1e-8">0.00505151</value>
            <value variable="E"    tolerance="1e-6">6.58168</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="rho"  tolerance="1e-7">1.19018</value>
            <value variable="rhou" tolerance="1e-7">1.19307</value>
            <value variable="rhov" tolerance="1e-7">0.00851642</value>
            <value variable="E"    tolerance="1e-6">3.33323</value>
        </metric>
    </metrics>
</test>
