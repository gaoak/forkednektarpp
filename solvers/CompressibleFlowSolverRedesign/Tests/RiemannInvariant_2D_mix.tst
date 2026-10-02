<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>
    Characteristic farfield from the Riemann invariants, on a mixed
    quadrilateral/triangle mesh, with slip walls above and below.

    Built on legacy's square_mix.xml rather than ported from a single legacy
    case: every legacy session using RiemannInvariant also needs something not
    yet implemented. Verified instead against legacy running this same file,
    which it accepts unchanged, and every L2 and Linf value agrees to all
    printed digits.

    Two things about the setup are deliberate and both took a false start to
    get right.

    The freestream comes from the session parameters rhoInf, pInf, uInf and
    vInf, not from the region's VALUE, which this condition ignores entirely.
    That is legacy's convention and it is why this condition does not share the
    machinery of the entropy Riemann conditions, which read theirs from the
    boundary values.

    The perturbation sits on the inflow boundary rather than mid-domain. With
    it in the middle the farfield and a plain freestream Dirichlet agree to five
    or six digits, and rightly so: where the interior is already at the
    freestream, the characteristic reconstruction returns the freestream, and a
    test of that is a test of nothing. Placing it on the boundary gives the
    reconstruction a perturbed interior to work with, and the two then separate
    - rhov L2 0.00665988 against 0.00658314, rhou 2.06173 against 2.06186.

    A second, independent check that the condition is live rather than
    coincidentally right: raising uInf from 1.0 to 1.2 moves rhov L2 from
    0.00502 to 0.00725 on the mid-domain variant. The freestream parameters are
    read and used.
    </description>
    <executable>CompressibleFlowSolverRedesign</executable>
    <parameters>RiemannInvariant_2D_mix.xml</parameters>
    <files>
        <file description="Session File">RiemannInvariant_2D_mix.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="rho"  tolerance="1e-7">2.06207</value>
            <value variable="rhou" tolerance="1e-7">2.06173</value>
            <value variable="rhov" tolerance="1e-8">0.00665988</value>
            <value variable="E"    tolerance="1e-6">6.5812</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="rho"  tolerance="1e-7">1.33945</value>
            <value variable="rhou" tolerance="1e-7">1.35082</value>
            <value variable="rhov" tolerance="1e-7">0.0134992</value>
            <value variable="E"    tolerance="1e-6">3.3353</value>
        </metric>
    </metrics>
</test>
