<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>
    Inviscid slip wall, symmetry plane and zeroth order extrapolation, on a
    mixed quadrilateral/triangle mesh.

    Three conditions at once, which is the cheapest way to cover them: the
    bottom of the channel is tagged Wall and the top Symmetry, so both tags of
    BndCondSlipWallCFEOp are exercised and shown to behave identically as legacy
    implements them; the outlet is ExtrapOrder0; the inlet is a plain Dirichlet.

    Built on legacy's square_mix.xml rather than ported from one legacy case,
    because no single legacy session covers these three without also needing
    StagnationInflow or RinglebFlow. The verification is therefore against
    legacy running this same file, which it accepts unchanged, and every L2 and
    Linf value agrees to all printed digits.

    The initial condition carries a density and momentum bump, and it has to.
    A uniform flow parallel to a slip wall has no normal momentum to mirror, so
    the condition would be indistinguishable from doing nothing; the bump
    reflects off the walls and makes it visible. Replacing the two wall regions
    with a freestream Dirichlet moves rhov L2 from 0.00501723 to 0.00497336 and
    E from 6.58033 to 6.58019, so this file does pin the mirror rather than
    merely running through it.

    ExtrapOrder0 is a different matter and worth being clear about: its Apply()
    is empty by construction, because the caller has already seeded the boundary
    storage with the interior state. What this case checks for it is that
    claiming the tag causes that seeding to happen at all - an unclaimed region
    would keep the session values instead, and since the tag guard landed that
    is a fatal error rather than a wrong answer.

    Euler rather than Navier-Stokes: a slip wall is a statement about the
    inviscid flux, and dropping the viscous terms keeps what is under test to
    the boundary conditions.
    </description>
    <executable>CompressibleFlowSolverRedesign</executable>
    <parameters>SlipWall_ExtrapOrder0_2D_mix.xml</parameters>
    <files>
        <file description="Session File">SlipWall_ExtrapOrder0_2D_mix.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="rho"  tolerance="1e-7">2.01447</value>
            <value variable="rhou" tolerance="1e-7">2.01446</value>
            <value variable="rhov" tolerance="1e-8">0.00501723</value>
            <value variable="E"    tolerance="1e-6">6.58033</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="rho"  tolerance="1e-7">1.19072</value>
            <value variable="rhou" tolerance="1e-7">1.19438</value>
            <value variable="rhov" tolerance="1e-7">0.00864813</value>
            <value variable="E"    tolerance="1e-6">3.31211</value>
        </metric>
    </metrics>
</test>
