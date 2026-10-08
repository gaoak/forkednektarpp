<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>
    2D unsteady IP explicit diffusion, periodic in x.

    Values updated alongside ExDiffusion_2D_IP_hybrid_m3 when the Dirichlet
    boundary treatment changed to the standard SIPG one; the two non-periodic
    regions here are affected in the same way. See that file, and
    library/Operators/BndCondOps/EXTERIOR_STATE_CONVENTION.md.
    </description>
    <executable>ADRSolverRedesign</executable>
    <parameters>ExDiffusion_2D_IP_hybrid_m3_periodic.xml</parameters>
    <files>
        <file description="Session File">ExDiffusion_2D_IP_hybrid_m3_periodic.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="1e-8">0.000880039</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="1e-8">0.00283322</value>
        </metric>
    </metrics>
</test>
