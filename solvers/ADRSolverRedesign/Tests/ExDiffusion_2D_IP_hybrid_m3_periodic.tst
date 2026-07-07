<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>2D unsteady IP explicit diffusion, periodic in x</description>
    <executable>ADRSolverRedesign</executable>
    <parameters>ExDiffusion_2D_IP_hybrid_m3_periodic.xml</parameters>
    <files>
        <file description="Session File">ExDiffusion_2D_IP_hybrid_m3_periodic.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="5e-7">0.000873111</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="1e-12">0.00283018</value>
        </metric>
    </metrics>
</test>
