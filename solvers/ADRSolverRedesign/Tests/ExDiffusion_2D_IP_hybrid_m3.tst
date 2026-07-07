<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>2D unsteady IP explicit diffusion, order 4, P=3</description>
    <executable>ADRSolverRedesign</executable>
    <parameters>ExDiffusion_2D_IP_hybrid_m3.xml</parameters>
    <files>
        <file description="Session File">ExDiffusion_2D_IP_hybrid_m3.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="5e-6">0.00440033</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="2e-7">0.0137002</value>
        </metric>
    </metrics>
</test>
