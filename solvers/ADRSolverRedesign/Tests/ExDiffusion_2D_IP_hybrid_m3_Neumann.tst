<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>2D unsteady IP explicit diffusion, P=3, all-Neumann BCs on the exact steady solution u = x</description>
    <executable>ADRSolverRedesign</executable>
    <parameters>ExDiffusion_2D_IP_hybrid_m3_Neumann.xml</parameters>
    <files>
        <file description="Session File">ExDiffusion_2D_IP_hybrid_m3_Neumann.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="1e-12">0.0</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="1e-12">0.0</value>
        </metric>
    </metrics>
</test>
