<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>2D unsteady IP explicit diffusion, order 4, P=8</description>
    <executable>ADRSolverRedesign</executable>
    <parameters>ExDiffusion_2D_IP_hybrid_m8.xml</parameters>
    <files>
        <file description="Session File">ExDiffusion_2D_IP_hybrid_m8.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="6e-8">6.01062e-05</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="8e-6">0.00078869</value>
        </metric>
    </metrics>
</test>
