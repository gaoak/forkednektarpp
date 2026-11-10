<?xml version="1.0" encoding="utf-8"?>
<test>
    <description> 2D Poisson with a spatially varying full diffusion tensor </description>
    <executable>ADRSolverRedesign</executable>
    <parameters> SteadyVariableDiffusion2D.xml</parameters>
    <files>
        <file description="Session File"> SteadyVariableDiffusion2D.xml </file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="1e-9"> 0 </value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="1e-9"> 0 </value>
        </metric>
    </metrics>
</test>
