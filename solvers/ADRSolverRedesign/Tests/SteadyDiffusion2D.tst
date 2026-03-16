<?xml version="1.0" encoding="utf-8"?>
<test>
    <description> 2D Poisson CG implicit diffusion with Redesign operators </description>
    <executable>ADRSolverRedesign</executable>
    <parameters> SteadyDiffusion2D.xml</parameters>
    <files>
        <file description="Session File"> SteadyDiffusion2D.xml </file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="1e-10"> 1.5e-09 </value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="1e-10"> 2.4e-09 </value>
        </metric>
    </metrics>
</test>
