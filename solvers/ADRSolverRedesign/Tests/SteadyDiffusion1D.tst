<?xml version="1.0" encoding="utf-8"?>
<test>
    <description> 1D Poisson CG implicit diffusion with Redesign operators </description>
    <executable>ADRSolverRedesign</executable>
    <parameters> SteadyDiffusion1D.xml</parameters>
    <files>
        <file description="Session File"> SteadyDiffusion1D.xml </file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="1e-12"> 0.0 </value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="1e-12"> 0.0 </value>
        </metric>
    </metrics>
</test>
