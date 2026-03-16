<?xml version="1.0" encoding="utf-8"?>
<test>
    <description> 3D Poisson CG implicit diffusion with Redesign operators </description>
    <executable>ADRSolverRedesign</executable>
    <parameters> SteadyDiffusion3D.xml</parameters>
    <files>
        <file description="Session File"> SteadyDiffusion3D.xml </file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="5e-09"> 4.0e-09 </value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="5e-09"> 8.0e-09 </value>
        </metric>
    </metrics>
</test>
