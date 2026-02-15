<?xml version="1.0" encoding="utf-8"?>
<test>
    <description> 3D SteadyADRRedesign CG implicit diffusion with Redesign operators </description>
    <executable>RedesignSolver</executable>
    <parameters> SteadyAdvecDiff3DBaseFlow.xml</parameters>
    <files>
        <file description="Session File"> SteadyAdvecDiff3DBaseFlow.xml </file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="1e-10"> 7.3e-09 </value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="1e-10"> 2.316e-07 </value>
        </metric>
    </metrics>
</test>
