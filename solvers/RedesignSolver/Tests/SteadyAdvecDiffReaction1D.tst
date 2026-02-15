<?xml version="1.0" encoding="utf-8"?>
<test>
    <description> 1D SteadyADRRedesign CG implicit diffusion with Redesign operators </description>
    <executable>RedesignSolver</executable>
    <parameters> SteadyAdvecDiffReaction1D.xml</parameters>
    <files>
        <file description="Session File"> SteadyAdvecDiffReaction1D.xml </file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="1e-12"> 5.0e-12 </value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="1e-12"> 1.37e-11 </value>
        </metric>
    </metrics>
</test>
