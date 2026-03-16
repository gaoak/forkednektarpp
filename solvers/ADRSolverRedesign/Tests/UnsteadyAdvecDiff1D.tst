<?xml version="1.0" encoding="utf-8"?>
<test>
    <description> 1D UnsteadyADRRedesign CG implicit diffusion with Redesign operators </description>
    <executable>ADRSolverRedesign</executable>
    <parameters> UnsteadyAdvecDiff1D.xml</parameters>
    <files>
        <file description="Session File"> UnsteadyAdvecDiff1D.xml </file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="1e-9"> 1.75744e-10 </value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="1e-9"> 2.50852e-10 </value>
        </metric>
    </metrics>
</test>
