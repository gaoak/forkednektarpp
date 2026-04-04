<?xml version="1.0" encoding="utf-8"?>
<test>
    <description> 3D unsteady CG implicit diffusion with Redesign operators </description>
    <executable>ADRSolverRedesign</executable>
    <parameters> UnsteadyDiffusion3D.xml</parameters>
    <files>
        <file description="Session File"> UnsteadyDiffusion3D.xml </file>
    </files>
    <metrics>
        <metric type="L2" id="1">
	        <value variable="u" tolerance="1e-8"> 5.50e-08 </value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="3e-8"> 4.87e-08 </value>
        </metric>
    </metrics>
</test>
