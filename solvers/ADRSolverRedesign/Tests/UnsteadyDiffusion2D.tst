<?xml version="1.0" encoding="utf-8"?>
<test>
    <description> 2D unsteady CG implicit diffusion with Redesign operators </description>
    <executable>ADRSolverRedesign</executable>
    <parameters> UnsteadyDiffusion2D.xml</parameters>
    <files>
        <file description="Session File"> UnsteadyDiffusion2D.xml </file>
    </files>
    <metrics>
        <metric type="L2" id="1">
	        <value variable="u" tolerance="1e-8"> 1.82e-08 </value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="6e-8"> 6.04e-08 </value>
        </metric>
    </metrics>
</test>
