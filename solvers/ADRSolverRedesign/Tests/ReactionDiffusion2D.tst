<?xml version="1.0" encoding="utf-8"?>
<test>
    <description> 2D unsteady implicit diffusion with explicit reaction using Redesign operators </description>
    <executable>ADRSolverRedesign</executable>
    <parameters> ReactionDiffusion2D.xml</parameters>
    <files>
        <file description="Session File"> ReactionDiffusion2D.xml </file>
    </files>
    <metrics>
        <metric type="L2" id="1">
	        <value variable="u" tolerance="2.5e-9"> 9e-09 </value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="2.5e-9"> 2.9e-08 </value>
        </metric>
    </metrics>
</test>
