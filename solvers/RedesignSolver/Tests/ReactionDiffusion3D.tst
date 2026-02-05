<?xml version="1.0" encoding="utf-8"?>
<test>
    <description> 3D unsteady implicit diffusion with explicit reaction using Redesign operators </description>
    <executable>RedesignSolver</executable>
    <parameters> ReactionDiffusion3D.xml</parameters>
    <files>
        <file description="Session File"> ReactionDiffusion3D.xml </file>
    </files>
    <metrics>
        <metric type="L2" id="1">
	        <value variable="u" tolerance="8e-6"> 3.55654e-05 </value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="6e-6"> 2.94899e-05 </value>
        </metric>
    </metrics>
</test>
