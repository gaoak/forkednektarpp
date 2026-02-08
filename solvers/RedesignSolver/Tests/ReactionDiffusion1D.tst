<?xml version="1.0" encoding="utf-8"?>
<test>
    <description> 1D unsteady implicit diffusion with explicit reaction using Redesign operators </description>
    <executable>RedesignSolver</executable>
    <parameters> ReactionDiffusion1D.xml</parameters>
    <files>
        <file description="Session File"> ReactionDiffusion1D.xml </file>
    </files>
    <metrics>
        <metric type="L2" id="1">
	        <value variable="u" tolerance="2e-11">  8.6e-10 </value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="2e-11"> 1.38e-09 </value>
        </metric>
    </metrics>
</test>
