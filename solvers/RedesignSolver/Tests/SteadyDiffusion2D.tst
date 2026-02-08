<?xml version="1.0" encoding="utf-8"?>
<test>
    <description> 2D Poisson CG implicit diffusion with Redesign operators </description>
    <executable>RedesignSolver</executable>
    <parameters> SteadyDiffusion2D.xml</parameters>
    <files>
        <file description="Session File"> SteadyDiffusion2D.xml </file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="1e-8"> 1.19628e-16 </value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="1e-8"> 3.33067e-16 </value>
        </metric>
    </metrics>
</test>
