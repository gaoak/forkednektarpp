<?xml version="1.0" encoding="utf-8"?>
<test>
    <description> 1D Poisson CG implicit diffusion with Redesign operators </description>
    <executable>RedesignSolver</executable>
    <parameters> SteadyDiffusion1D.xml</parameters>
    <files>
        <file description="Session File"> SteadyDiffusion1D.xml </file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="1e-9"> 1.19628e-16 </value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="1e-9"> 3.33067e-16 </value>
        </metric>
    </metrics>
</test>
