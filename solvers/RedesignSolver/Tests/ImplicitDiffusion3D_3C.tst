<?xml version="1.0" encoding="utf-8"?>
<test>
    <description> 3D unsteady CG implicit diffusion with 3 components </description>
    <executable>RedesignSolver</executable>
    <parameters> ImplicitDiffusion3D_3C.xml</parameters>
    <files>
        <file description="Session File"> ImplicitDiffusion3D_3C.xml </file>
    </files>
    <metrics>
        <metric type="L2" id="1">
           <value variable="u" tolerance="2e-8"> 5.47870e-08 </value>
           <value variable="v" tolerance="1e-8"> 2.73935e-08 </value>
           <value variable="w" tolerance="1e-8"> 1.36967e-08 </value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="2e-7"> 4.86119e-08 </value>
            <value variable="v" tolerance="6e-8"> 2.43060e-08 </value>
            <value variable="w" tolerance="3e-8"> 1.21530e-08 </value>
        </metric>
    </metrics>
</test>
