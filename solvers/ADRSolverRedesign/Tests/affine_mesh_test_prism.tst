<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>3D affine unsteady advection-diffusion consistency on a prism mesh with 3 components</description>
    <executable>ADRSolverRedesign</executable>
    <parameters>affine_mesh_test_prism.xml</parameters>
    <files>
        <file description="Session File">affine_mesh_test_prism.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="1e-10">0</value>
            <value variable="v" tolerance="1e-10">0</value>
            <value variable="w" tolerance="1e-10">0</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="1e-10">0</value>
            <value variable="v" tolerance="1e-10">0</value>
            <value variable="w" tolerance="1e-10">0</value>
        </metric>
    </metrics>
</test>
