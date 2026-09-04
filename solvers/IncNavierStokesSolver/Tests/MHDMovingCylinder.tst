<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>Magnetohydrodynamic flow around a moving circular cylinder</description>
    <executable>IncNavierStokesSolver</executable>
    <parameters>Cyl2DMesh.xml MHDMovingCylinder.xml</parameters>
    <files>
        <file description="Mesh File">Cyl2DMesh.xml</file>
        <file description="Session File">MHDMovingCylinder.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="4.3e-5">0.01178135</value>
            <value variable="v" tolerance="4.3e-5">0.01182895</value>
            <value variable="phi" tolerance="1e-12">0</value>
            <value variable="p" tolerance="0.003438">0.4597935</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="1e-7">0.0599995</value>
            <value variable="v" tolerance="1e-4">0.0501</value>
            <value variable="phi" tolerance="1e-12">0</value>
            <value variable="p" tolerance="0.002608">0.331783</value>
        </metric>
    </metrics>
</test>
