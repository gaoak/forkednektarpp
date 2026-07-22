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
            <value variable="u" tolerance="5e-4">0.00732911</value>
            <value variable="v" tolerance="5e-4">0.0071968</value>
            <value variable="phi" tolerance="1e-12">0</value>
            <value variable="p" tolerance="5e-2">2.72804</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="2e-3">0.0106152</value>
            <value variable="v" tolerance="5e-4">0.00898934</value>
            <value variable="phi" tolerance="1e-12">0</value>
            <value variable="p" tolerance="5e-2">0.437964</value>
        </metric>
    </metrics>
</test>
