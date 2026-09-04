<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>Magnetohydrodynamic flow around a moving circular cylinder with implicit Lorentz damping</description>
    <executable>IncNavierStokesSolver</executable>
    <parameters>Cyl2DMesh.xml MHDMovingCylinderImplicitLorentz.xml</parameters>
    <files>
        <file description="Mesh File">Cyl2DMesh.xml</file>
        <file description="Session File">MHDMovingCylinderImplicitLorentz.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="5e-4">0.00736499</value>
            <value variable="v" tolerance="5e-4">0.00723379</value>
            <value variable="phi" tolerance="1e-12">0</value>
            <value variable="p" tolerance="5e-2">2.73834</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="2e-3">0.0106745</value>
            <value variable="v" tolerance="5e-4">0.00903534</value>
            <value variable="phi" tolerance="1e-12">0</value>
            <value variable="p" tolerance="5e-2">0.439727</value>
        </metric>
    </metrics>
</test>
