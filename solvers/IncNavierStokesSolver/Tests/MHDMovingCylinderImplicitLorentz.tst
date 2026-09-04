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
            <value variable="u" tolerance="5e-4">0.0117921</value>
            <value variable="v" tolerance="5e-4">0.0118397</value>
            <value variable="phi" tolerance="1e-12">0</value>
            <value variable="p" tolerance="5e-2">0.460653</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="2e-3">0.0599995</value>
            <value variable="v" tolerance="5e-4">0.0501</value>
            <value variable="phi" tolerance="1e-12">0</value>
            <value variable="p" tolerance="5e-2">0.332435</value>
        </metric>
    </metrics>
</test>
