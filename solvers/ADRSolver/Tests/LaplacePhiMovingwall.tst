<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>Prescribed uniform wall velocity on curved NACA0012 edges</description>
    <executable>ADRSolver</executable>
    <parameters>NACA0012_Re400.xml LaplacePhiMovingwall.xml</parameters>
    <files>
        <file description="Mesh File">NACA0012_Re400.xml</file>
        <file description="Session File">LaplacePhiMovingwall.xml</file>
        <file description="Endpoint velocities">Movingwall.dat</file>
    </files>
    <metrics>
        <metric type="L2" id="1"><value variable="phi0" tolerance="1e-6">1.3798686e-01</value></metric>
        <metric type="Linf" id="2"><value variable="phi0" tolerance="1e-6">1.5289871e-01</value></metric>
    </metrics>
</test>
