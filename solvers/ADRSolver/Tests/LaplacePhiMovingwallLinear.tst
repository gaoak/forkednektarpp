<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>Movingwall endpoint interpolation and outward normal flux: phi=x*y</description>
    <executable>ADRSolver</executable>
    <parameters>LaplacePhiMovingwallLinear.xml</parameters>
    <files>
        <file description="Session File">LaplacePhiMovingwallLinear.xml</file>
        <file description="Endpoint velocities">MovingwallLinear.dat</file>
    </files>
    <metrics>
        <metric type="L2" id="1"><value variable="phi0" tolerance="1e-10">0</value></metric>
        <metric type="Linf" id="2"><value variable="phi0" tolerance="1e-10">0</value></metric>
    </metrics>
</test>
