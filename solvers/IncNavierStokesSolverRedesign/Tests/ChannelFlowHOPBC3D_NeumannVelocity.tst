<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>ChannelFlowHOPBC3D_NeumannVelocity: pressure-driven channel flow on a mixed hex/prism mesh, with HOP pressure BCs on an all-Neumann velocity inflow and on slip walls</description>
    <executable>IncNavierStokesSolverRedesign</executable>
    <parameters>ChannelFlowHOPBC3D_NeumannVelocity.xml</parameters>
    <files>
        <file description="Session File">ChannelFlowHOPBC3D_NeumannVelocity.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="1e-6">0</value>
            <value variable="v" tolerance="1e-6">0</value>
            <value variable="w" tolerance="1e-6">0</value>
            <value variable="p" tolerance="1e-6">0</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="1e-6">0</value>
            <value variable="v" tolerance="1e-6">0</value>
            <value variable="w" tolerance="1e-6">0</value>
            <value variable="p" tolerance="1e-6">0</value>
        </metric>
    </metrics>
</test>
