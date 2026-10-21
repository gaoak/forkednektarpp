<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>ChannelFlowHOPBC3D_Mixed: pressure-driven channel flow on a mixed hex/prism mesh</description>
    <executable>IncNavierStokesSolverRedesign</executable>
    <parameters>ChannelFlowHOPBC3D_Mixed.xml</parameters>
    <files>
        <file description="Session File">ChannelFlowHOPBC3D_Mixed.xml</file>
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
