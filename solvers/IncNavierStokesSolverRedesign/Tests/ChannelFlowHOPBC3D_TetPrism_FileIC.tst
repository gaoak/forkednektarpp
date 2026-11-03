<?xml version="1.0" encoding="utf-8"?>
<!--
    Control for ChannelFlowHOPBC3D_TetPrism_MixedOrder_FileIC at equal order.
    Every field is P=2, and the P=3 file is truncated mode by mode. No step
    is taken, so the norms are those of the processed initial condition
    against the cubic state that the file holds.
-->
<test>
    <description>ChannelFlowHOPBC3D_TetPrism_FileIC: equal-order P=2 initial condition read from a P=3 field file</description>
    <executable>IncNavierStokesSolverRedesign</executable>
    <parameters>ChannelFlowHOPBC3D_TetPrism_FileIC.xml</parameters>
    <files>
        <file description="Session File">ChannelFlowHOPBC3D_TetPrism_FileIC.xml</file>
        <file description="Initial Condition">ChannelFlowHOPBC3D_TetPrism_FileIC_ini.fld</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="1e-6">0.111955</value>
            <value variable="v" tolerance="1e-6">0.0769577</value>
            <value variable="w" tolerance="1e-6">0.122344</value>
            <value variable="p" tolerance="1e-6">0.0299158</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="1e-6">0.2</value>
            <value variable="v" tolerance="1e-6">0.1</value>
            <value variable="w" tolerance="1e-6">0.15</value>
            <value variable="p" tolerance="1e-6">0.0447214</value>
        </metric>
    </metrics>
</test>
