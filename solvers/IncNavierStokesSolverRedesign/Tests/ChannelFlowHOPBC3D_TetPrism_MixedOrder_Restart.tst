<?xml version="1.0" encoding="utf-8"?>
<!--
    The first segment is ChannelFlowHOPBC3D_TetPrism_MixedOrder_FileIC. It
    writes its state at a P=2 velocity and a P=1 pressure, the pressure
    through the velocity expansion with its P=2 modes zero. The second
    segment restarts from that file at the same orders and takes one
    semi-implicit step, so the pressure read back must equal the P=1
    pressure written.
-->
<test>
    <description>ChannelFlowHOPBC3D_TetPrism_MixedOrder_Restart: restart from a field file with a P=2 velocity and a P=1 pressure written through the velocity expansion, one step</description>
    <segment type="sequential">
        <executable>IncNavierStokesSolverRedesign</executable>
        <parameters>ChannelFlowHOPBC3D_TetPrism_FileIC.xml ChannelFlowHOPBC3D_TetPrism_MixedOrder.xml</parameters>
        <processes>1</processes>
    </segment>
    <segment type="sequential">
        <executable>IncNavierStokesSolverRedesign</executable>
        <parameters>ChannelFlowHOPBC3D_TetPrism_FileIC.xml ChannelFlowHOPBC3D_TetPrism_MixedOrder_Restart.xml</parameters>
        <processes>1</processes>
    </segment>
    <files>
        <file description="Session File">ChannelFlowHOPBC3D_TetPrism_FileIC.xml</file>
        <file description="Mixed-order expansions">ChannelFlowHOPBC3D_TetPrism_MixedOrder.xml</file>
        <file description="Restart conditions">ChannelFlowHOPBC3D_TetPrism_MixedOrder_Restart.xml</file>
        <file description="Initial Condition">ChannelFlowHOPBC3D_TetPrism_FileIC_ini.fld</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="1e-6">0.109805</value>
            <value variable="v" tolerance="1e-6">0.0772504</value>
            <value variable="w" tolerance="1e-6">0.119239</value>
            <value variable="p" tolerance="1e-5">4.69382</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="1e-6">0.2</value>
            <value variable="v" tolerance="1e-6">0.1</value>
            <value variable="w" tolerance="1e-6">0.15</value>
            <value variable="p" tolerance="1e-5">6.0391</value>
        </metric>
    </metrics>
</test>
