<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>
        The first segment runs ChannelFlow2D with a P=2 velocity and a P=3
        pressure. It writes its state to ChannelFlow2D.fld, where the
        pressure, which does not embed in the velocity expansion, is stored
        in field definitions of its own. The second segment restarts from
        that file at the same orders and takes no step, so the metrics are
        the errors of the state read back. The plane Poiseuille flow is
        exact in these spaces.
    </description>
    <segment type="sequential">
        <executable>IncNavierStokesSolverRedesign</executable>
        <parameters>ChannelFlow2D.xml ChannelFlow2D_HigherOrderPressure.xml</parameters>
        <processes>1</processes>
    </segment>
    <segment type="sequential">
        <executable>IncNavierStokesSolverRedesign</executable>
        <parameters>ChannelFlow2D.xml ChannelFlow2D_HigherOrderPressure.xml ChannelFlow2D_MixedOrder_Restart.xml</parameters>
        <processes>1</processes>
    </segment>
    <files>
        <file description="Session File">ChannelFlow2D.xml</file>
        <file description="Higher-order pressure expansions">ChannelFlow2D_HigherOrderPressure.xml</file>
        <file description="Restart conditions">ChannelFlow2D_MixedOrder_Restart.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="1e-10">0</value>
            <value variable="v" tolerance="1e-10">0</value>
            <value variable="p" tolerance="1e-10">0</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="1e-10">0</value>
            <value variable="v" tolerance="1e-10">0</value>
            <value variable="p" tolerance="1e-10">0</value>
        </metric>
    </metrics>
</test>
