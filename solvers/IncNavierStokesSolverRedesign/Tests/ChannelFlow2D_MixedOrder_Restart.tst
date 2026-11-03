<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>
        The first segment is ChannelFlow2D_MixedOrder. It writes its state at
        a P=3 velocity and a P=2 pressure to ChannelFlow2D.fld, the pressure
        through the velocity expansion with its P=3 modes zero. The second
        segment restarts from that file at the same orders and takes no step,
        so the metrics are the errors of the state read back. Writing the P=2
        coefficients under the velocity's definitions without converting
        them reads back with an L2 error of 4.6.
    </description>
    <segment type="sequential">
        <executable>IncNavierStokesSolverRedesign</executable>
        <parameters>ChannelFlow2D.xml ChannelFlow2D_MixedOrder.xml</parameters>
        <processes>1</processes>
    </segment>
    <segment type="sequential">
        <executable>IncNavierStokesSolverRedesign</executable>
        <parameters>ChannelFlow2D.xml ChannelFlow2D_MixedOrder.xml ChannelFlow2D_MixedOrder_Restart.xml</parameters>
        <processes>1</processes>
    </segment>
    <files>
        <file description="Session File">ChannelFlow2D.xml</file>
        <file description="Mixed-order expansions">ChannelFlow2D_MixedOrder.xml</file>
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
