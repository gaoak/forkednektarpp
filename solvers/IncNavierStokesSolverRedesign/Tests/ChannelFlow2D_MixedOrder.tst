<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>
        ChannelFlow2D with a P=3 velocity and a P=2 pressure that share their
        quadrature points. The plane Poiseuille flow is exact in these spaces.
    </description>
    <executable>IncNavierStokesSolverRedesign</executable>
    <parameters>ChannelFlow2D.xml ChannelFlow2D_MixedOrder.xml</parameters>
    <files>
        <file description="Session File">ChannelFlow2D.xml</file>
        <file description="Mixed-order expansions">ChannelFlow2D_MixedOrder.xml</file>
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
