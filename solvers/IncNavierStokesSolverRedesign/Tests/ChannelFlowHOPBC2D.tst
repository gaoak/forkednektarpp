<?xml version="1.0" encoding="utf-8"?>
<!--
    The session starts from an exact steady solution, so these metrics are
    errors. With the higher-order pressure boundary condition they stay at
    about 1e-13; a zero-flux Neumann pressure condition on the same regions
    gives L2 errors of 1e-2 to 1e-1.
-->
<test>
    <description>ChannelFlowHOPBC2D: plane Poiseuille flow with Neumann/high-order pressure BC at walls and inflow</description>
    <executable>IncNavierStokesSolverRedesign</executable>
    <parameters>ChannelFlowHOPBC2D.xml</parameters>
    <files>
        <file description="Session File">ChannelFlowHOPBC2D.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="1e-9">0</value>
            <value variable="v" tolerance="1e-9">0</value>
            <value variable="p" tolerance="1e-9">0</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="1e-9">0</value>
            <value variable="v" tolerance="1e-9">0</value>
            <value variable="p" tolerance="1e-9">0</value>
        </metric>
    </metrics>
</test>
