<?xml version="1.0" encoding="utf-8"?>
<!--
    ChannelFlowHOPBC2D with a P=2 velocity and a P=1 pressure that share
    their quadrature points. The plane Poiseuille flow is exact in these
    spaces, and the errors stay below 1.3e-13 over Serial, StdMat and Device
    in both formulations; the bounds are those of ChannelFlowHOPBC2D. The
    quadratic inflow profile on the high-order pressure boundary is exact in
    the velocity boundary space but not in the pressure one; subtracting its
    projection onto the pressure boundary expansion instead raises the errors
    to 0.5 for u and 3 for p.
-->
<test>
    <description>ChannelFlowHOPBC2D_MixedOrder: plane Poiseuille flow with high-order pressure BC at walls and inflow and a pressure one order below the velocity</description>
    <executable>IncNavierStokesSolverRedesign</executable>
    <parameters>ChannelFlowHOPBC2D.xml ChannelFlowHOPBC2D_MixedOrder.xml</parameters>
    <files>
        <file description="Mesh and base session">ChannelFlowHOPBC2D.xml</file>
        <file description="Mixed-order expansions">ChannelFlowHOPBC2D_MixedOrder.xml</file>
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
