<?xml version="1.0" encoding="utf-8"?>
<!--
    ChannelFlowHOPBC2D_MixedOrder in the weak pressure form. The plane
    Poiseuille flow is exact in the P=2 velocity and P=1 pressure spaces,
    and the errors stay below 4e-14 over Serial, StdMat and Device in both
    formulations; the bounds are those of ChannelFlowHOPBC2D_WeakPressure.
    Subtracting the projection of the inflow profile onto the pressure
    boundary expansion instead raises the errors to 0.5 for u and 2.6 for p.
-->
<test>
    <description>ChannelFlowHOPBC2D_MixedOrder_WeakPressure: plane Poiseuille flow with high-order pressure BC at walls and inflow and a pressure one order below the velocity in the weak pressure form</description>
    <executable>IncNavierStokesSolverRedesign</executable>
    <parameters>-I PressureFormulation=Weak ChannelFlowHOPBC2D.xml ChannelFlowHOPBC2D_MixedOrder.xml</parameters>
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
