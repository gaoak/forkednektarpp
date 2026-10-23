<?xml version="1.0" encoding="utf-8"?>
<!--
    ChannelFlowHOPBC2D in the weak pressure form. The plane Poiseuille flow
    is an exact steady solution for which the weak-form high-order pressure
    boundary flux, the viscous curl-curl term, is exact, so the discrete
    solution stays at the initial condition to close to machine precision.
-->
<test>
    <description>ChannelFlowHOPBC2D_WeakPressure: plane Poiseuille flow with high-order pressure BC at walls and inflow in the weak pressure form</description>
    <executable>IncNavierStokesSolverRedesign</executable>
    <parameters>-I PressureFormulation=Weak ChannelFlowHOPBC2D.xml</parameters>
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
