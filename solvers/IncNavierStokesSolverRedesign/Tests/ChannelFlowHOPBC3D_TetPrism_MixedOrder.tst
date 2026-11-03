<?xml version="1.0" encoding="utf-8"?>
<!--
    ChannelFlowHOPBC3D_TetPrism with a P=2 velocity and a P=1 pressure that
    share their quadrature points. The quadratic velocity and the linear
    pressure are exact in these spaces, so the error is that of the
    iterative solves (tolerance 1e-10): at most 1.8e-9 for the velocity and
    7.7e-8 for the pressure over Serial, StdMat and Device in both
    formulations. The quadratic velocity on the high-order pressure boundary
    is exact in the velocity boundary space but not in the pressure one;
    subtracting its projection onto the pressure boundary expansion instead
    raises the errors to 0.4 for u and 550 for p after the single step of
    dt=1e-3.
-->
<test>
    <description>ChannelFlowHOPBC3D_TetPrism_MixedOrder: non-zero HOP pressure flux on mixed tetrahedron/prism boundary faces with a pressure one order below the velocity</description>
    <executable>IncNavierStokesSolverRedesign</executable>
    <parameters>ChannelFlowHOPBC3D_TetPrism.xml ChannelFlowHOPBC3D_TetPrism_MixedOrder.xml</parameters>
    <files>
        <file description="Mesh and base session">ChannelFlowHOPBC3D_TetPrism.xml</file>
        <file description="Mixed-order expansions">ChannelFlowHOPBC3D_TetPrism_MixedOrder.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="1e-8">0</value>
            <value variable="v" tolerance="1e-8">0</value>
            <value variable="w" tolerance="1e-8">0</value>
            <value variable="p" tolerance="1e-6">0</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="1e-8">0</value>
            <value variable="v" tolerance="1e-8">0</value>
            <value variable="w" tolerance="1e-8">0</value>
            <value variable="p" tolerance="1e-6">0</value>
        </metric>
    </metrics>
</test>
