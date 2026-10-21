<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>ChannelFlowHOPBC3D_TetPrism: non-zero HOP pressure flux on mixed tetrahedron/prism boundary faces</description>
    <executable>IncNavierStokesSolverRedesign</executable>
    <parameters>ChannelFlowHOPBC3D_TetPrism.xml</parameters>
    <files>
        <file description="Session File">ChannelFlowHOPBC3D_TetPrism.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="1e-6">0</value>
            <value variable="v" tolerance="1e-6">0</value>
            <value variable="w" tolerance="1e-6">0</value>
            <value variable="p" tolerance="1e-6">0</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="1e-6">0</value>
            <value variable="v" tolerance="1e-6">0</value>
            <value variable="w" tolerance="1e-6">0</value>
            <value variable="p" tolerance="1e-6">0</value>
        </metric>
    </metrics>
</test>
