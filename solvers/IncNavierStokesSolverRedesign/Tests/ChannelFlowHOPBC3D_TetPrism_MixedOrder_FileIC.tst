<?xml version="1.0" encoding="utf-8"?>
<!--
    ChannelFlowHOPBC3D_TetPrism_FileIC with a P=2 velocity and a P=1
    pressure. No step is taken, so the norms are those of the processed
    initial condition against the cubic state that the P=3 file holds. The
    pressure is read through the velocity expansion, projected element by
    element onto the pressure expansion, averaged, and given its Dirichlet
    values on the outflow. Truncating the file mode by mode to P=1 instead,
    which keeps only the vertex values, gives an L2 pressure error of
    0.641208 rather than 0.269732.
-->
<test>
    <description>ChannelFlowHOPBC3D_TetPrism_MixedOrder_FileIC: P=2 velocity and P=1 pressure initial condition read from a P=3 field file</description>
    <executable>IncNavierStokesSolverRedesign</executable>
    <parameters>ChannelFlowHOPBC3D_TetPrism_FileIC.xml ChannelFlowHOPBC3D_TetPrism_MixedOrder.xml</parameters>
    <files>
        <file description="Session File">ChannelFlowHOPBC3D_TetPrism_FileIC.xml</file>
        <file description="Mixed-order expansions">ChannelFlowHOPBC3D_TetPrism_MixedOrder.xml</file>
        <file description="Initial Condition">ChannelFlowHOPBC3D_TetPrism_FileIC_ini.fld</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="1e-6">0.111955</value>
            <value variable="v" tolerance="1e-6">0.0769577</value>
            <value variable="w" tolerance="1e-6">0.122344</value>
            <value variable="p" tolerance="1e-6">0.269732</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="1e-6">0.2</value>
            <value variable="v" tolerance="1e-6">0.1</value>
            <value variable="w" tolerance="1e-6">0.15</value>
            <value variable="p" tolerance="1e-6">0.829092</value>
        </metric>
    </metrics>
</test>
