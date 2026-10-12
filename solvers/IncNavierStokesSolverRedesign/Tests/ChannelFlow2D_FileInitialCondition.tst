<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>
        2D Poiseuille channel started from a field file that is
        discontinuous between elements and is off the Dirichlet velocity on
        the walls and the inflow. The default ProjectInitialConditions,
        Average, averages the file and imposes the Dirichlet values of the
        velocity and the pressure at the start time. The metrics are the
        errors against the exact solution after five steps.
        ChannelFlow2D_FileInitialCondition_Galerkin projects the same file
        by the global mass matrix solve instead.
    </description>
    <executable>IncNavierStokesSolverRedesign</executable>
    <parameters>ChannelFlow2D_FileInitialCondition.xml</parameters>
    <files>
        <file description="Session File">ChannelFlow2D_FileInitialCondition.xml</file>
        <file description="Initial Condition">ChannelFlow2D_FileInitialCondition_ini.fld</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="1e-5">0.581426</value>
            <value variable="v" tolerance="1e-5">0.196632</value>
            <value variable="p" tolerance="1e-5">2.66526</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="1e-5">0.339267</value>
            <value variable="v" tolerance="1e-5">0.142453</value>
            <value variable="p" tolerance="1e-5">1.9813</value>
        </metric>
    </metrics>
</test>
