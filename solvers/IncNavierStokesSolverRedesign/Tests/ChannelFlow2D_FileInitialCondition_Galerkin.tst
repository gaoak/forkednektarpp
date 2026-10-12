<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>
        ChannelFlow2D_FileInitialCondition with ProjectInitialConditions
        = Galerkin: the file is transformed to the quadrature points and
        projected by the global mass matrix solve, with the Dirichlet values
        at the start time. The projection spreads the wall mismatch of the
        file into the interior, so the errors after five steps are larger
        than with Average: 0.73 rather than 0.58 in the L2 norm of u.
    </description>
    <executable>IncNavierStokesSolverRedesign</executable>
    <parameters>-I ProjectInitialConditions=Galerkin ChannelFlow2D_FileInitialCondition.xml</parameters>
    <files>
        <file description="Session File">ChannelFlow2D_FileInitialCondition.xml</file>
        <file description="Initial Condition">ChannelFlow2D_FileInitialCondition_ini.fld</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="1e-5">0.726878</value>
            <value variable="v" tolerance="1e-5">0.277327</value>
            <value variable="p" tolerance="1e-5">3.19271</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="1e-5">0.458053</value>
            <value variable="v" tolerance="1e-5">0.206327</value>
            <value variable="p" tolerance="1e-5">2.52503</value>
        </metric>
    </metrics>
</test>
