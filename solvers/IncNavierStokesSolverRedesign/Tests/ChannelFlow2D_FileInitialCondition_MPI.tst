<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>
        ChannelFlow2D_FileInitialCondition on two processes: the
        averaging counts the copies of a degree of freedom on the other
        partition, so the start state, and the metrics, are those of the
        serial run.
    </description>
    <executable>IncNavierStokesSolverRedesign</executable>
    <parameters>ChannelFlow2D_FileInitialCondition.xml</parameters>
    <processes>2</processes>
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
