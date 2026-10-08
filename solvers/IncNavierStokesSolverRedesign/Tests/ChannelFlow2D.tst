<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>ChannelFlow2D</description>
    <executable>IncNavierStokesSolverRedesign</executable>
    <parameters>ChannelFlow2D.xml</parameters>
    <files>
        <file description="Session File">ChannelFlow2D.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="1e-10">0</value>
            <value variable="v" tolerance="1e-10">0</value>
            <value variable="p" tolerance="1e-10">0</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="1e-10">0</value>
            <value variable="v" tolerance="1e-10">0</value>
            <value variable="p" tolerance="1e-10">0</value>
        </metric>
    </metrics>
</test>
