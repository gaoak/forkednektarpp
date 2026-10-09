<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>Two-dimensional Hartmann flow</description>
    <executable>IncNavierStokesSolver</executable>
    <parameters>HartmannFlow.xml</parameters>
    <files>
        <file description="Session File">HartmannFlow.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="1e-4">0.000215744</value>
            <value variable="v" tolerance="1e-12">1.88826e-16</value>
            <value variable="phi" tolerance="1e-12">0</value>
            <value variable="p" tolerance="1e-12">1.22306e-14</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="1e-4">0.000345854</value>
            <value variable="v" tolerance="1e-12">5.572e-16</value>
            <value variable="phi" tolerance="1e-12">0</value>
            <value variable="p" tolerance="1e-12">2.50953e-14</value>
        </metric>
    </metrics>
</test>
