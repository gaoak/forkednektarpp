<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>Three-dimensional homogeneous one-dimensional MHD duct flow with flow-rate control</description>
    <executable>IncNavierStokesSolver</executable>
    <parameters>MHDDuctFlow.xml</parameters>
    <files>
        <file description="Session File">MHDDuctFlow.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="1e-12">0</value>
            <value variable="v" tolerance="1e-12">0</value>
            <value variable="w" tolerance="1e-4">1.1841</value>
            <value variable="phi" tolerance="1e-4">2.20105</value>
            <value variable="p" tolerance="1e-12">0</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="1e-12">0</value>
            <value variable="v" tolerance="1e-12">0</value>
            <value variable="w" tolerance="1e-4">2.03048</value>
            <value variable="phi" tolerance="1e-4">3.93939</value>
            <value variable="p" tolerance="1e-12">0</value>
        </metric>
    </metrics>
</test>
