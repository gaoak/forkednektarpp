<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>TGVFlow3D Hex8 10 step redesign reference</description>
    <executable>CompressibleFlowSolverRedesign</executable>
    <parameters>TGVFlow3D_Hex8.xml</parameters>
    <files>
        <file description="Session File">TGVFlow3D_Hex8.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="rho" tolerance="1e-12">3.14864e-07</value>
            <value variable="rhou" tolerance="1e-12">0.000985134</value>
            <value variable="rhov" tolerance="1e-12">0.000985134</value>
            <value variable="rhow" tolerance="1e-12">0.00139239</value>
            <value variable="E" tolerance="1e-12">0.00286098</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="rho" tolerance="1e-12">5.59255e-08</value>
            <value variable="rhou" tolerance="1e-12">0.000119842</value>
            <value variable="rhov" tolerance="1e-12">0.000119842</value>
            <value variable="rhow" tolerance="1e-12">0.000239681</value>
            <value variable="E" tolerance="1e-12">0.000623629</value>
        </metric>
    </metrics>
</test>
