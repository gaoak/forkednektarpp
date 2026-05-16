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
            <value variable="rho" tolerance="1e-12">1.28857e-08</value>
            <value variable="rhou" tolerance="1e-12">0.000984287</value>
            <value variable="rhov" tolerance="1e-12">0.000984287</value>
            <value variable="rhow" tolerance="1e-12">0.00139196</value>
            <value variable="E" tolerance="1e-12">0.00283008</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="rho" tolerance="1e-12">8.33905e-09</value>
            <value variable="rhou" tolerance="1e-12">0.000121752</value>
            <value variable="rhov" tolerance="1e-12">0.000121752</value>
            <value variable="rhow" tolerance="1e-12">0.000241088</value>
            <value variable="E" tolerance="1e-12">0.000444461</value>
        </metric>
    </metrics>
</test>
