<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>Euler Isentropic Vortex P=4</description>
    <executable>CompressibleFlowSolverRedesign</executable>
    <parameters>IsentropicVortex.xml</parameters>
    <files>
        <file description="Session File">IsentropicVortex.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="rho" tolerance="1e-12">0.0162887</value>
            <value variable="rhou" tolerance="1e-12">0.0322702</value>
            <value variable="rhov" tolerance="1e-12">0.0456278</value>
            <value variable="E" tolerance="1e-12">0.101696</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="rho" tolerance="1e-12">0.023126</value>
            <value variable="rhou" tolerance="1e-12">0.0435401</value>
            <value variable="rhov" tolerance="1e-12">0.0395554</value>
            <value variable="E" tolerance="1e-12">0.156064</value>
        </metric>
    </metrics>
</test>

