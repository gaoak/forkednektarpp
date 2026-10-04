<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>Euler Isentropic Vortex P=3, Average Riemann solver</description>
    <executable>CompressibleFlowSolver</executable>
    <parameters>IsentropicVortex16_P3_Average.xml</parameters>
    <files>
        <file description="Session File">IsentropicVortex16_P3_Average.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="rho" tolerance="1e-12">0.00192959</value>
            <value variable="rhou" tolerance="1e-12">0.00356317</value>
            <value variable="rhov" tolerance="1e-12">0.00544082</value>
            <value variable="E" tolerance="1e-12">0.0123452</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="rho" tolerance="1e-12">0.0042399</value>
            <value variable="rhou" tolerance="1e-12">0.0071086</value>
            <value variable="rhov" tolerance="1e-12">0.0132303</value>
            <value variable="E" tolerance="1e-12">0.0326143</value>
        </metric>
    </metrics>
</test>
