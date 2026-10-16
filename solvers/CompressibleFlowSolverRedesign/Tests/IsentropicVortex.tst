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
            <value variable="rho" tolerance="1e-12">0.0341217</value>
            <value variable="rhou" tolerance="1e-12">0.0581278</value>
            <value variable="rhov" tolerance="1e-12">0.0516215</value>
            <value variable="E" tolerance="1e-12">0.191099</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="rho" tolerance="1e-12">0.0334495</value>
            <value variable="rhou" tolerance="1e-12">0.0621705</value>
            <value variable="rhov" tolerance="1e-12">0.0417275</value>
            <value variable="E" tolerance="1e-12">0.22457</value>
        </metric>
    </metrics>
</test>

