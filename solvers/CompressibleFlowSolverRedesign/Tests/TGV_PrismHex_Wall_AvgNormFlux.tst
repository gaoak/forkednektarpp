<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>Taylor-Green-type field on the mixed prism and hexahedral box, adiabatic walls on all faces, P3 orthogonal basis on Gauss points, 10 steps: average-of-normal-fluxes viscous trace flux.</description>
    <executable>CompressibleFlowSolverRedesign</executable>
    <parameters>PrismHex_Box.xml TGV_PrismHex_Wall_AvgNormFlux.xml</parameters>
    <files>
        <file description="Mesh">PrismHex_Box.xml</file>
        <file description="Session File">TGV_PrismHex_Wall_AvgNormFlux.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="rho" tolerance="1e-12">0.0131821</value>
            <value variable="rhou" tolerance="1e-12">0.0735571</value>
            <value variable="rhov" tolerance="1e-12">0.111397</value>
            <value variable="rhow" tolerance="1e-12">0.000701655</value>
            <value variable="E" tolerance="1e-12">3.29707</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="rho" tolerance="1e-12">0.0627966</value>
            <value variable="rhou" tolerance="1e-12">0.210897</value>
            <value variable="rhov" tolerance="1e-12">0.63753</value>
            <value variable="rhow" tolerance="1e-12">0.00316626</value>
            <value variable="E" tolerance="1e-12">15.8125</value>
        </metric>
    </metrics>
</test>
