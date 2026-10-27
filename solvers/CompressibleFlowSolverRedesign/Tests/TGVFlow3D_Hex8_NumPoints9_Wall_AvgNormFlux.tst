<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>Nine Gauss points per direction. TGV, 8-element hex, adiabatic walls on z, P3 orthogonal basis on Gauss points, 10 steps: average-of-normal-fluxes viscous trace flux.</description>
    <executable>CompressibleFlowSolverRedesign</executable>
    <parameters>TGVFlow3D_Hex8.xml TGVFlow3D_Hex8_NumPoints9_Wall_AvgNormFlux.xml</parameters>
    <files>
        <file description="Mesh">TGVFlow3D_Hex8.xml</file>
        <file description="Session File">TGVFlow3D_Hex8_NumPoints9_Wall_AvgNormFlux.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="rho" tolerance="1e-12">2.61839e-09</value>
            <value variable="rhou" tolerance="1e-12">0.000950342</value>
            <value variable="rhov" tolerance="1e-12">0.000950342</value>
            <value variable="rhow" tolerance="1e-12">0.00133192</value>
            <value variable="E" tolerance="1e-12">0.00257564</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="rho" tolerance="1e-12">1.01564e-09</value>
            <value variable="rhou" tolerance="1e-12">0.000203018</value>
            <value variable="rhov" tolerance="1e-12">0.000203018</value>
            <value variable="rhow" tolerance="1e-12">0.000338342</value>
            <value variable="E" tolerance="1e-12">0.000738786</value>
        </metric>
    </metrics>
</test>
