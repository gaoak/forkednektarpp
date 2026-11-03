<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>TGV, 3439-element tet, adiabatic walls on z, P3 orthogonal basis on Gauss points, 10 steps: average-of-normal-fluxes viscous trace flux.</description>
    <executable>CompressibleFlowSolverRedesign</executable>
    <parameters>TGV_8x8x8_Tet_ortho_P0.xml TGV_Tet_Wall_AvgNormFlux.xml</parameters>
    <files>
        <file description="Mesh">TGV_8x8x8_Tet_ortho_P0.xml</file>
        <file description="Session File">TGV_Tet_Wall_AvgNormFlux.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="rho" tolerance="1e-12">6.12406e-05</value>
            <value variable="rhou" tolerance="1e-12">0.00178201</value>
            <value variable="rhov" tolerance="1e-12">0.00175508</value>
            <value variable="rhow" tolerance="1e-12">0.0014157</value>
            <value variable="E" tolerance="1e-12">0.0156058</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="rho" tolerance="1e-12">0.00016387</value>
            <value variable="rhou" tolerance="1e-12">0.0032169</value>
            <value variable="rhov" tolerance="1e-12">0.00245302</value>
            <value variable="rhow" tolerance="1e-12">0.000867118</value>
            <value variable="E" tolerance="1e-12">0.0400692</value>
        </metric>
    </metrics>
</test>
