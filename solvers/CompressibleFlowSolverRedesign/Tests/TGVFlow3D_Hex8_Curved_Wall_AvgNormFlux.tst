<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>Curved-but-affine mesh of the eight hexahedra: every edge carries collinear curve points, so the elements take the deformed paths. TGV, 8-element hex, adiabatic walls on z, P3 orthogonal basis on Gauss points, 10 steps: average-of-normal-fluxes viscous trace flux.</description>
    <executable>CompressibleFlowSolverRedesign</executable>
    <parameters>TGVFlow3D_Hex8_Curved.xml TGVFlow3D_Hex8_Wall_AvgNormFlux.xml</parameters>
    <files>
        <file description="Mesh">TGVFlow3D_Hex8_Curved.xml</file>
        <file description="Session File">TGVFlow3D_Hex8_Wall_AvgNormFlux.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="rho" tolerance="1e-12">6.21792e-05</value>
            <value variable="rhou" tolerance="1e-12">0.00646911</value>
            <value variable="rhov" tolerance="1e-12">0.00646911</value>
            <value variable="rhow" tolerance="1e-12">0.00174688</value>
            <value variable="E" tolerance="1e-12">0.0160614</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="rho" tolerance="1e-12">8.29703e-06</value>
            <value variable="rhou" tolerance="1e-12">0.0012636</value>
            <value variable="rhov" tolerance="1e-12">0.0012636</value>
            <value variable="rhow" tolerance="1e-12">0.000285326</value>
            <value variable="E" tolerance="1e-12">0.00213775</value>
        </metric>
    </metrics>
</test>
