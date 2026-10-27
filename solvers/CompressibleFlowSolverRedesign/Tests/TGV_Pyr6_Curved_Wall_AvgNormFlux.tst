<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>Curved-but-affine mesh of the six pyramids: every edge carries collinear curve points, so the elements take the deformed paths. Taylor-Green-type initial field on the unit cube split into six pyramids (read from Pyr6_Box_Curved.xml), P3 orthogonal basis on Gauss points, adiabatic walls on all six faces, 10 steps: average-of-normal-fluxes viscous trace flux.</description>
    <executable>CompressibleFlowSolverRedesign</executable>
    <parameters>Pyr6_Box_Curved.xml TGV_Pyr6_Wall_AvgNormFlux.xml</parameters>
    <files>
        <file description="Mesh">Pyr6_Box_Curved.xml</file>
        <file description="Session File">TGV_Pyr6_Wall_AvgNormFlux.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="rho" tolerance="1e-12">0.00525324</value>
            <value variable="rhou" tolerance="1e-12">0.0378364</value>
            <value variable="rhov" tolerance="1e-12">0.0382327</value>
            <value variable="rhow" tolerance="1e-12">0.000805774</value>
            <value variable="E" tolerance="1e-12">1.31505</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="rho" tolerance="1e-12">0.0179108</value>
            <value variable="rhou" tolerance="1e-12">0.185186</value>
            <value variable="rhov" tolerance="1e-12">0.186321</value>
            <value variable="rhow" tolerance="1e-12">0.00457664</value>
            <value variable="E" tolerance="1e-12">4.50004</value>
        </metric>
    </metrics>
</test>
