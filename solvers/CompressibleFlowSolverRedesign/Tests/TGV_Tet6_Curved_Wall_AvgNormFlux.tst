<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>Curved-but-affine mesh of six tetrahedra: every edge carries collinear curve points, so the elements take the deformed paths. Taylor-Green-type initial field on the box split into six tetrahedra (read from Tet6_Box_Curved.xml), P3 orthogonal basis on Gauss points, adiabatic walls on all six faces, 10 steps: average-of-normal-fluxes viscous trace flux.</description>
    <executable>CompressibleFlowSolverRedesign</executable>
    <parameters>Tet6_Box_Curved.xml TGV_Tet6_Curved_Wall_AvgNormFlux.xml</parameters>
    <files>
        <file description="Mesh">Tet6_Box_Curved.xml</file>
        <file description="Session File">TGV_Tet6_Curved_Wall_AvgNormFlux.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="rho" tolerance="1e-12">0.00631235</value>
            <value variable="rhou" tolerance="1e-12">0.0456343</value>
            <value variable="rhov" tolerance="1e-12">0.0465052</value>
            <value variable="rhow" tolerance="1e-12">0.000573611</value>
            <value variable="E" tolerance="1e-12">1.58227</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="rho" tolerance="1e-12">0.0117575</value>
            <value variable="rhou" tolerance="1e-12">0.0957934</value>
            <value variable="rhov" tolerance="1e-12">0.0853287</value>
            <value variable="rhow" tolerance="1e-12">0.00189764</value>
            <value variable="E" tolerance="1e-12">2.94414</value>
        </metric>
    </metrics>
</test>
