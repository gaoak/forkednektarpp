<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>Taylor-Green-type field on the mixed prism and hexahedral box at variable order, prisms P3 and hexahedra P2 orthogonal on Gauss points, adiabatic walls on all faces, 10 steps: average-of-normal-fluxes viscous trace flux. The prism-hexahedron interfaces interpolate between the two orders, so the interior trace flux must take them on the packed path and not in place; values from the packed path.</description>
    <executable>CompressibleFlowSolverRedesign</executable>
    <parameters>PrismHex_Box.xml TGV_PrismHex_VarP_Wall_AvgNormFlux.xml</parameters>
    <files>
        <file description="Mesh">PrismHex_Box.xml</file>
        <file description="Session File">TGV_PrismHex_VarP_Wall_AvgNormFlux.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="rho" tolerance="1e-12">0.0124383</value>
            <value variable="rhou" tolerance="1e-12">0.0587399</value>
            <value variable="rhov" tolerance="1e-12">0.111393</value>
            <value variable="rhow" tolerance="1e-12">0.000689985</value>
            <value variable="E" tolerance="1e-12">3.11082</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="rho" tolerance="1e-12">0.0627646</value>
            <value variable="rhou" tolerance="1e-12">0.210174</value>
            <value variable="rhov" tolerance="1e-12">0.637628</value>
            <value variable="rhow" tolerance="1e-12">0.00316422</value>
            <value variable="E" tolerance="1e-12">15.8042</value>
        </metric>
    </metrics>
</test>
