<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>Taylor-Green-type initial field on the unit cube meshed with tetrahedra, hexahedra, prisms and pyramids (read from AllElmts_Box.xml), P3 orthogonal basis on Gauss points, adiabatic walls on all six faces, 10 steps: flux-of-averages viscous trace flux, whose gradient gather on the faces this mesh interpolates carries three components per variable.</description>
    <executable>CompressibleFlowSolverRedesign</executable>
    <parameters>-I ViscousTraceFlux=FluxOfAverages AllElmts_Box.xml TGV_AllElmts_Wall_AvgNormFlux.xml</parameters>
    <files>
        <file description="Mesh">AllElmts_Box.xml</file>
        <file description="Session File">TGV_AllElmts_Wall_AvgNormFlux.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="rho" tolerance="1e-12">0.00600194</value>
            <value variable="rhou" tolerance="1e-12">0.0473505</value>
            <value variable="rhov" tolerance="1e-12">0.0383158</value>
            <value variable="rhow" tolerance="1e-12">0.00483418</value>
            <value variable="E" tolerance="1e-12">1.50251</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="rho" tolerance="1e-12">0.0352913</value>
            <value variable="rhou" tolerance="1e-12">0.357717</value>
            <value variable="rhov" tolerance="1e-12">0.227277</value>
            <value variable="rhow" tolerance="1e-12">0.030316</value>
            <value variable="E" tolerance="1e-12">8.86499</value>
        </metric>
    </metrics>
</test>
