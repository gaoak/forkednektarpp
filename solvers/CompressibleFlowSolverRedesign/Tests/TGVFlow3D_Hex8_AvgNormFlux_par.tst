<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>TGV, 8-element hex P3 orthogonal basis on Gauss points, 10 steps: average-of-normal-fluxes viscous trace flux. Two ranks, the same reference as the serial run, so that the partition path is covered.</description>
    <executable>CompressibleFlowSolverRedesign</executable>
    <parameters>TGVFlow3D_Hex8.xml TGVFlow3D_Hex8_AvgNormFlux.xml</parameters>
    <processes>2</processes>
    <files>
        <file description="Mesh">TGVFlow3D_Hex8.xml</file>
        <file description="Session File">TGVFlow3D_Hex8_AvgNormFlux.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="rho" tolerance="1e-12">6.21795e-05</value>
            <value variable="rhou" tolerance="1e-12">0.00646674</value>
            <value variable="rhov" tolerance="1e-12">0.00646674</value>
            <value variable="rhow" tolerance="1e-12">0.00174687</value>
            <value variable="E" tolerance="1e-12">0.0160614</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="rho" tolerance="1e-12">8.29704e-06</value>
            <value variable="rhou" tolerance="1e-12">0.00126415</value>
            <value variable="rhov" tolerance="1e-12">0.00126415</value>
            <value variable="rhow" tolerance="1e-12">0.000285326</value>
            <value variable="E" tolerance="1e-12">0.00213776</value>
        </metric>
    </metrics>
</test>
