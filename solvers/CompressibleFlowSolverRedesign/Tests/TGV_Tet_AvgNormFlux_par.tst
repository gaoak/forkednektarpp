<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>TGV, 3439-element tet P3 orthogonal basis on Gauss points, 10 steps: average-of-normal-fluxes viscous trace flux. Two ranks, the same reference as the serial run, so that the partition path is covered.</description>
    <executable>CompressibleFlowSolverRedesign</executable>
    <parameters>TGV_8x8x8_Tet_ortho_P0.xml TGV_Tet_AvgNormFlux.xml</parameters>
    <processes>2</processes>
    <files>
        <file description="Mesh">TGV_8x8x8_Tet_ortho_P0.xml</file>
        <file description="Session File">TGV_Tet_AvgNormFlux.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="rho" tolerance="1e-12">6.13486e-05</value>
            <value variable="rhou" tolerance="1e-12">0.00107807</value>
            <value variable="rhov" tolerance="1e-12">0.00107186</value>
            <value variable="rhow" tolerance="1e-12">0.00141577</value>
            <value variable="E" tolerance="1e-12">0.0156324</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="rho" tolerance="1e-12">0.00016387</value>
            <value variable="rhou" tolerance="1e-12">0.00100139</value>
            <value variable="rhov" tolerance="1e-12">0.00165084</value>
            <value variable="rhow" tolerance="1e-12">0.000867117</value>
            <value variable="E" tolerance="1e-12">0.0400692</value>
        </metric>
    </metrics>
</test>
