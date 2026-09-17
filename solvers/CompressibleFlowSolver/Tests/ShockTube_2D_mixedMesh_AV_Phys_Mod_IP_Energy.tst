<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>N-S 2D shocktube with mixed mesh, physical AV, modal sensor reported as the energy ratio, interior AV on boundary traces, and interior penalty</description>
    <executable>CompressibleFlowSolver</executable>
    <parameters>ShockTube_2D_mixedMesh.xml ShockTube_2D_mixedMesh_AV_Phys_Mod_IP_Energy.xml</parameters>
    <files>
        <file description="Mesh File">ShockTube_2D_mixedMesh.xml</file>
        <file description="Session File">ShockTube_2D_mixedMesh_AV_Phys_Mod_IP_Energy.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="rho" tolerance="1e-9">2.36628e-05</value>
            <value variable="rhou" tolerance="1e-5">1.13616</value>
            <value variable="rhov" tolerance="1e-6">0.0919627</value>
            <value variable="E" tolerance="1e-1">737.232</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="rho" tolerance="1e-7">0.00334979</value>
            <value variable="rhou" tolerance="1e-4">73.7968</value>
            <value variable="rhov" tolerance="1e-4">7.11973</value>
            <value variable="E" tolerance="1e-1">20820.7</value>
        </metric>
    </metrics>
</test>
