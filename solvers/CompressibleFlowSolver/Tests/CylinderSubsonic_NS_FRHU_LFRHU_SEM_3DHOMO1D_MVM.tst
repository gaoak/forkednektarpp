<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>NS, Subsonic Cylinder, mixed bcs, FRHU advection and LFRHU diffusion, SEM</description>
    <executable>CompressibleFlowSolver</executable>
    <parameters>CylinderSubsonic_NS_FRHU_LFRHU_SEM_3DHOMO1D_MVM.xml</parameters>
    <files>
        <file description="Session File">CylinderSubsonic_NS_FRHU_LFRHU_SEM_3DHOMO1D_MVM.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="rho" tolerance="1e-05">5.50776</value>
            <value variable="rhou" tolerance="1e-02">2016.35</value>
            <value variable="rhov" tolerance="1e-04">24.5243</value>
            <value variable="rhow" tolerance="1e-04">24.4789</value>
            <value variable="E" tolerance="1e+01">6.27023e+06</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="rho" tolerance="1e-05"> 0.275036</value>
            <value variable="rhou" tolerance="1e-04">82.5652</value>
            <value variable="rhov" tolerance="1e-04">10.8489</value>
            <value variable="rhow" tolerance="1e-05">1.00004</value>
            <value variable="E" tolerance="1e+00">270824</value>
        </metric>
    </metrics>
</test>


