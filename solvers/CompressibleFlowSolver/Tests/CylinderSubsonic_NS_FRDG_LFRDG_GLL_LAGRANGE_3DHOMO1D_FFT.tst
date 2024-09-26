<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>NS, Subsonic Cylinder, mixed bcs, FRDG advection and LFRDG diffusion, GLL_LAGRANGE</description>
    <executable>CompressibleFlowSolver</executable>
    <parameters>CylinderSubsonic_NS_FRDG_LFRDG_GLL_LAGRANGE_3DHOMO1D_FFT.xml</parameters>
    <files>
        <file description="Session File">CylinderSubsonic_NS_FRDG_LFRDG_GLL_LAGRANGE_3DHOMO1D_FFT.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="rho" tolerance="1e-05">5.50775</value>
            <value variable="rhou" tolerance="1e-02">2016.35</value>
            <value variable="rhov" tolerance="1e-04">24.5455</value>
            <value variable="rhow" tolerance="1e-04">24.4789</value>
            <value variable="E" tolerance="1e+01">6.27023e+06</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="rho" tolerance="1e-06">0.297688</value>
            <value variable="rhou" tolerance="1e-04">86.8671</value>
            <value variable="rhov" tolerance="1e-04"> 15.3684</value>
            <value variable="rhow" tolerance="1e-05">1.00004</value>
            <value variable="E" tolerance="1e+00">277513</value>
        </metric>
    </metrics>
</test>


