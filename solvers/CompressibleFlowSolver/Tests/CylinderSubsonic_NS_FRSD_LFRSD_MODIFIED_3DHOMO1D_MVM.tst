<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>NS, Subsonic Cylinder, mixed bcs, FRSD advection and LFRSD diffusion, MODIFIED</description>
    <executable>CompressibleFlowSolver</executable>
    <parameters>CylinderSubsonic_NS_FRSD_LFRSD_MODIFIED_3DHOMO1D_MVM.xml</parameters>
    <files>
        <file description="Session File">CylinderSubsonic_NS_FRSD_LFRSD_MODIFIED_3DHOMO1D_MVM.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="rho" tolerance="1e-05">5.50775</value>
            <value variable="rhou" tolerance="1e-02">2016.35</value>
            <value variable="rhov" tolerance="1e-04">24.5221</value>
            <value variable="rhow" tolerance="1e-04">24.4789</value>
            <value variable="E" tolerance="1e+01">6.27023e+06</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="rho" tolerance="1e-05">0.281925</value>
            <value variable="rhou" tolerance="1e-04">85.4194</value>
            <value variable="rhov" tolerance="1e-04">12.2322</value>
            <value variable="rhow" tolerance="1e-05">1.00004</value>
            <value variable="E" tolerance="1e+00">272860</value>
        </metric>
    </metrics>
</test>


