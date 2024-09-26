<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>NS, Couette flow, mixed bcs, FRDG advection and LFRDG diffusion, GLL_LAGRANGE</description>
    <executable>CompressibleFlowSolver</executable>
    <parameters>Couette_FRDG_LFRDG_GLL_LAGRANGE_3DHOMO1D_MVM.xml</parameters>
    <files>
        <file description="Session File">Couette_FRDG_LFRDG_GLL_LAGRANGE_3DHOMO1D_MVM.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="rho" tolerance="1e-09">0.000683283</value>
            <value variable="rhou" tolerance="1e-04">68.0612</value>
            <value variable="rhov" tolerance="1e-06">0.250835</value>
            <value variable="rhow" tolerance="1e-10">1.11168e-05</value>
            <value variable="E" tolerance="1e-01">24775.5</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="rho" tolerance="1e-09">0.00207212</value>
            <value variable="rhou" tolerance="1e-04">83.3320</value>
            <value variable="rhov" tolerance="1e-06">0.752415</value>
            <value variable="rhow" tolerance="1e-10">5.78284e-05</value>
            <value variable="E" tolerance="1e-01">18726.3</value>
        </metric>
    </metrics>
</test>


