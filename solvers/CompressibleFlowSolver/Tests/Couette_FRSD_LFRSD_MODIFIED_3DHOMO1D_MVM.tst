<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>NS, Couette flow, mixed bcs, FRSD advection and LFRSD diffusion, MODIFIED</description>
    <executable>CompressibleFlowSolver</executable>
    <parameters>Couette_FRSD_LFRSD_MODIFIED_3DHOMO1D_MVM.xml</parameters>
    <files>
        <file description="Session File">Couette_FRSD_LFRSD_MODIFIED_3DHOMO1D_MVM.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="rho" tolerance="1e-09">0.000553063</value>
            <value variable="rhou" tolerance="1e-04">68.0640</value>
            <value variable="rhov" tolerance="1e-06">0.202862</value>
            <value variable="rhow" tolerance="1e-11">9.83517e-06</value>
            <value variable="E" tolerance="1e-01">24776.3</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="rho" tolerance="1e-09">0.00162562</value>
            <value variable="rhou" tolerance="1e-04">83.3449</value>
            <value variable="rhov" tolerance="1e-06">0.588196</value>
            <value variable="rhow" tolerance="1e-11">4.14619e-05</value>
            <value variable="E" tolerance="1e-01">18878.4</value>
        </metric>
    </metrics>
</test>


