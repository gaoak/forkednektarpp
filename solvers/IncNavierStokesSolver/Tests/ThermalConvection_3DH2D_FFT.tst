<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>Thermal convection with pure-Neumann pressure, 3D homogeneous 2D using FFTW</description>
    <executable>IncNavierStokesSolver</executable>
    <parameters>ThermalConvection_3DH2D_FFT.xml</parameters>
    <files>
        <file description="Session File">ThermalConvection_3DH2D_FFT.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="1e-5">0.00155298</value>
            <value variable="v" tolerance="1e-5">0.00112434</value>
            <value variable="w" tolerance="1e-5">0.00112434</value>
            <value variable="theta" tolerance="1e-5">1.1547</value>
            <value variable="p" tolerance="1e-3">182.574</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="1e-5">0.0022522</value>
            <value variable="v" tolerance="1e-5">0.00123552</value>
            <value variable="w" tolerance="1e-5">0.00123552</value>
            <value variable="theta" tolerance="1e-5">1</value>
            <value variable="p" tolerance="1e-3">125.07</value>
        </metric>
    </metrics>
</test>
