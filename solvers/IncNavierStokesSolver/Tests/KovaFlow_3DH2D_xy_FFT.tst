<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>Kovasznay Flow 3D homogeneous 2D, flow in xy, FFT</description>
    <executable>IncNavierStokesSolver</executable>
    <parameters>KovaFlow_3DH2D_xy_FFT.xml</parameters>
    <files>
        <file description="Session File">KovaFlow_3DH2D_xy_FFT.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="1e-12">1.0452e-07</value>
            <value variable="v" tolerance="1e-13">1.53988e-08</value>
            <value variable="w" tolerance="1e-12">0</value>
	    <value variable="p" tolerance="1e-11">3.88691e-06</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="1e-12">1.79047e-07</value>
            <value variable="v" tolerance="1e-13">3.05497e-08</value>
            <value variable="w" tolerance="1e-12">0</value>
	    <value variable="p" tolerance="1e-11">4.77709e-06</value>
        </metric>
    </metrics>
</test>
