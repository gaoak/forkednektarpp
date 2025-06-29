<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>cNS, manufactured, C-shape, Prism, variable P, see issue #306</description>
    <executable>CompressibleFlowSolver</executable>
    <parameters>Annulus90deg3D_WeakDG_SIP_MODIFIED_GL_Prism.xml</parameters>
    <processes>4</processes>
    <files>
        <file description="Session File">Annulus90deg3D_WeakDG_SIP_MODIFIED_GL_Prism.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="rho" tolerance="1e-05">6.28146e-04</value>
            <value variable="rhou" tolerance="1e-05">4.24957e-03</value>
            <value variable="rhov" tolerance="1e-05">1.77853e-03</value>
            <value variable="rhow" tolerance="1e-05">6.00656e-12</value>
            <value variable="E" tolerance="1e-05">1.35263e+02</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="rho" tolerance="1e-05">1.17340e+00</value>
            <value variable="rhou" tolerance="1e-05">1.23174e+01</value>
            <value variable="rhov" tolerance="1e-05">6.68577e+00</value>
            <value variable="rhow" tolerance="1e-05">5.31904e-08</value>
            <value variable="E" tolerance="1e-05">2.52680e+05</value>
        </metric>
    </metrics>
</test>