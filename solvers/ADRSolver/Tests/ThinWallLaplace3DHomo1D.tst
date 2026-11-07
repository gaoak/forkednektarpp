<?xml version='1.0' encoding='utf-8'?>
<test>
    <description>Thin conducting wall surface stiffness in Homogeneous1D: exact 3D harmonic solution cosh(beta*y)*cos(beta*z), nonzero Fourier mode, DirectStaticCond</description>
    <executable>ADRSolver</executable>
    <parameters>ThinWallLaplace3DHomo1D.xml</parameters>
    <files>
        <file description="Session File">ThinWallLaplace3DHomo1D.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="0">
            <value variable="phi" tolerance="1e-12">0</value>
        </metric>
        <metric type="Linf" id="1">
            <value variable="phi" tolerance="1e-12">0</value>
        </metric>
    </metrics>
</test>
