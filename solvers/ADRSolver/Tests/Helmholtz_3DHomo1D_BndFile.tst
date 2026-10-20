<?xml version="1.0" encoding="utf-8" ?>
<test>
    <description>3DH1D Helmholtz with Dirichlet, Neumann and Robin conditions read from file and multiplied by a z-dependent expression</description>
    <executable>ADRSolver</executable>
    <parameters>Helmholtz_3DHomo1D_BndFile.xml</parameters>
    <files>
        <file description="Session File">Helmholtz_3DHomo1D_BndFile.xml</file>
        <file description="Session File">Helmholtz_3DHomo1D_BndFile.bc</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="1e-12">5.83424e-15</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="1e-12">2.84217e-14</value>
        </metric>
    </metrics>
</test>
