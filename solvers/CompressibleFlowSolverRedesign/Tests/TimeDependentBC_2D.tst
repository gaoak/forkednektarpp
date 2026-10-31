<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>Euler equations: a plane shear wave advected by a uniform flow, with TimeDependent Dirichlet conditions set to the exact solution on every side. Ported from the legacy TimeDependentAbsorption test without its absorption forcing, which the exact solution does not need, and with explicit time stepping.</description>
    <executable>CompressibleFlowSolverRedesign</executable>
    <parameters>TimeDependentBC_2D.xml</parameters>
    <files>
        <file description="Session File">TimeDependentBC_2D.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="rho" tolerance="1e-8">0.00435988</value>
            <value variable="rhou" tolerance="1e-7">0.0148819</value>
            <value variable="rhov" tolerance="1e-6">0.0147496</value>
            <value variable="E" tolerance="1e-7">0.0481127</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="rho" tolerance="1e-8">0.000430549</value>
            <value variable="rhou" tolerance="1e-8">0.00135732</value>
            <value variable="rhov" tolerance="1e-8">0.00135767</value>
            <value variable="E" tolerance="1e-7">0.00495821</value>
        </metric>
    </metrics>
</test>
