<?xml version="1.0" encoding="utf-8"?>
<!--
    Kovasznay flow with the higher-order pressure boundary condition on every
    boundary, in the weak pressure form. The pressure system is singular:
    the volume forcing is orthogonal to the constant null space, so a
    constant forcing is added to cancel the surface integral of the Neumann
    boundary flux, and the pressure norms are reported with the mean error
    removed. The bounds cover the observed Serial and Device error with a
    margin.
-->
<test>
    <description>KovasznayFlowHOPBC2D_Singular_WeakPressure: exact flow with a singular pressure system in the weak pressure form</description>
    <executable>IncNavierStokesSolverRedesign</executable>
    <parameters>-I ProjectInitialConditions=Galerkin -I PressureFormulation=Weak KovasznayFlowHOPBC2D_Singular.xml</parameters>
    <files>
        <file description="Session File">KovasznayFlowHOPBC2D_Singular.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="5e-8">0</value>
            <value variable="v" tolerance="5e-8">0</value>
            <value variable="p" tolerance="5e-8">0</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="5e-7">0</value>
            <value variable="v" tolerance="5e-7">0</value>
            <value variable="p" tolerance="5e-7">0</value>
        </metric>
    </metrics>
</test>
