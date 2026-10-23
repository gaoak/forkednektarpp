<?xml version="1.0" encoding="utf-8"?>
<!--
    The bounds cover both formulations: semi-implicit runs the strong
    pressure form and linear-implicit the weak one, their defaults. Observed
    on Serial, AVX and Device (semi-implicit / linear-implicit): L2 u
    1.016e-9 / 1.254e-9, v 1.1765e-8 / 1.1713e-8, p 8.5225e-9 / 8.8303e-9;
    Linf u 2.654e-9 / 2.511e-9, v 1.63561e-8 both, p 2.0458e-8 / 1.6565e-8.
-->
<test>
    <description>KovasznayFlow2D</description>
    <executable>IncNavierStokesSolverRedesign</executable>
    <parameters>-I ProjectInitialConditions=Galerkin KovasznayFlow2D.xml</parameters>
    <files>
        <file description="Session File">KovasznayFlow2D.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="2e-10">1.15000e-09</value>
            <value variable="v" tolerance="5e-11">1.17390e-08</value>
            <value variable="p" tolerance="2e-10">8.67640e-09</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="2e-10">2.58240e-09</value>
            <value variable="v" tolerance="1e-10">1.63561e-08</value>
            <value variable="p" tolerance="2.5e-09">1.85114e-08</value>
        </metric>
    </metrics>
</test>
