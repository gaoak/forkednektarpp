<?xml version="1.0" encoding="utf-8"?>
<!--
    The pressure bounds cover both formulations: semi-implicit runs the
    strong pressure form and linear-implicit the weak one, their defaults.
    Observed on Serial, AVX and Device (semi-implicit / linear-implicit):
    p L2 7.285e-3 / 6.536e-3, Linf 3.347e-3 / 2.802e-3.
-->
<test>
    <description>TGVFlow3D Hex8 10-step redesign reference</description>
    <executable>IncNavierStokesSolverRedesign</executable>
    <parameters>-P NumSteps=10 TGVFlow3D_Hex8.xml</parameters>
    <files>
        <file description="Session File">TGVFlow3D_Hex8.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="1e-06">0.0011981</value>
            <value variable="v" tolerance="1e-06">0.0011981</value>
            <value variable="w" tolerance="1e-06">0.0013911</value>
            <value variable="p" tolerance="5e-04">0.0069107</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="3e-05">0.00038100</value>
            <value variable="v" tolerance="3e-05">0.00038100</value>
            <value variable="w" tolerance="3e-05">0.00023500</value>
            <value variable="p" tolerance="4e-04">0.00307430</value>
        </metric>
        <metric type="TABLE" id="3">
            <file filename="TGVFlow3D_Hex8_energy.eny" tolerance="1e-7">
                <columns>Time KineticEnergy Enstrophy</columns>
                <row>0.0000000000000000e+00 1.2499999931535860e-01 3.7500014550027772e-01</row>
                <row>5.0000000000000001e-04 1.2499624974328795e-01 3.7498890645660754e-01</row>
                <row>1.0000000000000002e-03 1.2499250027698855e-01 3.7497768728886627e-01</row>
            </file>
        </metric>
    </metrics>
</test>
