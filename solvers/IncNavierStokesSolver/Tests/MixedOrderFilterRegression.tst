<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>Checkpoint and AeroForces filters with a lower-order pressure on curved continuous tetrahedra</description>
    <segment type="sequential">
        <executable>IncNavierStokesSolver</executable>
        <parameters>MixedOrderFilterRegression.xml</parameters>
        <processes>1</processes>
    </segment>
    <segment type="sequential">
        <executable>../../utilities/FieldConvert/FieldConvert</executable>
        <parameters>-m printfldnorms MixedOrderFilterRegression.xml MixedOrderFilterRegression_2.chk stdout</parameters>
        <processes>1</processes>
    </segment>
    <files>
        <file description="Session File">MixedOrderFilterRegression.xml</file>
    </files>
    <metrics>
        <!-- Norms of the final checkpoint -->
        <metric type="L2" id="1">
            <value variable="u" tolerance="1e-9">2.50559</value>
            <value variable="v" tolerance="1e-9">0.169392</value>
            <value variable="w" tolerance="1e-9">0.00402516</value>
            <value variable="p" tolerance="1e-8">1.18439</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="1e-9">2.13629</value>
            <value variable="v" tolerance="1e-9">0.179532</value>
            <value variable="w" tolerance="1e-9">0.00479478</value>
            <value variable="p" tolerance="1e-8">1.13674</value>
        </metric>
        <metric type="FileExists" id="3">
            <file pattern=".*/MixedOrderFilterRegression_.*\.chk">3</file>
            <file pattern=".*/MixedOrderFilterRegression\.fce">1</file>
        </metric>
    </metrics>
</test>
