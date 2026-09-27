<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>LowEnergyBlock on 27 Hex elements: disjoint edge and face universal IDs</description>
    <executable>ADRSolver</executable>
    <parameters>-v --no-exp-opt Helmholtz3D_LowEnergyBlock_ReorderedHex.xml</parameters>
    <processes>8</processes>
    <files>
        <file description="Session File">Helmholtz3D_LowEnergyBlock_ReorderedHex.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="1e-8">0.00777569</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="1e-7">0.0153467</value>
        </metric>
    </metrics>
</test>
