<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>Fully 3D Channel Flow. Synthetic turbulence generation is introduced in the flow field using forcing tag. </description>
    <executable>IncNavierStokesSolver</executable>
    <parameters>ChanFlow3D_infTurb.xml</parameters>
    <files>
        <file description="Session File">ChanFlow3D_infTurb.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="1e-5">3.64539</value>
            <value variable="v" tolerance="1e-7">0.0268234</value>
            <value variable="w" tolerance="1e-6">0.030717</value>
            <value variable="p" tolerance="1e-7">0.0224162</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="1e-5">1.40919</value>
            <value variable="v" tolerance="1e-6">0.145714</value>
            <value variable="w" tolerance="1e-6">0.119218</value>
            <value variable="p" tolerance="1e-7">0.0906175</value>
        </metric>
    </metrics>
</test>


