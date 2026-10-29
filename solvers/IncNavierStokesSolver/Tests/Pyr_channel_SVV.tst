<?xml version="1.0" encoding="utf-8" ?>
<test>
    <description>3D channel flow, Pyramidic elements, using SVV</description>
    <executable>IncNavierStokesSolver</executable>
    <parameters>Pyr_channel_SVV.xml</parameters>
    <files>
        <file description="Session File">Pyr_channel_SVV.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="1e-9">9.09293e-16</value>
            <value variable="v" tolerance="1e-9">8.77192e-16</value>
            <value variable="w" tolerance="1e-9">3.06821e-15</value>
            <value variable="p" tolerance="1e-8">4.50247e-14</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="5e-8">4.12488e-15</value>
            <value variable="v" tolerance="5e-8">4.76221e-15</value>
            <value variable="w" tolerance="1e-8">1.7264e-14</value>
            <value variable="p" tolerance="1e-6">5.70211e-13</value>
        </metric>
    </metrics>
</test>
