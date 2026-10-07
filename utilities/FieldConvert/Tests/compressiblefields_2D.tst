<?xml version="1.0" encoding="utf-8"?>
<test>
    <description> Compute compressible flow fields in 2D (compare with OutputExtraFields)</description>
    <executable>FieldConvert</executable>
    <parameters> -f -e -m removefield:fieldname=u,v,p,T,s,a,Mach,Sensor -m compressiblefields intake0.xml compressiblefields_conditions.xml intake0.fld compressiblefields_2D.fld</parameters>
    <files>
        <file description="Session File">intake0.xml</file>
        <file description="Session File">compressiblefields_conditions.xml</file>
        <file description="Session File">intake0.fld</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="x" tolerance="1e-4">26.7415</value>
            <value variable="y" tolerance="1e-4">22.3235</value>
            <value variable="rho" tolerance="1e-5">8.47976</value>
            <value variable="rhou" tolerance="1e-5">8.15752</value>
            <value variable="rhov" tolerance="1e-6">0.336093</value>
            <value variable="E" tolerance="1e-5">5.83638</value>
            <value variable="u" tolerance="1e-5">7.51321</value>
            <value variable="v" tolerance="1e-6">0.164422</value>
            <value variable="p" tolerance="1e-6">0.779726</value>
            <value variable="T" tolerance="1e-5">7.74765</value>
            <value variable="s" tolerance="1e-8">0.00479406</value>
            <value variable="a" tolerance="1e-5">2.54481</value>
            <value variable="Mach" tolerance="1e-4">22.4279</value>
        </metric>
    </metrics>
</test>
