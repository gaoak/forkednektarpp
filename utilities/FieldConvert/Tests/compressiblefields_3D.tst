<?xml version="1.0" encoding="utf-8"?>
<test>
    <description> Compute compressible flow fields in 3D</description>
    <executable>FieldConvert</executable>
    <parameters> -f -e -m compressiblefields wss_3D_periodic.xml wss_3D_periodic.fld compressiblefields_3D.fld</parameters>
    <files>
        <file description="Session File">wss_3D_periodic.xml</file>
        <file description="Session File">wss_3D_periodic.fld</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="x" tolerance="1e-6">0.204124</value>
            <value variable="y" tolerance="1e-6">0.188193</value>
            <value variable="z" tolerance="1e-7">0.0204124</value>
            <value variable="rho" tolerance="1e-8">0.00782982</value>
            <value variable="rhou" tolerance="1e-5">2.48245</value>
            <value variable="rhov" tolerance="1e-12">0</value>
            <value variable="rhow" tolerance="1e-12">0</value>
            <value variable="E" tolerance="1e-2">1962.73</value>
            <value variable="u" tolerance="1e-3">100.26</value>
            <value variable="v" tolerance="1e-12">0</value>
            <value variable="w" tolerance="1e-12">0</value>
            <value variable="p" tolerance="1e-3">640.257</value>
            <value variable="T" tolerance="1e-4">90.081</value>
            <value variable="s" tolerance="1e-2">1609.54</value>
            <value variable="a" tolerance="1e-3">105.939</value>
            <value variable="Mach" tolerance="1e-6">0.333986</value>
        </metric>
    </metrics>
</test>
