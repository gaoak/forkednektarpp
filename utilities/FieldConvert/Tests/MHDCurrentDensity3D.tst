<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>Compute quasi-static MHD electric current in 3D</description>
    <executable>FieldConvert</executable>
    <parameters>-f -m fieldfromstring:fieldstr="x+2*y+3*z":fieldname="phi" -m MHDCurrentDensity -e chan3D.xml MHDCurrentDensityConditions.xml chan3D.fld MHDCurrentDensity3D.fld</parameters>
    <files>
        <file description="Mesh file">chan3D.xml</file>
        <file description="MHD conditions">MHDCurrentDensityConditions.xml</file>
        <file description="Field file">chan3D.fld</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="x" tolerance="1e-5">1.63299</value>
            <value variable="y" tolerance="1e-5">1.63299</value>
            <value variable="z" tolerance="1e-5">1.63299</value>
            <value variable="u" tolerance="1e-5">2.06559</value>
            <value variable="v" tolerance="1e-8">0</value>
            <value variable="w" tolerance="1e-8">0</value>
            <value variable="p" tolerance="1e-5">6.53197</value>
            <value variable="phi" tolerance="1e-5">6.1101</value>
            <value variable="Jx" tolerance="1e-10">0</value>
            <value variable="Jy" tolerance="1e-4">24.7871</value>
            <value variable="Jz" tolerance="1e-4">20.6559</value>
        </metric>
    </metrics>
</test>
