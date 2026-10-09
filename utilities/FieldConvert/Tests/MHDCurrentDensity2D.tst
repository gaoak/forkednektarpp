<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>Compute quasi-static MHD electric current in 2D</description>
    <executable>FieldConvert</executable>
    <parameters>-f -m fieldfromstring:fieldstr="x+2*y":fieldname="phi" -m MHDCurrentDensity -e taylor_vortex_2D.xml MHDCurrentDensityConditions.xml taylor_vortex_2D.fld MHDCurrentDensity2D.fld</parameters>
    <files>
        <file description="Mesh file">taylor_vortex_2D.xml</file>
        <file description="MHD conditions">MHDCurrentDensityConditions.xml</file>
        <file description="Field file">taylor_vortex_2D.fld</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="x" tolerance="1e-5">5.69822</value>
            <value variable="y" tolerance="1e-5">5.69822</value>
            <value variable="u" tolerance="1e-5">1.38207</value>
            <value variable="v" tolerance="1e-5">1.38207</value>
            <value variable="p" tolerance="1e-5">0.608019</value>
            <value variable="phi" tolerance="1e-5">16.117</value>
            <value variable="Jx" tolerance="1e-4">16.5848</value>
            <value variable="Jy" tolerance="1e-4">16.5848</value>
            <value variable="Jz" tolerance="1e-4">25.8566</value>
        </metric>
    </metrics>
</test>
