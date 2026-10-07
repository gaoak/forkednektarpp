<?xml version="1.0" encoding="utf-8"?>
<test>
    <description> Compute compressible flow fields for a 3DH1D expansion (conservative variables built from an incompressible field, so that p = 1e5 + p_inc)</description>
    <executable>FieldConvert</executable>
    <parameters> -f -e -m fieldfromstring:fieldstr="1.2+0.1*u":fieldname="rho" -m fieldfromstring:fieldstr="rho*u":fieldname="rhou" -m fieldfromstring:fieldstr="rho*v":fieldname="rhov" -m fieldfromstring:fieldstr="rho*w":fieldname="rhow" -m fieldfromstring:fieldstr="(1e5+p)/0.4+0.5*rho*(u*u+v*v+w*w)":fieldname="E" -m removefield:fieldname="u,v,w,p" -m compressiblefields chan3DH1D.xml chan3DH1D.fld compressiblefields_3DH1D.fld</parameters>
    <files>
        <file description="Session File">chan3DH1D.xml</file>
        <file description="Session File">chan3DH1D.fld</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="x" tolerance="1e-6">0.57735</value>
            <value variable="y" tolerance="1e-6">0.57735</value>
            <value variable="z" tolerance="1e-6">0.522913</value>
            <value variable="rho" tolerance="1e-5">1.21669</value>
            <value variable="rhou" tolerance="1e-6">0.223003</value>
            <value variable="rhov" tolerance="1e-12">0</value>
            <value variable="rhow" tolerance="1e-12">0</value>
            <value variable="E" tolerance="1">250003</value>
            <value variable="u" tolerance="1e-6">0.182574</value>
            <value variable="v" tolerance="1e-12">0</value>
            <value variable="w" tolerance="1e-12">0</value>
            <value variable="p" tolerance="1">100001</value>
            <value variable="T" tolerance="1e-3">286.344</value>
            <value variable="s" tolerance="1e-2">4003.54</value>
            <value variable="a" tolerance="1e-3">339.226</value>
            <value variable="Mach" tolerance="1e-9">0.000539271</value>
        </metric>
    </metrics>
</test>
