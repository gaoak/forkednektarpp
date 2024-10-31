<?xml version="1.0" encoding="utf-8"?>
<test>
    <description> interpolation on a manifold </description>
    <executable>FieldConvert</executable>
    <parameters> -e -f -v -m interppoints:fromxml=naca0012_3D_bnd.xml:fromfld=naca0012_3D_bnd.fld:topts=naca0012_3D_interp.csv:distTolerance=1E-2 naca0012_3D_interp.dat </parameters>
    <files>
        <file description="Session File">naca0012_3D_bnd.xml</file>
        <file description="Field File">naca0012_3D_bnd.fld</file>
        <file description="Field File">naca0012_3D_interp.csv</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="x" tolerance="1e-4">0.557936</value>
            <value variable="y" tolerance="1e-4">0.155526</value>
            <value variable="z" tolerance="1e-4">0.195195</value>
            <value variable="Shear_x" tolerance="1e-4">0.0911352</value>
            <value variable="Shear_y" tolerance="1e-4">0.067545</value>
            <value variable="Shear_z" tolerance="1e-4">0.013328</value>
            <value variable="Shear_mag" tolerance="1e-4">0.114146</value>
            <value variable="Norm_x" tolerance="1e-4">0.296038</value>
            <value variable="Norm_y" tolerance="1e-4">0.955176</value>
            <value variable="Norm_z" tolerance="1e-4">2.92847e-17</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="x" tolerance="1e-4">0.966252</value>
            <value variable="y" tolerance="1e-4">0.260036</value>
            <value variable="z" tolerance="1e-4">0.195195</value>
            <value variable="Shear_x" tolerance="1e-4">0.719677</value>
            <value variable="Shear_y" tolerance="1e-4">1.14682</value>
            <value variable="Shear_z" tolerance="1e-4">0.0390763</value>
            <value variable="Shear_mag" tolerance="1e-4">1.18736</value>
            <value variable="Norm_x" tolerance="1e-4">0.993076</value>
            <value variable="Norm_y" tolerance="1e-4">0.999989</value>
            <value variable="Norm_z" tolerance="1e-4">3.90539e-16</value>
        </metric>
    </metrics>
</test>
