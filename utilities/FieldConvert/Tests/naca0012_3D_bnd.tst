<?xml version="1.0" encoding="utf-8"?>
<test>
    <description> Process fld directory without P0000000.fld </description>
    <executable>FieldConvert</executable>
    <parameters> -e -f naca0012_3D_bnd.xml naca0012_3D_bnd.fld naca0012_3D_bnd.plt </parameters>
    <files>
        <file description="Session File">naca0012_3D_bnd.xml</file>
        <file description="Field File">naca0012_3D_bnd.fld</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="x" tolerance="1e-4">1.77221</value>
            <value variable="y" tolerance="1e-4">0.493958</value>
            <value variable="z" tolerance="1e-4">9.22368</value>
            <value variable="Shear_x" tolerance="1e-4">0.313682</value>
            <value variable="Shear_y" tolerance="1e-4">0.32154</value>
            <value variable="Shear_z" tolerance="1e-4">0.0586516</value>
            <value variable="Shear_mag" tolerance="1e-4">0.453013</value>
            <value variable="Norm_x" tolerance="1e-4">0.98711</value>
            <value variable="Norm_y" tolerance="1e-4">3.03884</value>
            <value variable="Norm_z" tolerance="1e-4">7.14927e-16</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="x" tolerance="1e-4">0.966252</value>
            <value variable="y" tolerance="1e-4">0.0215327</value>
            <value variable="z" tolerance="1e-4">5</value>
            <value variable="Shear_x" tolerance="1e-4">0.820384</value>
            <value variable="Shear_y" tolerance="1e-4">1.11074</value>
            <value variable="Shear_z" tolerance="1e-4">0.808733</value>
            <value variable="Shear_mag" tolerance="1e-4">1.57498</value>
            <value variable="Norm_x" tolerance="1e-4">0.999827</value>
            <value variable="Norm_y" tolerance="1e-4">0.992511</value>
            <value variable="Norm_z" tolerance="1e-4">3.57464e-14</value>
        </metric>
    </metrics>
</test>
