<?xml version="1.0" encoding="utf-8" ?>
<test>
    <description>Curve a straight boundary onto y = f(x) with the curve module</description>
    <executable>NekMesh</executable>
    <parameters>-m curve:surf=1:function="0.05*sin(PI*(x+1)/2)":N=5 -m jac:list:quality square_tri_lin.msh curve_square-out.xml:xml:test</parameters>
    <files>
        <file description="Input File">square_tri_lin.msh</file>
    </files>
    <metrics>
        <metric type="regex" id="1">
            <regex>.*Total negative Jacobians: (\d+)</regex>
            <matches>
                <match>
                    <field id="0">0</field>
                </match>
            </matches>
        </metric>
        <!-- The straight mesh integrates to exactly 100%, printed without a
             decimal point, so this both pins the value and fails outright if
             the module made no difference. -->
        <metric type="regex" id="2">
            <regex>.*Integration of Jacobian: (\d+)\.\d+%</regex>
            <matches>
                <match>
                    <field id="0">93</field>
                </match>
            </matches>
        </metric>
    </metrics>
</test>
