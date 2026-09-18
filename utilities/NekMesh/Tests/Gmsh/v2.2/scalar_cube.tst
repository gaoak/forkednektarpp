<?xml version="1.0" encoding="utf-8" ?>
<test>
    <description>Impose a scalar function z = f(x,y) on a surface composite</description>
    <executable>NekMesh</executable>
    <parameters>-m scalar:surf=1:nq=5:scalar="0.05*x*y" -m jac:list:quality cube_tet_lin.msh scalar_cube-out.xml:xml:test</parameters>
    <files>
        <file description="Input File">cube_tet_lin.msh</file>
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
                    <field id="0">99</field>
                </match>
            </matches>
        </metric>
    </metrics>
</test>
