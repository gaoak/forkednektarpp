<?xml version="1.0" encoding="utf-8" ?>
<test>
    <description>Project a boundary arc onto a cylinder with the cyl module</description>
    <executable>NekMesh</executable>
    <parameters>-m cyl:surf=2:r=0.1:xc=0.0:yc=0.0:N=5 -m jac:list:quality peralign_rot_cyl.msh cyl_arc-out.xml:xml:test</parameters>
    <files>
        <file description="Input File">peralign_rot_cyl.msh</file>
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
