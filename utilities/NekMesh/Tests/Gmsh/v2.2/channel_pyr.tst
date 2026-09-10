<?xml version="1.0" encoding="utf-8" ?>
<test>
    <description>Gmsh (v2.2) high-order pyramid channel, order 4</description>
    <executable>NekMesh</executable>
    <parameters>-m jac:list channel_pyr.msh channel_pyr-out.xml:xml:test</parameters>
    <files>
        <file description="Input File">channel_pyr.msh</file>
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
    </metrics>
</test>
