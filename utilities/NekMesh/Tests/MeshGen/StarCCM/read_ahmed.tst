<?xml version="1.0" encoding="utf-8"?>
<test>
    <description> NekMesh reading a binary Star-CCM+ mesh of prisms and tetrahedra </description>
    <executable>NekMesh</executable>
    <parameters> -v -m jac:list projectcad_ahmed.ccm read_ahmed-out.xml:xml:test </parameters>
    <files>
        <file description="Input File">projectcad_ahmed.ccm</file>
    </files>
    <metrics>
        <metric type="regex" id="1">
            <regex>.*# of prisms: (\d+)</regex>
            <matches>
                <match>
                    <field id="0">1218</field>
                </match>
            </matches>
        </metric>
        <metric type="regex" id="2">
            <regex>.*# of pyramids: (\d+)</regex>
            <matches>
                <match>
                    <field id="0">0</field>
                </match>
            </matches>
        </metric>
        <metric type="regex" id="3">
            <regex>.*# of tetrahedra: (\d+)</regex>
            <matches>
                <match>
                    <field id="0">3965</field>
                </match>
            </matches>
        </metric>
        <metric type="regex" id="4">
            <regex>.*No. of nodes\s+: (\d+)</regex>
            <matches>
                <match>
                    <field id="0">1915</field>
                </match>
            </matches>
        </metric>
        <metric type="regex" id="5">
            <regex>.*No. of 3D elements\s+: (\d+)</regex>
            <matches>
                <match>
                    <field id="0">5183</field>
                </match>
            </matches>
        </metric>
        <metric type="regex" id="6">
            <regex>.*No. of boundary elements\s+: (\d+)</regex>
            <matches>
                <match>
                    <field id="0">2292</field>
                </match>
            </matches>
        </metric>
        <metric type="regex" id="7">
            <regex>.*Total negative Jacobians: (\d+)</regex>
            <matches>
                <match>
                    <field id="0">0</field>
                </match>
            </matches>
        </metric>
    </metrics>
</test>
