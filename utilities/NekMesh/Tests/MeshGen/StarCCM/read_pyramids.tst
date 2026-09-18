<?xml version="1.0" encoding="utf-8"?>
<test>
    <description> NekMesh reading a binary Star-CCM+ mesh including pyramids </description>
    <executable>NekMesh</executable>
    <parameters> -v -m jac:list projectcad_pyramids.ccm read_pyramids-out.xml:xml:test </parameters>
    <files>
        <file description="Input File">projectcad_pyramids.ccm</file>
    </files>
    <metrics>
        <metric type="regex" id="1">
            <regex>.*# of prisms: (\d+)</regex>
            <matches>
                <match>
                    <field id="0">189</field>
                </match>
            </matches>
        </metric>
        <metric type="regex" id="2">
            <regex>.*# of pyramids: (\d+)</regex>
            <matches>
                <match>
                    <field id="0">10</field>
                </match>
            </matches>
        </metric>
        <metric type="regex" id="3">
            <regex>.*# of tetrahedra: (\d+)</regex>
            <matches>
                <match>
                    <field id="0">2606</field>
                </match>
            </matches>
        </metric>
        <metric type="regex" id="4">
            <regex>.*No. of nodes\s+: (\d+)</regex>
            <matches>
                <match>
                    <field id="0">832</field>
                </match>
            </matches>
        </metric>
        <metric type="regex" id="5">
            <regex>.*No. of 3D elements\s+: (\d+)</regex>
            <matches>
                <match>
                    <field id="0">2805</field>
                </match>
            </matches>
        </metric>
        <metric type="regex" id="6">
            <regex>.*No. of boundary elements\s+: (\d+)</regex>
            <matches>
                <match>
                    <field id="0">1077</field>
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
