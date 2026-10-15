<?xml version="1.0" encoding="utf-8" ?>
<test>
	<description>2D h-adaptation for hybrid Quad-Triag Mesh </description>
    <executable>NekMesh</executable>
    <parameters>-m h_adaptation:uniform=1:order=4:finest -m h_adaptation:uniform=1:order=4:finest -m jac:list -m connectivity cyl.xml test.xml:xml:test</parameters>
    <files>
        <file description="Input File">cyl.xml</file>
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
        <metric type="regex" id="11">
            <regex>.*Node count\s+:\s+(\d+)</regex>
            <matches>
                <match>
                    <field id="0">574</field>
                </match>
            </matches>
        </metric>
        <metric type="regex" id="12">
            <regex>.*Edge count\s+:\s+(\d+)</regex>
            <matches>
                <match>
                    <field id="0">1454</field>
                </match>
            </matches>
        </metric>
        <metric type="regex" id="13">
            <regex>.*- Elements\s+:\s+(\d+)</regex>
            <matches>
                <match>
                    <field id="0">880</field>
                </match>
            </matches>
        </metric>
        <metric type="regex" id="14">
            <regex>.*Bnd elements\s+:\s+(\d+)</regex>
            <matches>
                <match>
                    <field id="0">76</field>
                </match>
            </matches>
        </metric>
        <metric type="regex" id="15">
            <regex>.*Number of composites\s+:\s+(\d+)</regex>
            <matches>
                <match>
                    <field id="0">10</field>
                </match>
            </matches>
        </metric>
        <metric type="regex" id="16">
            <regex>.*Lower mesh extent\s+:\s+(\S+)\s+(\S+)\s+(\S+)</regex>
            <matches>
                <match>
                    <field id="0" tolerance="1e-8">-5</field>
                    <field id="1" tolerance="1e-8">-5</field>
                    <field id="2" tolerance="1e-8">0</field>
                </match>
            </matches>
        </metric>
        <metric type="regex" id="17">
            <regex>.*Upper mesh extent\s+:\s+(\S+)\s+(\S+)\s+(\S+)</regex>
            <matches>
                <match>
                    <field id="0" tolerance="1e-8">15</field>
                    <field id="1" tolerance="1e-8">5</field>
                    <field id="2" tolerance="1e-8">0</field>
                </match>
            </matches>
        </metric>
        <metric type="regex" id="18">
            <regex>.*- Segment\s+:\s+(\d+)\s+(\d+)\s+(\d+)\s*</regex>
            <matches>
                <match>
                    <field id="0">0</field>
                    <field id="1">76</field>
                    <field id="2">76</field>
                </match>
            </matches>
        </metric>
        <metric type="regex" id="19">
            <regex>.*- Triangle\s+:\s+(\d+)\s+(\d+)\s+(\d+)\s*</regex>
            <matches>
                <match>
                    <field id="0">0</field>
                    <field id="1">688</field>
                    <field id="2">688</field>
                </match>
            </matches>
        </metric>
        <metric type="regex" id="20">
            <regex>.*- Quadrilateral\s+:\s+(\d+)\s+(\d+)\s+(\d+)\s*</regex>
            <matches>
                <match>
                    <field id="0">0</field>
                    <field id="1">192</field>
                    <field id="2">192</field>
                </match>
            </matches>
        </metric>
        <metric type="nowarning" id="3">
            <regex>Link Check Error</regex>
        </metric>
        <metric type="nowarning" id="4">
            <regex>Untagged</regex>
        </metric>
    </metrics>
</test>
