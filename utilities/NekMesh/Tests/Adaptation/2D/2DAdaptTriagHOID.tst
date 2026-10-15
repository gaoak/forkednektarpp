<?xml version="1.0" encoding="utf-8" ?>
<test>
	<description>2D h-splitting based on IDs  </description>
    <executable>NekMesh</executable>
    <parameters>-m h_adaptation:order=4:finest=1:adaptfile=AdaptID2D.txt -m jac:list -m connectivity ring.xml ringAdaptID2d.xml:xml:test</parameters>
    <files>
        <file description="Input File">ring.xml</file>
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
                    <field id="0">12</field>
                </match>
            </matches>
        </metric>
        <metric type="regex" id="12">
            <regex>.*Edge count\s+:\s+(\d+)</regex>
            <matches>
                <match>
                    <field id="0">23</field>
                </match>
            </matches>
        </metric>
        <metric type="regex" id="13">
            <regex>.*- Elements\s+:\s+(\d+)</regex>
            <matches>
                <match>
                    <field id="0">12</field>
                </match>
            </matches>
        </metric>
        <metric type="regex" id="14">
            <regex>.*Bnd elements\s+:\s+(\d+)</regex>
            <matches>
                <match>
                    <field id="0">10</field>
                </match>
            </matches>
        </metric>
        <metric type="regex" id="15">
            <regex>.*Number of composites\s+:\s+(\d+)</regex>
            <matches>
                <match>
                    <field id="0">7</field>
                </match>
            </matches>
        </metric>
        <metric type="regex" id="16">
            <regex>.*Lower mesh extent\s+:\s+(\S+)\s+(\S+)\s+(\S+)</regex>
            <matches>
                <match>
                    <field id="0" tolerance="1e-8">-0.001</field>
                    <field id="1" tolerance="1e-8">0</field>
                    <field id="2" tolerance="1e-8">0</field>
                </match>
            </matches>
        </metric>
        <metric type="regex" id="17">
            <regex>.*Upper mesh extent\s+:\s+(\S+)\s+(\S+)\s+(\S+)</regex>
            <matches>
                <match>
                    <field id="0" tolerance="1e-8">0.001</field>
                    <field id="1" tolerance="1e-8">0.001</field>
                    <field id="2" tolerance="1e-8">0</field>
                </match>
            </matches>
        </metric>
        <metric type="regex" id="18">
            <regex>.*- Segment\s+:\s+(\d+)\s+(\d+)\s+(\d+)\s*</regex>
            <matches>
                <match>
                    <field id="0">0</field>
                    <field id="1">10</field>
                    <field id="2">10</field>
                </match>
            </matches>
        </metric>
        <metric type="regex" id="19">
            <regex>.*- Triangle\s+:\s+(\d+)\s+(\d+)\s+(\d+)\s*</regex>
            <matches>
                <match>
                    <field id="0">0</field>
                    <field id="1">12</field>
                    <field id="2">12</field>
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
