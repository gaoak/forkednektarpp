<?xml version="1.0" encoding="utf-8" ?>
<test>
    <description>Extraction of curved surface</description>
    <executable>NekMesh</executable>
    <parameters>-m jac:list -m extract:surf=3 extract_curved_edge.xml extract_curved_edge-out.xml:xml:test:stats</parameters>
    <files>
        <file description="Input File">extract_curved_edge.xml</file>
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
        <metric type="regex" id="2">
            <regex>.*Elements\s*:\s*(\d+)</regex>
            <matches>
                <match>
                    <field id="0">1</field>
                </match>
            </matches>
        </metric>
        <metric type="regex" id="3">
            <regex>.*Bnd elements\s*:\s*(\d+)</regex>
            <matches>
                <match>
                    <field id="0">0</field>
                </match>
            </matches>
        </metric>
    </metrics>
</test>
