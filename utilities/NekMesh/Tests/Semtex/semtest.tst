<?xml version="1.0" encoding="utf-8" ?>
<test>
    <description>Read a Semtex session with a curved element and surface groups</description>
    <executable>NekMesh</executable>
    <parameters>-m jac:list:quality semtest.sem semtest-out.xml:xml:test:stats</parameters>
    <files>
        <file description="Input File">semtest.sem</file>
        <file description="High-order mesh from meshpr">semtest.msh</file>
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
        <!-- Two unit quadrilaterals, the second raised to order three by the
             CURVES section. This is sensitive to the element interiors and not
             just the edges: reading a CURVES block against the wrong element
             gives a negative Jacobian and 31.62% here. -->
        <metric type="regex" id="2">
            <regex>.*Integration of Jacobian: ([0-9]*\.[0-9]+)%?</regex>
            <matches>
                <match>
                    <field id="0" tolerance="1e-3">91.5091</field>
                </match>
            </matches>
        </metric>
        <!-- One composite of elements plus one per surface group, and the six
             sides the SURFACES section names, which is the boundary tagging
             path. -->
        <metric type="regex" id="3">
            <regex>.*Number of composites\s*:\s*(\d+)</regex>
            <matches>
                <match>
                    <field id="0">3</field>
                </match>
            </matches>
        </metric>
        <metric type="regex" id="4">
            <regex>.*Bnd elements\s*:\s*(\d+)</regex>
            <matches>
                <match>
                    <field id="0">6</field>
                </match>
            </matches>
        </metric>
    </metrics>
</test>
