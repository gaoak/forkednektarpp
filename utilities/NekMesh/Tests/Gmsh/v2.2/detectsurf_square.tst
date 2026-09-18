<?xml version="1.0" encoding="utf-8" ?>
<test>
    <description>Detect the contiguous surface of a 2D mesh and composite its boundary</description>
    <executable>NekMesh</executable>
    <parameters>-m detect square_tri_lin.msh detectsurf_square-out.xml:xml:test:stats</parameters>
    <files>
        <file description="Input File">square_tri_lin.msh</file>
    </files>
    <metrics>
        <!-- One block is found, so its 16 boundary edges become a single new
             composite alongside the composite of 32 elements: the 40 interior
             edges must not be picked up. -->
        <metric type="regex" id="1">
            <regex>.*Number of composites\s*:\s*(\d+)</regex>
            <matches>
                <match>
                    <field id="0">2</field>
                </match>
            </matches>
        </metric>
        <metric type="regex" id="2">
            <regex>.*Bnd elements\s*:\s*(\d+)</regex>
            <matches>
                <match>
                    <field id="0">16</field>
                </match>
            </matches>
        </metric>
    </metrics>
</test>
