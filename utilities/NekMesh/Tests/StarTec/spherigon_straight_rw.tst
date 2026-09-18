<?xml version="1.0" encoding="utf-8" ?>
<test>
    <description>Spherigon smooth tagged boundary faces of a 3D prism/tet mesh</description>
    <executable>NekMesh</executable>
    <parameters>-m spherigon:surf=8,9,10,13 -m extract:surf=13 -m jac:list:quality spherigon_bl_straight_rw.dat spherigon_straight_rw-out.xml:xml:test</parameters>
    <files>
        <file description="Input File">spherigon_bl_straight_rw.dat</file>
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
        <!-- Extracting surface 13 isolates the smoothed faces, which are only
             a few hundred of 45704 elements: unsmoothed it integrates to
             exactly 100%, against 122.899% once the spherigon has run. -->
        <metric type="regex" id="2">
            <regex>.*Integration of Jacobian: ([0-9]*\.[0-9]+)%?</regex>
            <matches>
                <match>
                    <field id="0" tolerance="2e-2">122.899</field>
                </match>
            </matches>
        </metric>
    </metrics>
</test>
