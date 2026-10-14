<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>
        A ring of prisms closed with a turn in it, which no numbering can
        orient, is split into elements that need no orienting (first order)
    </description>
    <executable>NekMesh</executable>
    <parameters> -v -m jac:list twisted_prism_ring.cgns twisted_prism_ring-out.xml:xml:test </parameters>
    <files>
        <file description="Input File">twisted_prism_ring.cgns</file>
    </files>
    <metrics>
        <metric type="regex" id="1">
            <regex>.*Total negative Jacobians: (\d+)</regex>
            <matches>
                <match><field id="0">0</field></match>
            </matches>
        </metric>
        <!-- Eleven of the twelve prisms are left alone; the one that cannot be
             oriented becomes two tetrahedra and three pyramids coned to a new
             node at its centre. -->
        <metric type="regex" id="2">
            <regex>.*# of prisms:\s*(\d+)</regex>
            <matches>
                <match><field id="0">11</field></match>
            </matches>
        </metric>
        <metric type="regex" id="3">
            <regex>.*# of pyramids:\s*(\d+)</regex>
            <matches>
                <match><field id="0">3</field></match>
            </matches>
        </metric>
        <metric type="regex" id="4">
            <regex>.*# of tetrahedra:\s*(\d+)</regex>
            <matches>
                <match><field id="0">2</field></match>
            </matches>
        </metric>
        <!-- Coning keeps the prism's five faces exactly as they were, so the
             boundary is untouched and the mesh stays conformal. -->
        <metric type="regex" id="5">
            <regex>.*No. of boundary elements\s*:\s*(\d+)</regex>
            <matches>
                <match><field id="0">36</field></match>
            </matches>
        </metric>
    </metrics>
</test>
