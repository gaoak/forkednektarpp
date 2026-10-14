<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>
        The single-zone half of the two-zone pair, which fixes what the mesh
        is so that the two-zone conversion can be held to it
    </description>
    <executable>NekMesh</executable>
    <parameters> -v -m jac:list one_zone.cgns one_zone-out.xml:xml:test </parameters>
    <files>
        <file description="Input File">one_zone.cgns</file>
    </files>
    <!-- Both files are generated together by MakeTwoZoneMesh.cpp and are the
         same box of hexahedra; this one is written as one zone. The numbers
         asserted here and in two_zone.tst are identical on purpose: the pair
         is only meaningful if the same numbers are claimed of both, and this
         test is what shows they are the mesh's own rather than an artefact of
         how the split one is read.

         It is also where the negative Jacobian count is held to zero for a
         plain stacked box of hexahedra, which it was not before: the upper
         layer came out inside out. -->
    <metrics>
        <metric type="regex" id="1">
            <regex>.*Total negative Jacobians: (\d+)</regex>
            <matches>
                <match><field id="0">0</field></match>
            </matches>
        </metric>
        <metric type="regex" id="2">
            <regex>.*# of hexa:\s*(\d+)</regex>
            <matches>
                <match><field id="0">16</field></match>
            </matches>
        </metric>
        <metric type="regex" id="3">
            <regex>.*No. of nodes\s*:\s*(\d+)</regex>
            <matches>
                <match><field id="0">45</field></match>
            </matches>
        </metric>
        <metric type="regex" id="4">
            <regex>.*No. of boundary elements\s*:\s*(\d+)</regex>
            <matches>
                <match><field id="0">40</field></match>
            </matches>
        </metric>
        <metric type="regex" id="5">
            <regex>.*# of bndry elmts in xmin\s*:\s*(\d+)</regex>
            <matches>
                <match><field id="0">4</field></match>
            </matches>
        </metric>
        <metric type="regex" id="6">
            <regex>.*# of bndry elmts in ymin\s*:\s*(\d+)</regex>
            <matches>
                <match><field id="0">8</field></match>
            </matches>
        </metric>
        <metric type="regex" id="7">
            <regex>.*# of bndry elmts in zmin\s*:\s*(\d+)</regex>
            <matches>
                <match><field id="0">8</field></match>
            </matches>
        </metric>
        <metric type="nowarning" id="8"/>
    </metrics>
</test>
