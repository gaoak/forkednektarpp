<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>
        A mesh split across two abutting zones converts to the same thing as
        the same mesh written as one zone
    </description>
    <executable>NekMesh</executable>
    <parameters> -v -m jac:list two_zone.cgns two_zone-out.xml:xml:test </parameters>
    <files>
        <file description="Input File">two_zone.cgns</file>
    </files>
    <!-- The same box of hexahedra as one_zone.cgns, cut down the middle into
         two zones that abut across the cut, with the vertices on the cut
         written once in each zone. Every number below is one_zone.tst's, so
         what is being asserted is that the split makes no difference: the
         zones are read into one mesh, their duplicated vertices become one
         vertex each, the faces of the cut become interior rather than
         boundary, and the halves of a part ID that each zone names separately
         come together under the family they share.

         A reader taking only the first zone, as this one did, halves the
         element count. One leaving the duplicated vertices alone keeps 64
         rather than 56 and leaves the mesh in two pieces touching along a
         seam. One keeping the cut as a boundary reports 60 boundary faces
         rather than 54. -->
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
        <!-- The cut is 3 x 3 vertices, written once per zone. -->
        <metric type="regex" id="8">
            <regex>.*Merged (\d+) duplicated vertices.*</regex>
            <matches>
                <match><field id="0">9</field></match>
            </matches>
        </metric>
        <!-- The file declares which faces abut; the merge is checked against
             it rather than merely believed. -->
        <metric type="regex" id="9">
            <regex>.*All (\d+) declared abutting face pairs came together.*</regex>
            <matches>
                <match><field id="0">8</field></match>
            </matches>
        </metric>
        <metric type="nowarning" id="10"/>
    </metrics>
</test>
