<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>
        Every pyramid that cannot be numbered as the reader would like is
        shielded, not only the first
    </description>
    <executable>NekMesh</executable>
    <parameters> -v -m jac:list two_impossible_pyramids.cgns two_impossible_pyramids-out.xml:xml:test </parameters>
    <files>
        <file description="Input File">two_impossible_pyramids.cgns</file>
    </files>
    <!-- Two translated copies of ImpossiblePyramids_linear.cgns in one zone,
         far enough apart to share nothing, so each is shielded exactly as it
         is on its own. See MakeTwoImpossiblePyramids.cpp, which generates it.

         A pyramid's apex has to be the highest-numbered node of the pyramid,
         since that is where the Duffy transformation collapses, and where the
         mesh makes that impossible ResetNodes replaces the pyramid with a
         smaller one and four tetrahedra. It used to stop looking once it had
         done that to one pyramid, leaving every later one without an apex id
         at all. Every other pyramid fixture here contains exactly one such
         pyramid, so nothing noticed. This file shields once and then fails
         with

           Orientation of triangular face (id = 13) is inconsistent with
           face 4 of pyramid element (id = 3) since Dir2 is aligned with Dir1

         The four pyramids below are the two originals and the two smaller
         ones that replace them; the eight tetrahedra are four per shield. -->
    <metrics>
        <metric type="regex" id="1">
            <regex>.*Total negative Jacobians: (\d+)</regex>
            <matches>
                <match><field id="0">0</field></match>
            </matches>
        </metric>
        <metric type="regex" id="2">
            <regex>.*# of pyramids:\s*(\d+)</regex>
            <matches>
                <match><field id="0">4</field></match>
            </matches>
        </metric>
        <metric type="regex" id="3">
            <regex>.*# of tetrahedra:\s*(\d+)</regex>
            <matches>
                <match><field id="0">8</field></match>
            </matches>
        </metric>
        <metric type="regex" id="4">
            <regex>.*No. of 3D elements\s*:\s*(\d+)</regex>
            <matches>
                <match><field id="0">12</field></match>
            </matches>
        </metric>
    </metrics>
</test>
