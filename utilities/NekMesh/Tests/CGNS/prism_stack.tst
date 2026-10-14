<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>
        A stack of prisms whose elements are written the same way round as
        each other converts, whatever order each one lists its nodes in
    </description>
    <executable>NekMesh</executable>
    <parameters> -v -m jac:list prism_stack.cgns prism_stack-out.xml:xml:test </parameters>
    <files>
        <file description="Input File">prism_stack.cgns</file>
    </files>
    <!-- An irregularly triangulated square extruded into four graded layers
         of prisms, with each prism's corners cyclically rotated by an amount
         a fixed seed picks. The rotation is what this is testing. It is the
         same element written a different way round, and a mesher is free to
         write it either way, but the reader rebuilds an element from its
         faces rather than from its stored node order and anchors what it
         builds on the first triangular and first quadrilateral face it meets,
         so the rotation decides which corner of each triangular cross-section
         the element ends up collapsed onto.

         Those corners have to agree all the way along a stack, because the
         modified C^0 basis applies the Duffy transformation about them and
         two elements sharing a face must collapse onto the same place. The
         pass in InputCGNS::ResetNodes that is meant to deliver that was
         inoperative: the flags saying which faces it had already numbered
         were never set, so the branches that carry the choice from one prism
         to the next could not run and every prism chose afresh. This file
         did not convert, failing with

           Orientation of triangular face (id = 36) is inconsistent with
           face 1 of prism element (id = 80) since Dir2 is aligned with Dir1

         which is the same error a mesh of this shape from a real exporter
         gives. Rotating the prisms is the whole difficulty: the same mesh
         written with every prism the same way round converts either way. -->
    <metrics>
        <metric type="regex" id="1">
            <regex>.*Total negative Jacobians: (\d+)</regex>
            <matches>
                <match><field id="0">0</field></match>
            </matches>
        </metric>
        <metric type="regex" id="2">
            <regex>.*# of prisms:\s*(\d+)</regex>
            <matches>
                <match><field id="0">288</field></match>
            </matches>
        </metric>
        <metric type="regex" id="3">
            <regex>.*No. of nodes\s*:\s*(\d+)</regex>
            <matches>
                <match><field id="0">245</field></match>
            </matches>
        </metric>
        <metric type="regex" id="4">
            <regex>.*No. of boundary elements\s*:\s*(\d+)</regex>
            <matches>
                <match><field id="0">240</field></match>
            </matches>
        </metric>
        <!-- The two triangulated surfaces, one at each end of the stack. -->
        <metric type="regex" id="5">
            <regex>.*# of bndry elmts in wall\s*:\s*(\d+)</regex>
            <matches>
                <match><field id="0">72</field></match>
            </matches>
        </metric>
        <metric type="regex" id="6">
            <regex>.*# of bndry elmts in farfield\s*:\s*(\d+)</regex>
            <matches>
                <match><field id="0">72</field></match>
            </matches>
        </metric>
        <metric type="nowarning" id="7"/>
    </metrics>
</test>
