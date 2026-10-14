<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>
        Two stacks of prisms whose layers run at right angles to each other
        convert, whatever order their nodes are numbered in
    </description>
    <executable>NekMesh</executable>
    <parameters> -v -m jac:list angled_prism_layers.cgns angled_prism_layers-out.xml:xml:test </parameters>
    <files>
        <file description="Input File">angled_prism_layers.cgns</file>
    </files>
    <!-- The smallest mesh that puts the reader where a boundary layer round a
         fillet does. Each quadrilateral face of a prism holds one opposite
         pair of its extrusion edges, and two prisms sharing such a face may
         take different pairs: one stack rises in z here and the other runs
         out in y, and they share the square at x = 1. One stack's extrusion
         edge is then an edge of the other's cross-section.

         InputCGNS::ResetNodes groups nodes into lines by joining the ends of
         every extrusion edge, and here that pulls all four corners of the
         shared square onto one line, since each stack joins one pair and
         between them they close the loop. Both prisms of a stack are then
         left with two corners of a cross-section on the same line, and the
         order of the lines cannot say which of those two comes out higher --
         which it must, because that corner is the one OrientPrism collapses
         onto and the Duffy transformation has to meet across the shared
         faces. What settles it is the third corner, which is the highest at
         both ends as long as its line is numbered after theirs.

         Each stack is two prisms deep so that the triangular faces between
         them are interior and so have to agree. With the line order left to
         chance this file was refused with

           Orientation of triangular face (id = 3) is inconsistent with face 1
           of prism element (id = 1) since Dir2 is aligned with Dir1

         Only some numberings go wrong, which is why the generator takes a
         seed; this is seed 4. See MakeAngledPrismLayers.cpp. -->
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
                <match><field id="0">4</field></match>
            </matches>
        </metric>
        <metric type="regex" id="3">
            <regex>.*No. of nodes\s*:\s*(\d+)</regex>
            <matches>
                <match><field id="0">14</field></match>
            </matches>
        </metric>
        <metric type="regex" id="4">
            <regex>.*No. of boundary elements\s*:\s*(\d+)</regex>
            <matches>
                <match><field id="0">14</field></match>
            </matches>
        </metric>
        <metric type="nowarning" id="5"/>
    </metrics>
</test>
