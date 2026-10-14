<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>
        Gmsh physical names become composite names, including across the
        composites one mixed-type physical group is split into
    </description>
    <executable>NekMesh</executable>
    <parameters> -v -m jac:list cube_all_physnames.msh cube_all_physnames-out.xml:xml:test:cond=cube_all_physnames-cond.xml </parameters>
    <files>
        <file description="Input File">cube_all_physnames.msh</file>
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
        <!-- This is cube_all with names added. Physical group 0 is the volume
             and holds tetrahedra, pyramids, prisms and hexahedra, which a
             composite cannot, so it is split across composites 0, 7, 8 and 9.
             All four take the group's one name, which is what lets a session
             say C[fluid] and mean the whole volume; no single ID can. -->
        <metric type="regex" id="2">
            <regex>.*composite (\d+) named 'fluid'</regex>
            <matches>
                <match><field id="0">0</field></match>
                <match><field id="0">7</field></match>
                <match><field id="0">8</field></match>
                <match><field id="0">9</field></match>
            </matches>
        </metric>
        <!-- The six faces of the cube are one composite each, so their names
             land on them directly. -->
        <metric type="regex" id="3">
            <regex>.*composite (\d+) named 'xmin'</regex>
            <matches>
                <match><field id="0">1</field></match>
            </matches>
        </metric>
        <metric type="regex" id="4">
            <regex>.*composite (\d+) named 'zmax'</regex>
            <matches>
                <match><field id="0">6</field></match>
            </matches>
        </metric>
        <!-- Only the six faces become boundary regions; the volume is not one. -->
        <metric type="regex" id="5">
            <regex>.*Writing conditions file '.*' with (\d+) boundary regions.*</regex>
            <matches>
                <match><field id="0">6</field></match>
            </matches>
        </metric>
        <metric type="nowarning" id="6"/>
    </metrics>
</test>
