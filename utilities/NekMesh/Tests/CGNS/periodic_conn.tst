<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>
        A periodic grid connectivity names its faces by element, and each side
        becomes a composite of its own
    </description>
    <executable>NekMesh</executable>
    <parameters> -v -m jac:list periodic_conn.cgns periodic_conn-out.xml:xml:test </parameters>
    <files>
        <file description="Input File">periodic_conn.cgns</file>
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
        <!-- The connectivities address their faces at FaceCenter, so the
             numbers are element numbers. Reading them as vertices instead and
             then selecting faces whose corners all appear -- which is what
             this module used to do whatever the grid location said -- gave a
             face set bearing no relation to the one declared. Each side is
             100 faces exactly. -->
        <metric type="regex" id="2">
            <regex>.*# of bndry elmts in periodic_a\s*:\s*(\d+)</regex>
            <matches>
                <match><field id="0">100</field></match>
            </matches>
        </metric>
        <metric type="regex" id="3">
            <regex>.*# of bndry elmts in periodic_b\s*:\s*(\d+)</regex>
            <matches>
                <match><field id="0">100</field></match>
            </matches>
        </metric>
        <!-- Both sides of a periodic pair arrive in one section, so grouping
             by section alone gives a single composite holding both, which is
             of no use for setting the condition up: a boundary cannot be made
             periodic with itself. The two composites above are what let the
             pair be written in a session.

             The pair took 100 of the inlet's 104 faces and all 100 of the
             outlet's, so the inlet keeps 4 and the outlet no longer forms a
             composite at all. Those remainders are the assertion that the
             declared sets were resolved exactly rather than approximately,
             and the wall, which no connectivity names, is untouched. -->
        <metric type="regex" id="4">
            <regex>.*# of bndry elmts in Body1\.Inlet\s*:\s*(\d+)</regex>
            <matches>
                <match><field id="0">4</field></match>
            </matches>
        </metric>
        <metric type="regex" id="5">
            <regex>.*# of bndry elmts in Body1\.Wall\s*:\s*(\d+)</regex>
            <matches>
                <match><field id="0">420</field></match>
            </matches>
        </metric>
        <!-- 100 + 100 + 4 + 420, i.e. the whole boundary of the source mesh:
             nothing was double counted and nothing lost. -->
        <metric type="regex" id="6">
            <regex>.*No. of boundary elements\s*:\s*(\d+)</regex>
            <matches>
                <match><field id="0">624</field></match>
            </matches>
        </metric>
        <metric type="nowarning" id="7"/>
    </metrics>
</test>
