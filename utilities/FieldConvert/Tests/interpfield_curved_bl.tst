<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>
        Interpolate a linear field between two splittings of a curved
        boundary-layer mesh
    </description>
    <executable>FieldConvert</executable>
    <parameters>-f -e -m interpfield:fromxml=interpfield_curved_bl_from.xml:fromfld=interpfield_curved_bl_from.fld:defaultvalue=100 -m fieldfromstring:fieldstr="u-(x+2*y)":fieldname="err" interpfield_curved_bl_to.xml interpfield_curved_bl_to.fld</parameters>
    <files>
        <file description="Source mesh">interpfield_curved_bl_from.xml</file>
        <file description="Source field">interpfield_curved_bl_from.fld</file>
        <file description="Target mesh">interpfield_curved_bl_to.xml</file>
    </files>
    <!-- Both meshes split the same coarse mesh of curved quadrilaterals into
         four layers along its bottom wall, uniformly and with a growth ratio
         of 3.95, and share their boundary. Only the elements with centroid
         x > 0.3 are kept, to keep the test quick. The geometry is of degree 5
         and the expansion of order 5, so the source field u = x + 2y is exact
         on both meshes, and err is zero wherever interpfield places the
         target point correctly. A point it cannot find takes the default
         value 100, so it shows up as an error of order 100.

         The point search in curved elements used to fail here: points went
         unfound, and one was placed at the wrong local coordinates by a
         Newton iteration that had not converged. A 2D iteration started from
         the element centre also misses points near the top corner at x = 1.
         The test tolerances allow for the Newton iteration's own tolerance. -->
    <metrics>
        <metric type="L2" id="1">
            <value variable="x" tolerance="1e-6">0.567121</value>
            <value variable="y" tolerance="1e-6">0.263987</value>
            <value variable="u" tolerance="1e-5">0.84655</value>
            <value variable="err" tolerance="1e-5">0</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="x" tolerance="1e-6">1</value>
            <value variable="y" tolerance="1e-6">0.77419</value>
            <value variable="u" tolerance="1e-4">2.14405</value>
            <value variable="err" tolerance="1e-4">0</value>
        </metric>
    </metrics>
</test>
