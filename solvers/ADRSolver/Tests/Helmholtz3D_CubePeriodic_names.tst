<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>
        3D Helmholtz/Steady Diffusion Reaction P=3, where every composite and
        boundary region is named and the conditions - including the periodic
        pair - refer to them by name
    </description>
    <executable>ADRSolver</executable>
    <parameters>Helmholtz3D_CubePeriodic_names.xml</parameters>
    <files>
      <file description="Session File">Helmholtz3D_CubePeriodic_names.xml</file>
    </files>
    <!-- Helmholtz3D_CubePeriodic.xml with names put on its composites and
         boundary regions, so the values are that test's to the digit. The
         parallel counterpart of this test is where the names are really
         exercised; this one holds the serial answer the two have to agree on. -->
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="5e-5">4.73887e-05</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="4e-3">0.00876237 </value>
        </metric>
    </metrics>
</test>
