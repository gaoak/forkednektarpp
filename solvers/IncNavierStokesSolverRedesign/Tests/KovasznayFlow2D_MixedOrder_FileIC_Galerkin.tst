<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>
        KovasznayFlow2D_MixedOrder_FileIC with ProjectInitialConditions set
        to Galerkin. The pressure read through the velocity expansion and
        projected element by element is projected by the global mass matrix
        solve, which gives the Galerkin projection of the file at the
        velocity order. Truncating the file mode by mode to P=1 instead
        gives an L2 pressure error of 0.0760666 rather than 0.0573624.
    </description>
    <executable>IncNavierStokesSolverRedesign</executable>
    <parameters>-I ProjectInitialConditions=Galerkin KovasznayFlow2D_MixedOrder_FileIC.xml</parameters>
    <files>
        <file description="Session File">KovasznayFlow2D_MixedOrder_FileIC.xml</file>
        <file description="Initial Condition">KovasznayFlow2D_MixedOrder_FileIC_ini.fld</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="1e-6">0.290983</value>
            <value variable="v" tolerance="1e-8">0.000186111</value>
            <value variable="p" tolerance="1e-6">0.0573624</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="1e-6">0.322039</value>
            <value variable="v" tolerance="1e-8">0.000170892</value>
            <value variable="p" tolerance="1e-6">0.0824288</value>
        </metric>
    </metrics>
</test>
