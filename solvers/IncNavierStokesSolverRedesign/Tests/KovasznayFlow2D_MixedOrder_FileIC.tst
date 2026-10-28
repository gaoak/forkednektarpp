<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>
        Kovasznay flow with a P=2 velocity and a P=1 pressure
        (MODIFIEDQUADPLUS1, on the velocity's quadrature points), started
        from a P=5 field file that holds all three fields. No step is
        taken, so the norms are those of the initial condition. The pressure
        is read through the velocity expansion, projected element by element
        onto its own, averaged, and given its Dirichlet values. The metrics
        are the legacy IncNavierStokesSolver's for the same session.
        Truncating the file mode by mode to P=1 instead, which keeps only
        the vertex values, gives an L2 pressure error of 0.0760666 rather
        than 0.0602351.
    </description>
    <executable>IncNavierStokesSolverRedesign</executable>
    <parameters>KovasznayFlow2D_MixedOrder_FileIC.xml</parameters>
    <files>
        <file description="Session File">KovasznayFlow2D_MixedOrder_FileIC.xml</file>
        <file description="Initial Condition">KovasznayFlow2D_MixedOrder_FileIC_ini.fld</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="1e-6">0.290983</value>
            <value variable="v" tolerance="1e-8">0.000189662</value>
            <value variable="p" tolerance="1e-6">0.0602351</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="1e-6">0.322039</value>
            <value variable="v" tolerance="1e-8">0.000219499</value>
            <value variable="p" tolerance="1e-6">0.0824274</value>
        </metric>
    </metrics>
</test>
