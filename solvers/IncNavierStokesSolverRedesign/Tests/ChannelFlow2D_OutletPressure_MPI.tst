<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>
        2D Poiseuille channel, one element wide, on two processes, with the
        pressure Dirichlet only on the outlet. The partition that does not
        reach the outlet sees only Neumann pressure conditions locally, so
        it must not decide on its own that the pressure system is singular.
        If it does, only that rank removes the pressure mean, which is a
        collective reduction, and the run deadlocks.
    </description>
    <executable>IncNavierStokesSolverRedesign</executable>
    <parameters>ChannelFlow2D_OutletPressure_MPI.xml</parameters>
    <processes>2</processes>
    <files>
        <file description="Session File">ChannelFlow2D_OutletPressure_MPI.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="1e-10">0</value>
            <value variable="v" tolerance="1e-10">0</value>
            <value variable="p" tolerance="1e-10">0</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="1e-10">0</value>
            <value variable="v" tolerance="1e-10">0</value>
            <value variable="p" tolerance="1e-10">0</value>
        </metric>
    </metrics>
</test>
