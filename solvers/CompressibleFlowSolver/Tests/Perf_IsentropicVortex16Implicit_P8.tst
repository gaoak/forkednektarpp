<?xml version="1.0" encoding="utf-8"?>
<test runs="10">
    <description>Euler Isentropic Vortex P=8, Implicit</description>
    <executable>CompressibleFlowSolver</executable>
    <parameters>Perf_IsentropicVortex16Implicit_P8.xml</parameters>
    <files>
        <file description="Session File">Perf_IsentropicVortex16Implicit_P8.xml</file>
    </files>
    <!--
        This case runs at the default implicit iteration tolerances
        (NonlinIterTolRelativeL2 = 1e-3, LinSysRelativeTolInNonlin = 5e-2), so
        the errors below are not a converged solution: they are where 20 steps
        of an under-converged Newton/GMRES happen to land. Any change that
        perturbs the arithmetic at roundoff level reorders the Krylov basis and
        moves them by a few per cent. On master alone, merely switching the
        collection implementation (which changes nothing but summation order)
        moves Linf(rho) by 9.5%; removing the Riemann solver's rotation onto
        the trace normal moved it by 20%, while the converged answer was
        unchanged to six significant figures.

        The tolerances are therefore set at roughly a quarter of each value:
        wide enough not to fire on arithmetic reassociation, narrow enough to
        catch the case silently computing something wrong. They are not a
        statement about accuracy. Correctness for the implicit solver is
        covered by CylinderSubsonic_WeakDG_Implicit and, under
        NEKTAR_TEST_ALL, IsentropicVortex16Implicit_P4.

        Converging the case instead (1e-12 / 1e-10) makes the errors
        reproducible to all printed digits across builds, but costs roughly 4x
        the run time, which is not worth it for a throughput-oriented job.
    -->
    <metrics>
        <metric type="L2" id="1">
            <value variable="rho" tolerance="7e-07">2.74197e-06</value>
            <value variable="rhou" tolerance="1e-06">4.38622e-06</value>
            <value variable="rhov" tolerance="2e-06">6.16852e-06</value>
            <value variable="E" tolerance="4e-06">1.42358e-05</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="rho" tolerance="2e-06">9.3434e-06</value>
            <value variable="rhou" tolerance="3e-06">1.09661e-05</value>
            <value variable="rhov" tolerance="2e-06">9.54716e-06</value>
            <value variable="E" tolerance="7e-06">2.77985e-05</value>
        </metric>
        <metric type="ExecutionTime" id="3">
            <value tolerance="1e0" hostname="42.debian-bullseye-performance-build-and-test">37.0</value>
        </metric>
    </metrics>
</test>
