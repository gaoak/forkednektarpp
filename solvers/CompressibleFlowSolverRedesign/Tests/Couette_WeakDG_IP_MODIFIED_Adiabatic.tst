<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>
    Couette flow with an adiabatic no-slip wall, ported from the legacy
    CompressibleFlowSolver case of the same name.

    The initial state is the isothermal Couette profile, whose wall temperature
    gradient is non-zero, so it does not satisfy dT/dn = 0. A solver that
    imposes the adiabatic condition must therefore drift away from it, and that
    drift is what is measured. The case separates three outcomes by decades:

      adiabatic imposed      E ~ 6e1   (this file)
      no thermal condition   E ~ 1e-8  (the energy flux weight left at 1)
      viscosity defaulted    E ~ 2e-3  (ReferenceValues section missing)

    Metrics and tolerances are the legacy file's, unchanged. The redesign
    reproduces every one of them to all printed digits, so any movement here is
    a regression rather than a discretisation difference.
    </description>
    <executable>CompressibleFlowSolverRedesign</executable>
    <parameters>Couette_WeakDG_IP_MODIFIED_Adiabatic.xml</parameters>
    <files>
        <file description="Session File">Couette_WeakDG_IP_MODIFIED_Adiabatic.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="rho"  tolerance="1e-6">0.001587</value>
            <value variable="rhou" tolerance="1e-7">0.00988025</value>
            <value variable="rhov" tolerance="1e-7">0.0476664</value>
            <value variable="E"    tolerance="1e-4">61.7316</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="rho"  tolerance="1e-7">0.00648549</value>
            <value variable="rhou" tolerance="1e-6">0.0120572</value>
            <value variable="rhov" tolerance="1e-7">0.0450055</value>
            <value variable="E"    tolerance="1e-4">52.0678</value>
        </metric>
    </metrics>
</test>
