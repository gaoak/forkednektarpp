<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>
    Couette flow with an isothermal no-slip wall, held at Twall = 300.15.

    The same session as Couette_WeakDG_IP_MODIFIED_Adiabatic with the wall
    tagged WallViscous instead of WallAdiabatic, so the two differ only in the
    thermal condition. The initial profile puts the wall at T0 = 0.8*Twall, so
    an isothermal wall drives a strong transient the adiabatic one does not,
    and the cases separate by two orders of magnitude:

      isothermal at Twall    E ~ 3e3   (this file)
      adiabatic              E ~ 6e1   (the sibling test)

    Against the legacy solver on the same case the agreement is exact on rho
    and to six significant figures elsewhere. It is not bit-for-bit, and the
    difference is accumulation rather than discretisation: at 10 steps the two
    agree to every printed digit, at 100 the sixth digit parts, and it stays
    there. Legacy overwrites the averaged energy in place
    (NavierStokesCFE::SpecialBndTreat) where the redesign expresses the same
    condition through the exterior state, which round-trips through the total
    energy and rounds differently. See
    library/Operators/BndCondOps/EXTERIOR_STATE_CONVENTION.md.

    Tolerances are therefore set to catch a regression, not to assert bitwise
    agreement with legacy.
    </description>
    <executable>CompressibleFlowSolverRedesign</executable>
    <parameters>Couette_WeakDG_IP_MODIFIED_Isothermal.xml</parameters>
    <files>
        <file description="Session File">Couette_WeakDG_IP_MODIFIED_Isothermal.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="rho"  tolerance="1e-6">0.0891671</value>
            <value variable="rhou" tolerance="1e-5">0.739442</value>
            <value variable="rhov" tolerance="1e-4">2.18057</value>
            <value variable="E"    tolerance="1e-1">2960.91</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="rho"  tolerance="1e-6">0.286544</value>
            <value variable="rhou" tolerance="1e-5">0.791275</value>
            <value variable="rhov" tolerance="1e-4">2.92904</value>
            <value variable="E"    tolerance="1e-1">2997.98</value>
        </metric>
    </metrics>
</test>
