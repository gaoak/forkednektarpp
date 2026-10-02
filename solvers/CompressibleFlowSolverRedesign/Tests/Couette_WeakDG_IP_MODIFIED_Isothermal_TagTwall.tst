<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>
    The isothermal Couette case with the wall temperature written on the
    boundary region instead of as a session parameter:

        USERDEFINEDTYPE="WallViscous:twall=Twall"

    It carries the same metrics as Couette_WeakDG_IP_MODIFIED_Isothermal, and
    that is the whole point - the two routes to the same temperature must give
    the same answer, so this pins the region syntax against the session
    parameter it has to remain compatible with. If the two ever diverge, one of
    them is wrong and this says so.

    The value is deliberately the expression `Twall` rather than the literal
    300.15, because the two exercise different things. A literal only shows the
    parsing works; naming a session parameter shows the expression is evaluated
    through the session's own interpreter, which is what makes `twall=TInf`
    usable and is not what the older `Rotated:` tag does with its bare
    interpreter.

    Checked in the other direction too, though a test cannot express it: with
    `twall=260.0` the E error moves from 2960.91 to 1032.99, so the region value
    is genuinely what drives the wall and is not being quietly ignored in favour
    of the session parameter that is still present in this file.
    </description>
    <executable>CompressibleFlowSolverRedesign</executable>
    <parameters>Couette_WeakDG_IP_MODIFIED_Isothermal_TagTwall.xml</parameters>
    <files>
        <file description="Session File">Couette_WeakDG_IP_MODIFIED_Isothermal_TagTwall.xml</file>
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
