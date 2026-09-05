<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>
        2D unsteady implicit diffusion on a hybrid tri+quad mesh, run as 50
        steps, stopped, and resumed from the field file for 50 more. The
        metrics are those of the resumed run - each segment redirects to the
        same output.out, so the metrics see the last one - and match the
        uninterrupted 100 step case to within the coefficient round trip.
        The mesh is deliberately hybrid: GetFieldDefinitions() emits one
        definition per element shape and the reader is called once per
        definition, accumulating, so a reader that starts from zero on each
        call returns only the shape it saw last and silently zeros the rest.
        No single shape mesh can detect that. Checked against the bug it is
        written for: with the accumulate fix reverted the resumed run reports
        an L2 error of 0.642792, the norm of the exact solution, because the
        two quadrilaterals come back as zero.
    </description>
    <segment type="sequential">
        <executable>ADRSolverRedesign</executable>
        <parameters>ReactionDiffusion2D_Restart_a.xml</parameters>
        <processes>1</processes>
    </segment>
    <segment type="sequential">
        <executable>ADRSolverRedesign</executable>
        <parameters>ReactionDiffusion2D_Restart_b.xml</parameters>
        <processes>1</processes>
    </segment>
    <files>
        <file description="Session File"> ReactionDiffusion2D_Restart_a.xml </file>
        <file description="Session File"> ReactionDiffusion2D_Restart_b.xml </file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="2.5e-9"> 8.81474e-09 </value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="5.0e-9"> 3.06653e-08 </value>
        </metric>
    </metrics>
</test>
