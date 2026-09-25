<?xml version="1.0" encoding="utf-8"?>
<test>
    <description>
    2D unsteady IP explicit diffusion, order 4, P=3.

    Values updated when the Dirichlet boundary treatment changed to the
    standard SIPG one - boundary average g, jump 2(g - u+) - carried through a
    reflected exterior state rather than through boundary-specific averaging
    weights. See library/Operators/BndCondOps/EXTERIOR_STATE_CONVENTION.md.

    The previous treatment averaged half-and-half against the raw boundary
    value, which imposed the condition at half strength. Both are consistent -
    a p-refinement study converges spectrally either way - but the new one is
    uniformly more accurate, here by 10% in L2 and 15% in Linf, with the gap
    closing as p rises. There is no legacy IP counterpart to match against;
    legacy's nearest case on this mesh is LDG, at L2 0.0083.
    </description>
    <executable>ADRSolverRedesign</executable>
    <parameters>ExDiffusion_2D_IP_hybrid_m3.xml</parameters>
    <files>
        <file description="Session File">ExDiffusion_2D_IP_hybrid_m3.xml</file>
    </files>
    <metrics>
        <metric type="L2" id="1">
            <value variable="u" tolerance="1e-8">0.00427953</value>
        </metric>
        <metric type="Linf" id="2">
            <value variable="u" tolerance="1e-8">0.0129826</value>
        </metric>
    </metrics>
</test>
