<?xml version="1.0" encoding="utf-8" ?>
<test>
    <description>Variational optimiser gives the same mesh however many threads it is run on. The node colouring is what makes that true, so this fails if a node ends up in a colour set alongside one it shares an element with</description>
    <segment type="sequential">
        <executable>NekMesh</executable>
        <parameters>-f varopti_cube_sphere.msh varopti_cube_sphere_thread_a.xml:xml:test:uncompress -m varopti:hyperelastic:maxiter=3:nq=4:numthreads=1</parameters>
    </segment>
    <segment type="sequential">
        <executable>NekMesh</executable>
        <parameters>-f varopti_cube_sphere.msh varopti_cube_sphere_thread_b.xml:xml:test:uncompress -m varopti:hyperelastic:maxiter=3:nq=4:numthreads=4</parameters>
    </segment>
    <files>
        <file description="Input File">varopti_cube_sphere.msh</file>
    </files>
    <metrics>
        <metric type="filesmatch" id="1">
            <compare>
                <file>varopti_cube_sphere_thread_a.xml</file>
                <file>varopti_cube_sphere_thread_b.xml</file>
            </compare>
        </metric>
    </metrics>
</test>
