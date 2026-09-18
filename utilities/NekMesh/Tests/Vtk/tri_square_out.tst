<?xml version="1.0" encoding="utf-8" ?>
<test>
    <description>Write a legacy VTK unstructured grid</description>
    <executable>NekMesh</executable>
    <parameters>tri_square.vtk tri_square_out-out.vtk:vtk:legacy</parameters>
    <files>
        <file description="Input File">tri_square.vtk</file>
    </files>
    <metrics>
        <!-- Smoke test only: the writer emits an UNSTRUCTURED_GRID while the
             reader takes POLYDATA, so there is no round trip to assert on, and
             a sha1 would pin the VTK library's formatting rather than ours. -->
        <metric type="fileexists" id="1">
            <file pattern="^.*tri_square_out-out\.vtk$">1</file>
        </metric>
        <metric type="nowarning" id="2"/>
    </metrics>
</test>
