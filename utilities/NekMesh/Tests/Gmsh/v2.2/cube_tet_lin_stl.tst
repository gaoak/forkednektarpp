<?xml version="1.0" encoding="utf-8" ?>
<test>
    <description>Write the surface of a linear tet cube as an STL file</description>
    <executable>NekMesh</executable>
    <parameters>cube_tet_lin.msh cube_tet_lin.stl:stl</parameters>
    <files>
        <file description="Input File">cube_tet_lin.msh</file>
    </files>
    <metrics>
        <metric type="file" id="1">
            <file filename="cube_tet_lin.stl">
                <sha1>cd9a63feca16432364dcb3c5b0513a46452e0e1d</sha1>
             </file>
         </metric>
    </metrics>
</test>
