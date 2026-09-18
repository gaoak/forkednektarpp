<?xml version="1.0" encoding="utf-8" ?>
<test>
    <description>Unit test of the Python interface for the Nektar::NekMesh::Mesh class.</description>
    <executable python="true">test_nekmesh_mesh.py</executable>
    <parameters>-v</parameters>
    <metrics>
        <metric type="pyunittest" id="1">
            <function>testMeshDimensions</function>
            <function>testMeshDimensionsAreTheGraphs</function>
            <function>testMeshElementTags</function>
            <function>testMeshElements</function>
            <function>testMeshElementsBadDimension</function>
            <function>testMeshVertices</function>
        </metric>
    </metrics>
</test>
