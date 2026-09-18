<?xml version="1.0" encoding="utf-8" ?>
<test>
    <description>Unit test of the Python interface for mesh vertices.</description>
    <executable python="true">test_nekmesh_node.py</executable>
    <parameters>-v</parameters>
    <metrics>
        <metric type="pyunittest" id="1">
            <function>testMeshOwnsVertices</function>
            <function>testVertexAutomaticID</function>
            <function>testVertexCoordim</function>
            <function>testVertexCoordinates</function>
            <function>testVertexGlobalID</function>
        </metric>
    </metrics>
</test>
