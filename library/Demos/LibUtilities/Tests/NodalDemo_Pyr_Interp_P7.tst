<?xml version="1.0" encoding="utf-8" ?>
<test>
    <description>Nodal pyramid interpolation, evenly spaced points, P = 7</description>
    <executable>NodalDemo</executable>
    <parameters>--order 7 --type 36 --interp -0.3,-0.636,-0.9</parameters>
    <metrics>
        <metric type="Linf" id="1">
            <value tolerance="1e-12">4.40947e-08</value>
        </metric>
    </metrics>
</test>
