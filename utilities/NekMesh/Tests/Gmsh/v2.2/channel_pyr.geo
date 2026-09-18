// Geometry that channel_pyr.msh was generated from, kept so that the mesh can
// be made again:
//
//   gmsh channel_pyr.geo -3 -order 4 -format msh22 -o channel_pyr.msh
//
// A unit cube whose faces are recombined into quadrilaterals. Gmsh caps each of
// them off with a pyramid and fills the middle with tetrahedra, which is the
// way to get high-order pyramids out of it -- a pyramid only appears where a
// quadrilateral face has to meet a tetrahedral region.
//
// The mesh this produces replaced an earlier one whose interior nodes were
// ordered the way Nektar++ numbers a nodal pyramid rather than the way gmsh
// writes one. The two differ, and reading a gmsh file has to follow gmsh, so
// the mesh is taken from gmsh itself rather than written by hand.

Point(1) = {0, 0, 0, 2};
Point(2) = {1, 0, 0, 2};
Point(3) = {1, 1, 0, 2};
Point(4) = {0, 1, 0, 2};
Point(5) = {0, 0, 1, 2};
Point(6) = {1, 0, 1, 2};
Point(7) = {1, 1, 1, 2};
Point(8) = {0, 1, 1, 2};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
Line(5) = {5, 6};
Line(6) = {6, 7};
Line(7) = {7, 8};
Line(8) = {8, 5};
Line(9) = {1, 5};
Line(10) = {2, 6};
Line(11) = {3, 7};
Line(12) = {4, 8};

Curve Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};
Curve Loop(2) = {5, 6, 7, 8};
Plane Surface(2) = {2};
Curve Loop(3) = {1, 10, -5, -9};
Plane Surface(3) = {3};
Curve Loop(4) = {2, 11, -6, -10};
Plane Surface(4) = {4};
Curve Loop(5) = {3, 12, -7, -11};
Plane Surface(5) = {5};
Curve Loop(6) = {4, 9, -8, -12};
Plane Surface(6) = {6};

Surface Loop(1) = {1, 2, 3, 4, 5, 6};
Volume(1) = {1};

Transfinite Curve {1:12} = 2;
Transfinite Surface {1:6};
Recombine Surface {1:6};

Physical Volume(1) = {1};
