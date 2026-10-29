// Conforming full fluid/solid mesh for the MHD student tutorial.
// Defaults: 4 fluid quads + 4 solid quads. Edit the three cell counts below.
L = 1;
H = 1;
h = 0.25;
nx = 2;
nyFluid = 2;
nyWall = 1;

Point(1) = {0, -h, 0, 1};
Point(2) = {L, -h, 0, 1};
Point(3) = {0, 0, 0, 1};
Point(4) = {L, 0, 0, 1};
Point(5) = {0, H, 0, 1};
Point(6) = {L, H, 0, 1};
Point(7) = {0, H+h, 0, 1};
Point(8) = {L, H+h, 0, 1};

Line(1) = {1, 2};
Line(2) = {2, 4};
Line(3) = {3, 4}; // Shared lower fluid/solid interface.
Line(4) = {1, 3};
Line(5) = {4, 6};
Line(6) = {5, 6}; // Shared upper fluid/solid interface.
Line(7) = {3, 5};
Line(8) = {6, 8};
Line(9) = {7, 8};
Line(10) = {5, 7};

Line Loop(1) = {1, 2, -3, -4};
Plane Surface(1) = {1};
Line Loop(2) = {3, 5, -6, -7};
Plane Surface(2) = {2};
Line Loop(3) = {6, 8, -9, -10};
Plane Surface(3) = {3};

// Gmsh counts points, so a line with nx cells needs nx+1 points.
// "Line" is the legacy spelling of "Curve", also accepted by Gmsh 2.x.
Transfinite Line {1, 3, 6, 9} = nx+1;
Transfinite Line {5, 7} = nyFluid+1;
Transfinite Line {2, 4, 8, 10} = nyWall+1;
Transfinite Surface {1, 2, 3};
Recombine Surface {1, 2, 3};

// Physical IDs become source Nektar++ composite IDs after conversion.
Physical Surface("fluid", 1) = {2};
Physical Surface("solid", 2) = {1, 3};
Physical Line("outer_bottom", 10) = {1};
Physical Line("outer_top", 11) = {9};
Physical Line("left", 12) = {4, 7, 10};
Physical Line("right", 13) = {2, 5, 8};
Physical Line("fluid_bottom", 14) = {3};
Physical Line("fluid_top", 15) = {6};

Mesh.ElementOrder = 1;
Mesh.MshFileVersion = 2.2;
Mesh.Binary = 0;
