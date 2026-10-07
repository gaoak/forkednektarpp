# Don't use macro -- this way we can just link against NekMesh and
# LocalRegions.
ADD_UTILITIES_EXECUTABLE(NekMesh COMPONENT nekmesh
    SOURCES NekMesh.cpp
    DEPENDS libNekMesh)

# Nektar++
ADD_NEKTAR_TEST(Nektar++/3d_sphere)
ADD_NEKTAR_TEST(Nektar++/bl_quad)
ADD_NEKTAR_TEST(Nektar++/bl_tube)
ADD_NEKTAR_TEST(Nektar++/extract_curved_edge)
ADD_NEKTAR_TEST(Nektar++/extract_curved_face)
ADD_NEKTAR_TEST(Nektar++/extract_detectbnd_tube)
ADD_NEKTAR_TEST(Nektar++/extrude)
ADD_NEKTAR_TEST(Nektar++/revolve)
ADD_NEKTAR_TEST(Nektar++/jac_list_tet_face)
ADD_NEKTAR_TEST(Nektar++/linearise_invalid_quad)
ADD_NEKTAR_TEST(Nektar++/linearise_invalid_tet)
ADD_NEKTAR_TEST(Nektar++/scaleHO)
ADD_NEKTAR_TEST(Nektar++/prismreorder_tube)

IF (NOT WIN32 AND NEKTAR_SOLVER_INCNAVIERSTOKES)
    ADD_NEKTAR_TEST(Nektar++/peralign_double_periodic_hybrid)
ENDIF ()

# MeshGen test
IF(NEKTAR_USE_MESHGEN)
    ADD_NEKTAR_TEST(MeshGen/GEO/2d_adapt)
    ADD_NEKTAR_TEST(MeshGen/GEO/2d_bl)
    ADD_NEKTAR_TEST(MeshGen/GEO/2d_t106c)
    ADD_NEKTAR_TEST(MeshGen/GEO/annulus)

    ADD_NEKTAR_TEST(MeshGen/NACA/2d_bl)

    ADD_NEKTAR_TEST(MeshGen/STEP/2d_bl_aerofoil)
    ADD_NEKTAR_TEST(MeshGen/STEP/2d_bl_aerofoil_determinism)
    ADD_NEKTAR_TEST(MeshGen/STEP/2d_bl_t106)
    ADD_NEKTAR_TEST(MeshGen/STEP/2d_circle_square)
    ADD_NEKTAR_TEST(MeshGen/STEP/2d_circle_square-radapt)
    ADD_NEKTAR_TEST(MeshGen/STEP/varopti_slide)
    ADD_NEKTAR_TEST(MeshGen/STEP/varopti_bl_cyl)
    ADD_NEKTAR_TEST(MeshGen/STEP/3d_bl_cyl)
    ADD_NEKTAR_TEST(MeshGen/STEP/3d_bl_cyl_determinism)
    ADD_NEKTAR_TEST(MeshGen/STEP/3d_bl_wing LENGTHY)
    ADD_NEKTAR_TEST(MeshGen/STEP/3d_sphere LENGTHY)
    
    # IndustrialPipeline tests
    ADD_NEKTAR_TEST(IndustrialPipeline/00_projectcad_shaft)
    ADD_NEKTAR_TEST(IndustrialPipeline/01_projectcad_shaftHO)
    ADD_NEKTAR_TEST(IndustrialPipeline/02_projectcad_3d_sphere_star)
    ADD_NEKTAR_TEST(IndustrialPipeline/03_projectcad_NACA0012)
    ADD_NEKTAR_TEST(IndustrialPipeline/04_bl_ShaftHO)
    ADD_NEKTAR_TEST(IndustrialPipeline/05_bl_NACA0012)

    IF(NEKTAR_USE_CCM)
        ADD_NEKTAR_TEST(MeshGen/StarCCM/projectcad_ahmed)
        ADD_NEKTAR_TEST(MeshGen/StarCCM/projectcad_pyramids)
    ENDIF()
ENDIF()

# Nektar tests
ADD_NEKTAR_TEST(Nektar/bl_hex_lin)
ADD_NEKTAR_TEST(Nektar/gz_uk)
ADD_NEKTAR_TEST(Nektar/tetsplit_tube)

# Gmsh tests
ADD_NEKTAR_TEST(Gmsh/v2.2/cube_all)
ADD_NEKTAR_TEST(Gmsh/v2.2/cube_hex_lin)
ADD_NEKTAR_TEST(Gmsh/v2.2/cube_hex)
ADD_NEKTAR_TEST(Gmsh/v2.2/cube_prism_lin)
ADD_NEKTAR_TEST(Gmsh/v2.2/cube_prism)
ADD_NEKTAR_TEST(Gmsh/v2.2/channel_pyr)
ADD_NEKTAR_TEST(Gmsh/v2.2/channel_pyr_ho_volume)
ADD_NEKTAR_TEST(Gmsh/v2.2/cube_tet_lin LENGTHY)
ADD_NEKTAR_TEST(Gmsh/v2.2/cube_tet)
ADD_NEKTAR_TEST(Gmsh/v2.2/square_tri_lin)
ADD_NEKTAR_TEST(Gmsh/v2.2/curve_square)
ADD_NEKTAR_TEST(Gmsh/v2.2/cube_tet_lin_stl)
ADD_NEKTAR_TEST(Gmsh/v2.2/cube_tet_lin_determinism)
ADD_NEKTAR_TEST(Gmsh/v2.2/cube_hex_ho_volume)
ADD_NEKTAR_TEST(Gmsh/v2.2/scalar_cube)
ADD_NEKTAR_TEST(Gmsh/v2.2/detectsurf_square)
ADD_NEKTAR_TEST(Gmsh/v2.2/insertsurface_square)
ADD_NEKTAR_TEST(Gmsh/v2.2/square_tri)
ADD_NEKTAR_TEST(Gmsh/v2.2/peralign_3d_cylinder)
ADD_NEKTAR_TEST(Gmsh/v2.2/identify_boundary_tet)

ADD_NEKTAR_TEST(Gmsh/v4.0/cube_all)
ADD_NEKTAR_TEST(Gmsh/v4.0/cube_hex_lin)
ADD_NEKTAR_TEST(Gmsh/v4.0/cube_hex)
ADD_NEKTAR_TEST(Gmsh/v4.0/cube_prism_lin)
ADD_NEKTAR_TEST(Gmsh/v4.0/cube_prism)
ADD_NEKTAR_TEST(Gmsh/v4.0/cube_tet_lin LENGTHY)
ADD_NEKTAR_TEST(Gmsh/v4.0/cube_tet)
ADD_NEKTAR_TEST(Gmsh/v4.0/extract_square_circ)
ADD_NEKTAR_TEST(Gmsh/v4.0/square_tri_lin)
ADD_NEKTAR_TEST(Gmsh/v4.0/square_tri)

ADD_NEKTAR_TEST(Gmsh/v4.1/cube_all)
ADD_NEKTAR_TEST(Gmsh/v4.1/cube_all_p6)
ADD_NEKTAR_TEST(Gmsh/v4.1/cube_all_p6_ho_volume)
ADD_NEKTAR_TEST(Gmsh/v4.1/cube_hex_lin)
ADD_NEKTAR_TEST(Gmsh/v4.1/cube_hex)
ADD_NEKTAR_TEST(Gmsh/v4.1/cube_prism_lin)
ADD_NEKTAR_TEST(Gmsh/v4.1/cube_prism)
ADD_NEKTAR_TEST(Gmsh/v4.1/cube_tet_lin LENGTHY)
ADD_NEKTAR_TEST(Gmsh/v4.1/cube_tet)
ADD_NEKTAR_TEST(Gmsh/v4.1/extract_square_circ)
ADD_NEKTAR_TEST(Gmsh/v4.1/square_quad_lin)
ADD_NEKTAR_TEST(Gmsh/v4.1/square_quad)
ADD_NEKTAR_TEST(Gmsh/v4.1/square_tri_lin)
ADD_NEKTAR_TEST(Gmsh/v4.1/square_tri)

# CGNS tests
IF(NEKTAR_USE_CGNS)
    ADD_NEKTAR_TEST(CGNS/HiOrderMesh_tet_hex_pyr_order4_midvol)
    ADD_NEKTAR_TEST(CGNS/ImpossiblePyramids_linear)
    ADD_NEKTAR_TEST(CGNS/ImpossiblePyramids_order4_midface)
    ADD_NEKTAR_TEST(CGNS/linear_tet_prism_pyra_mesh)
    ADD_NEKTAR_TEST(CGNS/linear_tet_pyra_hex_mesh)
    ADD_NEKTAR_TEST(CGNS/twoprisms_3rd_order)
    ADD_NEKTAR_TEST(CGNS/CGNS_HiOrder_elements/HEXA_32)
    ADD_NEKTAR_TEST(CGNS/CGNS_HiOrder_elements/HEXA_64)
    ADD_NEKTAR_TEST(CGNS/CGNS_HiOrder_elements/HEXA_98)
    ADD_NEKTAR_TEST(CGNS/CGNS_HiOrder_elements/HEXA_125)
    ADD_NEKTAR_TEST(CGNS/CGNS_HiOrder_elements/PENTA_15)
    ADD_NEKTAR_TEST(CGNS/CGNS_HiOrder_elements/PENTA_18)
    ADD_NEKTAR_TEST(CGNS/CGNS_HiOrder_elements/PENTA_24)
    ADD_NEKTAR_TEST(CGNS/CGNS_HiOrder_elements/PENTA_33)
    ADD_NEKTAR_TEST(CGNS/CGNS_HiOrder_elements/PENTA_38)
    ADD_NEKTAR_TEST(CGNS/CGNS_HiOrder_elements/PENTA_40)
    ADD_NEKTAR_TEST(CGNS/CGNS_HiOrder_elements/PENTA_66)
    ADD_NEKTAR_TEST(CGNS/CGNS_HiOrder_elements/PENTA_75)
    ADD_NEKTAR_TEST(CGNS/CGNS_HiOrder_elements/PYRA_13)
    ADD_NEKTAR_TEST(CGNS/CGNS_HiOrder_elements/PYRA_29)
    ADD_NEKTAR_TEST(CGNS/CGNS_HiOrder_elements/PYRA_55)
    ADD_NEKTAR_TEST(CGNS/CGNS_HiOrder_elements/TETRA_10)
    ADD_NEKTAR_TEST(CGNS/CGNS_HiOrder_elements/TETRA_20)
    ADD_NEKTAR_TEST(CGNS/CGNS_HiOrder_elements/TETRA_35)
ENDIF()

# Rot peralign tests
ADD_NEKTAR_TEST(Gmsh/v2.2/peralign_rot_cyl)
ADD_NEKTAR_TEST(Gmsh/v2.2/cyl_arc)

# VarOpti tests
ADD_NEKTAR_TEST(Gmsh/v2.2/varopti_cube_hex)
ADD_NEKTAR_TEST(Gmsh/v2.2/varopti_cube_hex_ho)
ADD_NEKTAR_TEST(Gmsh/v2.2/varopti_cube_prism)
ADD_NEKTAR_TEST(Gmsh/v2.2/varopti_cube_sphere)
ADD_NEKTAR_TEST(Gmsh/v2.2/varopti_cube_sphere_distortion)
ADD_NEKTAR_TEST(Gmsh/v2.2/varopti_cube_sphere_linearelastic)
ADD_NEKTAR_TEST(Gmsh/v2.2/varopti_cube_sphere_threads)
ADD_NEKTAR_TEST(Gmsh/v2.2/varopti_naca)
ADD_NEKTAR_TEST(Gmsh/v2.2/varopti_naca_overint0)
ADD_NEKTAR_TEST(Gmsh/v2.2/varopti_naca_distortion)
ADD_NEKTAR_TEST(Gmsh/v2.2/varopti_naca_winslow)
ADD_NEKTAR_TEST(Nektar++/varopti_untangle_quad)
ADD_NEKTAR_TEST(Nektar++/varopti_untangle_quad_distortion)
ADD_NEKTAR_TEST(Nektar++/varopti_untangle_quad_winslow)
ADD_NEKTAR_TEST(Nektar++/varopti_untangle_quad_hyperelastic)

# HDF5 tests
IF (NEKTAR_USE_HDF5)
    ADD_NEKTAR_TEST(Gmsh/v4.0/hdf5_cube_tet)
ENDIF()

# StarCCM tests
IF (NEKTAR_USE_CCM)
    # Reading the binary Star-CCM+ format on its own. The projectcad tests
    # above also read it, but need mesh generation, so without these the
    # reader has no coverage at all.
    ADD_NEKTAR_TEST(MeshGen/StarCCM/read_ahmed)
    ADD_NEKTAR_TEST(MeshGen/StarCCM/read_pyramids)
ENDIF()

IF (NEKTAR_USE_VTK)
    ADD_NEKTAR_TEST(Vtk/tri_square)
    ADD_NEKTAR_TEST(Vtk/tri_square_out)
ENDIF()

ADD_NEKTAR_TEST(Ply/spherigon_sphere)
ADD_NEKTAR_TEST(Semtex/semtest)
ADD_NEKTAR_TEST(StarTec/peralign_bl_cube)
ADD_NEKTAR_TEST(StarTec/spherigon_straight_rw)




# Nek5000 tests
ADD_NEKTAR_TEST(Nek5000/annulus)
ADD_NEKTAR_TEST(Nek5000/box)
