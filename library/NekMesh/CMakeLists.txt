SET(NEKMESH_SOURCES
    Module/Module.cpp
    Module/InputModules/InputGmsh.cpp
    Module/InputModules/InputNek.cpp
    Module/InputModules/InputNek5000.cpp
    Module/InputModules/InputNekpp.cpp
    Module/InputModules/InputPly.cpp
    Module/InputModules/InputSem.cpp
    Module/InputModules/InputSwan.cpp
    Module/InputModules/InputStarTec.cpp
    Module/OutputModules/OutputGmsh.cpp
    Module/OutputModules/OutputNekpp.cpp
    Module/OutputModules/OutputSTL.cpp
    Module/OutputModules/OutputStdOut.cpp
    Module/ProcessModules/ProcessBL.cpp
    Module/ProcessModules/ProcessCheckConnectivity.cpp
    Module/ProcessModules/ProcessCombine.cpp
    Module/ProcessModules/ProcessCurve.cpp
    Module/ProcessModules/ProcessCurvedEdges.cpp
    Module/ProcessModules/ProcessCyl.cpp
    Module/ProcessModules/ProcessDetectSurf.cpp
    Module/ProcessModules/ProcessExtractSurf.cpp
    Module/ProcessModules/ProcessJac.cpp
    Module/ProcessModules/ProcessLinear.cpp
    Module/ProcessModules/ProcessPerAlign.cpp
    Module/ProcessModules/ProcessScalar.cpp
    Module/ProcessModules/ProcessSpherigon.cpp
    Module/ProcessModules/ProcessTetSplit.cpp
    Module/ProcessModules/ProcessInsertSurface.cpp
    Module/ProcessModules/ProcessExtrude.cpp
    Module/ProcessModules/ProcessRevolve.cpp
    Module/ProcessModules/ProcessVarOpti/ProcessVarOpti.cpp
    Module/ProcessModules/ProcessVarOpti/PreProcessing.cpp
    Module/ProcessModules/ProcessVarOpti/NodeOpti.cpp
    Module/ProcessModules/ProcessVarOpti/ElUtil.cpp
    Module/ProcessModules/ProcessScale.cpp


    MeshElements/Mesh.cpp
    MeshElements/Element.cpp
    MeshElements/BooleanOperators.cpp
    MeshElements/Line.cpp
    MeshElements/Triangle.cpp
    MeshElements/Quadrilateral.cpp
    MeshElements/Tetrahedron.cpp
    MeshElements/Pyramid.cpp
    MeshElements/Prism.cpp
    MeshElements/Hexahedron.cpp
    CADSystem/ProcessLoadCAD.cpp
    )

SET(NEKMESH_HEADERS
    NekMeshDeclspec.h
    Module/Module.h
    Module/MacroElements.h
    Module/ModuleContext.h
    Module/SurfaceHints.h
    Module/InputModules/InputGmsh.h
    Module/InputModules/InputNek.h
    Module/InputModules/InputNek5000.h
    Module/InputModules/InputNekpp.h
    Module/InputModules/InputPly.h
    Module/InputModules/InputSem.h
    Module/InputModules/InputSwan.h
    Module/InputModules/InputStarTec.h
    Module/OutputModules/OutputGmsh.h
    Module/OutputModules/OutputNekpp.h
    Module/OutputModules/OutputSTL.h
    Module/OutputModules/OutputStdOut.h
    Module/ProcessModules/ProcessBL.h
    Module/ProcessModules/ProcessCheckConnectivity.h
    Module/ProcessModules/ProcessCombine.h
    Module/ProcessModules/ProcessCurve.h
    Module/ProcessModules/ProcessCurvedEdges.h
    Module/ProcessModules/ProcessCyl.h
    Module/ProcessModules/ProcessDetectSurf.h
    Module/ProcessModules/ProcessExtractSurf.h
    Module/ProcessModules/ProcessJac.h
    Module/ProcessModules/ProcessLinear.h
    Module/ProcessModules/ProcessPerAlign.h
    Module/ProcessModules/ProcessScalar.h
    Module/ProcessModules/ProcessSpherigon.h
    Module/ProcessModules/ProcessTetSplit.h
    Module/ProcessModules/ProcessInsertSurface.h
    Module/ProcessModules/ProcessExtrude.h
    Module/ProcessModules/ProcessRevolve.h
    Module/ProcessModules/ProcessVarOpti/ProcessVarOpti.h
    Module/ProcessModules/ProcessVarOpti/NodeOpti.h
    Module/ProcessModules/ProcessVarOpti/ElUtil.h
    Module/ProcessModules/ProcessScale.h

    MeshElements/Element.h
    MeshElements/Mesh.h
    MeshElements/HOAlignment.h
    MeshElements/ElementConfig.h
    CADSystem/ProcessLoadCAD.h
    )

IF(NEKTAR_USE_MESHGEN)
    SET(NEKMESH_SOURCES ${NEKMESH_SOURCES}
        SurfaceMeshing/CurveMesh.cpp
        Octree/Octant.cpp
        Octree/Octree.cpp
        Octree/ProcessLoadOctree.cpp
        SurfaceMeshing/SurfaceMesh.cpp
        SurfaceMeshing/HOSurfaceMesh.cpp
        SurfaceMeshing/FaceMesh.cpp
        SurfaceMeshing/OptimiseFunctions.cpp
        ExtLibInterface/TriangleInterface.cpp
        ExtLibInterface/TetGenInterface.cpp
        VolumeMeshing/VolumeMesh.cpp
        VolumeMeshing/TetMeshing/TetMesh.cpp
        VolumeMeshing/BLMeshing/BLMesh.cpp
        Optimisation/BGFS-B.cpp
        2DGenerator/2DGenerator.cpp
        Module/ProcessModules/ProcessVarOpti/NodeOptiCAD.cpp
        Module/ProcessModules/ProcessProjectCAD.cpp
        Module/InputModules/InputMCF.cpp
    )

    SET(NEKMESH_HEADERS ${NEKMESH_HEADERS}
        Octree/SourcePoint.hpp
        SurfaceMeshing/CurveMesh.h
        Octree/Octant.h
        Octree/Octree.h
        Octree/ProcessLoadOctree.h
        SurfaceMeshing/SurfaceMesh.h
        SurfaceMeshing/HOSurfaceMesh.h
        SurfaceMeshing/FaceMesh.h
        SurfaceMeshing/OptimiseFunctions.h
        ExtLibInterface/TriangleInterface.h
        ExtLibInterface/TetGenInterface.h
        VolumeMeshing/VolumeMesh.h
        VolumeMeshing/TetMeshing/TetMesh.h
        VolumeMeshing/BLMeshing/BLMesh.h
        Optimisation/BGFS-B.h
        Optimisation/OptimiseObj.h
        2DGenerator/2DGenerator.h
        Module/ProcessModules/ProcessVarOpti/NodeOptiCAD.h
        Module/ProcessModules/ProcessProjectCAD.h
        Module/InputModules/InputMCF.h
    )
ENDIF()

IF(NEKTAR_USE_CFI)
    # The CADfix CAD classes have moved to SpatialDomains and are ported, but
    # InputCADfix and OutputCADfix still build against the mesh representation
    # NekMesh used to carry its own: NodeSharedPtr, ElementSharedPtr,
    # Mesh::m_element and Mesh::m_expDim, none of which exist any more. They
    # would not compile, so refuse to configure rather than fail part way
    # through a build. The sources are kept as they stand, to be ported
    # against a CADfix licence to test with.
    MESSAGE(FATAL_ERROR
        "NEKTAR_USE_CFI is not currently supported: the NekMesh CADfix "
        "modules have not been ported to SpatialDomains geometry. Configure "
        "with -DNEKTAR_USE_CFI=OFF.")

    ADD_DEFINITIONS(-DNEKTAR_USE_CFI)
    SET(NEKMESH_HEADERS ${NEKMESH_HEADERS}
        Module/InputModules/InputCADfix.h
        Module/OutputModules/OutputCADfix.h
    )
    SET(NEKMESH_SOURCES ${NEKMESH_SOURCES}
        Module/InputModules/InputCADfix.cpp
        Module/OutputModules/OutputCADfix.cpp
    )
ENDIF()

IF (NEKTAR_USE_CCM)
    SET(NEKMESH_SOURCES ${NEKMESH_SOURCES} Module/InputModules/InputStar.cpp)
    SET(NEKMESH_HEADERS ${NEKMESH_HEADERS} Module/InputModules/InputStar.h)
ENDIF()

IF (NEKTAR_USE_CGNS)
    SET(NEKMESH_SOURCES ${NEKMESH_SOURCES} Module/InputModules/InputCGNS.cpp)
    SET(NEKMESH_HEADERS ${NEKMESH_HEADERS} Module/InputModules/InputCGNS.h)
ENDIF()

IF (NEKTAR_USE_VTK)
    SET(NEKMESH_SOURCES ${NEKMESH_SOURCES}
        Module/InputModules/InputVtk.cpp
        Module/OutputModules/OutputVtk.cpp)
    SET(NEKMESH_HEADERS ${NEKMESH_HEADERS}
        Module/InputModules/InputVtk.h
        Module/OutputModules/OutputVtk.h)
ENDIF()

ADD_DEFINITIONS(-DNEKMESH_EXPORTS)

ADD_NEKTAR_LIBRARY(libNekMesh
    SOURCES ${NEKMESH_SOURCES}
    HEADERS ${NEKMESH_HEADERS}
    DEPENDS LocalRegions SpatialDomains
    SUMMARY "Nektar++ NekMesh library"
    DESCRIPTION "This library provides mesh generation and routines for the NekMesh executable.")

SET_TARGET_PROPERTIES(libNekMesh PROPERTIES OUTPUT_NAME NekMesh)

IF(NEKTAR_USE_MESHGEN)
    TARGET_LINK_LIBRARIES(libNekMesh LINK_PRIVATE ${TETGEN_LIBRARY})
    TARGET_LINK_LIBRARIES(libNekMesh LINK_PRIVATE ${TRIANGLE_LIBRARY})
    TARGET_LINK_LIBRARIES(libNekMesh LINK_PRIVATE ${OCC_LIBRARIES})
    SET(NEKTAR_USE_THREAD_SAFETY ON CACHE BOOL "Guarantee thread safety for boost pthreads parallelisation in ProcessVarOpti." FORCE)

    ADD_DEPENDENCIES(libNekMesh oce-0.18.3 tetgen-1.5 triangle-1.6)

    IF(NEKTAR_USE_CFI)
	TARGET_LINK_LIBRARIES(libNekMesh LINK_PUBLIC ${CFI_LIBRARY_API})
	TARGET_LINK_LIBRARIES(libNekMesh LINK_PUBLIC ${CFI_LIBRARY_CXX})
    ENDIF()
ENDIF()

IF (NEKTAR_USE_CCM)
   MESSAGE(STATUS "CCMIO Lib: ${CCMIO_LIBRARY}")
   TARGET_LINK_LIBRARIES(libNekMesh LINK_PRIVATE ${CCMIO_LIBRARIES})
   ADD_DEPENDENCIES(libNekMesh libccmio-2.6.1)
ENDIF (NEKTAR_USE_CCM)

IF (NEKTAR_USE_CGNS)
   MESSAGE(STATUS "CGNS Lib: ${CGNS_LIBRARY}")
   TARGET_LINK_LIBRARIES(libNekMesh LINK_PRIVATE ${CGNS_LIBRARY})
ENDIF (NEKTAR_USE_CGNS)

IF (NEKTAR_USE_VTK)
    IF (THIRDPARTY_BUILD_VTK)
        TARGET_LINK_LIBRARIES(libNekMesh LINK_PRIVATE ${VTK_LIBRARIES})
        ADD_DEPENDENCIES(libNekMesh vtk-9.3.0)
    ELSE()
        IF (VTK_MAJOR_VERSION LESS 6)
            TARGET_LINK_LIBRARIES(libNekMesh LINK_PUBLIC vtkCommon vtkIO vtkGraphics)
        ELSEIF (VTK_MAJOR_VERSION  GREATER_EQUAL 9)
            TARGET_LINK_LIBRARIES(libNekMesh LINK_PUBLIC VTK::CommonCore VTK::IOLegacy VTK::IOXML)
            vtk_module_autoinit(
                TARGETS libNekMesh
                MODULES VTK::CommonCore VTK::IOLegacy VTK::IOXML)
        ELSE ()
                TARGET_LINK_LIBRARIES(libNekMesh LINK_PUBLIC vtkCommonCore vtkIOLegacy vtkIOXML)
        ENDIF ()
    ENDIF()
ENDIF (NEKTAR_USE_VTK)

IF (NEKTAR_BUILD_PYTHON)
    SUBDIRS(Python)
ENDIF()

INSTALL(DIRECTORY ./
        DESTINATION ${NEKTAR_INCLUDE_DIR}/NekMesh
        COMPONENT dev
        FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")
