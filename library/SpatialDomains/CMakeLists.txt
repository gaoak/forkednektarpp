SET(SPATIAL_DOMAINS_SOURCES
Conditions.cpp
GeomFactors.cpp
Geometry0D.cpp
Geometry1D.cpp
Geometry2D.cpp
Geometry3D.cpp
Geometry.cpp
GeometryLocator.cpp
HexGeom.cpp
MeshGraph.cpp
MeshGraphIO.cpp
MeshGraphIOXml.cpp
MeshGraphIOXmlCompressed.cpp
MeshPartition.cpp
PrismGeom.cpp
PyrGeom.cpp
QuadGeom.cpp
RefRegion.cpp
RefRegionLine.cpp
RefRegionParallelogram.cpp
RefRegionCylinder.cpp
RefRegionSphere.cpp
SegGeom.cpp
PointGeom.cpp
TetGeom.cpp
TriGeom.cpp
Movement/Zones.cpp
Movement/Movement.cpp
)

SET(SPATIAL_DOMAINS_HEADERS
Conditions.h
Curve.hpp
GeomFactors.h
HOAlignment.h
Geometry0D.h
Geometry1D.h
Geometry2D.h
Geometry3D.h
Geometry.h
GeometryLocator.h
HexGeom.h
LinearMeshGraph.hpp
MeshEntities.hpp
MeshGraph.h
MeshGraphIO.h
MeshGraphIOXml.h
MeshGraphIOXmlCompressed.h
MeshPartition.h
PrismGeom.h
PyrGeom.h
QuadGeom.h
RefRegion.h
RefRegionLine.h
RefRegionParallelogram.h
RefRegionCylinder.h
RefRegionSphere.h
SegGeom.h
PointGeom.h
SpatialDomainsDeclspec.h
TetGeom.h
TriGeom.h
SpatialDomains.hpp
Movement/Zones.h
Movement/InterfaceInterpolation.h
Movement/Movement.h
)

IF(NEKTAR_USE_HDF5)
    SET(SPATIAL_DOMAINS_HEADERS ${SPATIAL_DOMAINS_HEADERS} MeshGraphIOHDF5.h)
    SET(SPATIAL_DOMAINS_SOURCES ${SPATIAL_DOMAINS_SOURCES} MeshGraphIOHDF5.cpp)
ENDIF()

IF(NEKTAR_USE_SCOTCH)
    SET(SPATIAL_DOMAINS_HEADERS ${SPATIAL_DOMAINS_HEADERS} MeshPartitionScotch.h)
    SET(SPATIAL_DOMAINS_SOURCES ${SPATIAL_DOMAINS_SOURCES} MeshPartitionScotch.cpp)

    IF (NEKTAR_USE_MPI)
        SET(SPATIAL_DOMAINS_HEADERS ${SPATIAL_DOMAINS_HEADERS} MeshPartitionPtScotch.h)
        SET(SPATIAL_DOMAINS_SOURCES ${SPATIAL_DOMAINS_SOURCES} MeshPartitionPtScotch.cpp)
    ENDIF()
ENDIF()

IF(NEKTAR_USE_METIS)
    SET(SPATIAL_DOMAINS_HEADERS ${SPATIAL_DOMAINS_HEADERS} MeshPartitionMetis.h)
    SET(SPATIAL_DOMAINS_SOURCES ${SPATIAL_DOMAINS_SOURCES} MeshPartitionMetis.cpp)
ENDIF()

SET(SPATIAL_DOMAINS_SOURCES ${SPATIAL_DOMAINS_SOURCES}
    CADSystem/CADAssociation.cpp
    CADSystem/CADSystem.cpp
    CADSystem/CADCurve.cpp
    CADSystem/CADSurf.cpp
    CADSystem/CADVert.cpp
)
SET(SPATIAL_DOMAINS_HEADERS ${SPATIAL_DOMAINS_HEADERS}
    CADSystem/CADAssociation.h
    CADSystem/CADSystem.h
    CADSystem/CADObject.h
    CADSystem/CADCurve.h
    CADSystem/CADSurf.h
    CADSystem/CADVert.h
)

IF(NEKTAR_USE_MESHGEN)
    SET(SPATIAL_DOMAINS_SOURCES ${SPATIAL_DOMAINS_SOURCES}
        CADSystem/OCE/CADSystemOCE.cpp
        CADSystem/OCE/CADVertOCE.cpp
        CADSystem/OCE/CADCurveOCE.cpp
        CADSystem/OCE/CADSurfOCE.cpp
        CADSystem/OCE/TransfiniteSurface.cpp
    )
    SET(SPATIAL_DOMAINS_HEADERS ${SPATIAL_DOMAINS_HEADERS}
        CADSystem/OCE/CADSystemOCE.h
        CADSystem/OCE/CADVertOCE.h
        CADSystem/OCE/CADCurveOCE.h
        CADSystem/OCE/CADSurfOCE.h
        CADSystem/OCE/GeoParser.hpp
        CADSystem/OCE/TransfiniteSurface.h
        CADSystem/OCE/OpenCascade.h
    )
ENDIF()

IF(NEKTAR_USE_CFI)
    SET(SPATIAL_DOMAINS_SOURCES ${SPATIAL_DOMAINS_SOURCES}
        CADSystem/CFI/CADSystemCFI.cpp
        CADSystem/CFI/CADVertCFI.cpp
        CADSystem/CFI/CADCurveCFI.cpp
        CADSystem/CFI/CADSurfCFI.cpp
    )
    SET(SPATIAL_DOMAINS_HEADERS ${SPATIAL_DOMAINS_HEADERS}
        CADSystem/CFI/CADSystemCFI.h
        CADSystem/CFI/CADVertCFI.h
        CADSystem/CFI/CADCurveCFI.h
        CADSystem/CFI/CADSurfCFI.h
        CADSystem/CFI/CADElementCFI.h
    )
ENDIF()

ADD_DEFINITIONS(-DSPATIAL_DOMAINS_EXPORTS)

ADD_NEKTAR_LIBRARY(SpatialDomains
    SOURCES ${SPATIAL_DOMAINS_SOURCES}
    HEADERS ${SPATIAL_DOMAINS_HEADERS}
    DEPENDS StdRegions
    SUMMARY "Nektar++ SpatialDomains library"
    DESCRIPTION "This library provides the mappings between reference regions and physical regions in the domain.")

IF(NEKTAR_USE_MESHGEN)
    TARGET_LINK_LIBRARIES(SpatialDomains LINK_PRIVATE ${OCC_LIBRARIES})
    ADD_DEPENDENCIES(SpatialDomains oce-0.18.3)
ENDIF()

IF (NEKTAR_USE_METIS)
    # Metis
    TARGET_LINK_LIBRARIES(SpatialDomains LINK_PRIVATE ${METIS_LIB})
    ADD_DEPENDENCIES(SpatialDomains metis-5.1.0)
    TARGET_LINK_LIBRARIES(SpatialDomains LINK_PRIVATE ${METIS_LIBRARY})
ENDIF()

IF (NEKTAR_USE_SCOTCH)
    IF (NEKTAR_USE_MPI)
        # If compiling Scotch, we are building a static library so we do
        # not want to propagate the dependency in the link interface
        IF (THIRDPARTY_BUILD_SCOTCH)
            TARGET_LINK_LIBRARIES(SpatialDomains LINK_PRIVATE
                ${PTSCOTCH_LIBRARY} ${PTSCOTCHERR_LIBRARY})
        ELSE()
            TARGET_LINK_LIBRARIES(SpatialDomains LINK_PUBLIC
                ${PTSCOTCH_LIBRARY} ${PTSCOTCHERR_LIBRARY})
        ENDIF()

        TARGET_LINK_LIBRARIES(SpatialDomains LINK_PRIVATE ${GSMPI_LIBRARY} ${XXT_LIBRARY})
    ELSE ()
        # Scotch
        IF (THIRDPARTY_BUILD_SCOTCH)
            TARGET_LINK_LIBRARIES(SpatialDomains LINK_PRIVATE
                ${SCOTCH_LIBRARY} ${SCOTCHERR_LIBRARY})
        ELSE()
            TARGET_LINK_LIBRARIES(SpatialDomains LINK_PUBLIC
                ${SCOTCH_LIBRARY} ${SCOTCHERR_LIBRARY})
        ENDIF()
    ENDIF()
    ADD_DEPENDENCIES(SpatialDomains scotch-7.0.1)
ENDIF()

IF (NEKTAR_BUILD_PYTHON)
    SUBDIRS(Python)
ENDIF()

INSTALL(DIRECTORY ./ DESTINATION ${NEKTAR_INCLUDE_DIR}/SpatialDomains
    COMPONENT dev FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")
