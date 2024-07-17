SUBDIRS(LibUtilities StdRegions SpatialDomains LocalRegions Collections MultiRegions MatrixFreeOps)

IF(NEKTAR_BUILD_PYTHON)
    SUBDIRS(Python)
ENDIF()

IF (NEKTAR_BUILD_REDESIGN)
    SUBDIRS(Operators)
ENDIF()
