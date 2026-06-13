SUBDIRS(LibUtilities StdRegions SpatialDomains LocalRegions Collections MultiRegions MatrixFreeOps)

IF(NEKTAR_BUILD_PYTHON)
    SUBDIRS(Python)
ENDIF()

IF (NEKTAR_ENABLE_DEVICE_SUPPORT)
    SUBDIRS(Operators)
ENDIF()
