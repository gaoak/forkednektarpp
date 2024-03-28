MACRO(ADD_OPERATOR name)
    SET(NAME_HEADERS ${name}_HEADERS)
    SET(NAME_SOURCES ${name}_SOURCES)

    FILE(GLOB ${NAME_HEADERS}
        RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        *${name}.hpp
        ${name}/*.hpp
    )
    FILE(GLOB ${NAME_SOURCES}
        RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        *${name}.cpp
        ${name}/*.cpp
    )
    
    IF(NEKTAR_ENABLE_CUDA)
        SET(NAME_CUDA_HEADERS ${name}_CUDA_HEADERS)
        SET(NAME_CUDA_SOURCES ${name}_CUDA_SOURCES)

        FILE(GLOB ${NAME_CUDA_HEADERS}
            RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
            ${name}/*.cuh
        )
        FILE(GLOB ${NAME_CUDA_SOURCES}
            RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
            ${name}/*.cu
        )

        SET(${NAME_HEADERS}
            ${${NAME_HEADERS}}
            ${${NAME_CUDA_HEADERS}}
        )
        SET(${NAME_SOURCES}
            ${${NAME_SOURCES}}
            ${${NAME_CUDA_SOURCES}}
        )
    ENDIF()
    SET(OPERATORS_HEADERS
        ${OPERATORS_HEADERS}
        ${${NAME_HEADERS}}
    )
    SET(OPERATORS_SOURCES
        ${OPERATORS_SOURCES}
        ${${NAME_SOURCES}}
    )
ENDMACRO()


SET(OPERATORS_HEADERS
Operator.hpp
Field.hpp
MemoryRegionCPU.hpp
)

SET(OPERATORS_SOURCES
Operator.cpp
Field.cpp
)

ADD_OPERATOR(Identity)
ADD_OPERATOR(BwdTrans)
ADD_OPERATOR(IProductWRTBase)
ADD_OPERATOR(IProductWRTDerivBase)
ADD_OPERATOR(PhysDeriv)
ADD_OPERATOR(Mass)
ADD_OPERATOR(Helmholtz)
ADD_OPERATOR(ConjGrad)
ADD_OPERATOR(AssmbScatr)
ADD_OPERATOR(NullPrecon)
ADD_OPERATOR(DiagPrecon)
ADD_OPERATOR(FwdTrans)
ADD_OPERATOR(HelmSolve)
ADD_OPERATOR(Matrix)
ADD_OPERATOR(DirBndCond)
ADD_OPERATOR(NeuBndCond)
ADD_OPERATOR(RobBndCond)
ADD_OPERATOR(AddTraceIntegral)
ADD_OPERATOR(MultiplyByElmtInvMass)
#ADD_OPERATOR(Reduce)


if(NEKTAR_ENABLE_CUDA)
    SET(OPERATORS_SOURCES
        ${OPERATORS_SOURCES}
        MemoryRegionCUDA.cpp
    )
    SET(OPERATORS_HEADERS
        ${OPERATORS_HEADERS}
        MemoryRegionCUDA.hpp
    )
    SET(ADDITIONAL_OPERATOR_DEPS
    CUDA::cublas
    )
endif()

SOURCE_GROUP(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${OPERATORS_HEADERS} ${OPERATORS_SOURCES})

ADD_DEFINITIONS(-DOPERATORS_EXPORTS -I${CMAKE_CURRENT_SOURCE_DIR})

ADD_NEKTAR_LIBRARY(Operators
    SOURCES ${OPERATORS_SOURCES}
    HEADERS ${OPERATORS_HEADERS}
    DEPENDS MultiRegions
    SUMMARY "Nektar++ Operators library"
    DESCRIPTION "This library provides classes related to operators for Nektar++.")

INSTALL(DIRECTORY ./ DESTINATION ${NEKTAR_INCLUDE_DIR}/Operators COMPONENT dev FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp" PATTERN "*.cuh")

