###############################################################################
##
## File: CheckCGNSFixture.cmake
##
## For more information, please see: http://www.nektar.info
##
## The MIT License
##
## Copyright (c) 2006 Division of Applied Mathematics, Brown University (USA),
## Department of Aeronautics, Imperial College London (UK), and Scientific
## Computing and Imaging Institute, University of Utah (USA).
##
## Permission is hereby granted, free of charge, to any person obtaining a
## copy of this software and associated documentation files (the "Software"),
## to deal in the Software without restriction, including without limitation
## the rights to use, copy, modify, merge, publish, distribute, sublicense,
## and/or sell copies of the Software, and to permit persons to whom the
## Software is furnished to do so, subject to the following conditions:
##
## The above copyright notice and this permission notice shall be included
## in all copies or substantial portions of the Software.
##
## THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
## OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
## FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
## THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
## LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
## FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
## DEALINGS IN THE SOFTWARE.
##
## Description: Regenerate a committed CGNS test fixture and check that the
## generator still produces it.
##
###############################################################################

# Run with cmake -P. Expects:
#
#   GENERATOR  the generator executable
#   GENARGS    its arguments, the file names among them being bare names
#   COPY       files to copy from SRCDIR into the work directory first, for a
#              generator that reads or edits an existing fixture; an entry of
#              the form src=dst is renamed on the way in, which is what a
#              generator that edits its input in place needs
#   EXPECT     the fixture names the run should produce
#   SRCDIR     where the committed fixtures live
#   WORKDIR    a scratch directory, emptied first
#   CGNSDIFF   the cgnsdiff executable
#
# The comparison is cgnsdiff rather than a byte compare: CGNS writes through
# HDF5, whose object headers carry the time each node was written, so two runs
# of the same generator differ in a couple of thousand bytes while describing
# the same mesh. cgnsdiff -d compares the tree and the node data and says
# nothing when they agree -- it reports differences on stdout but exits 0
# either way, so it is the output that is checked here.

file(REMOVE_RECURSE "${WORKDIR}")
file(MAKE_DIRECTORY "${WORKDIR}")

foreach(f ${COPY})
    if (f MATCHES "^(.+)=(.+)$")
        configure_file("${SRCDIR}/${CMAKE_MATCH_1}" "${WORKDIR}/${CMAKE_MATCH_2}"
                       COPYONLY)
    else()
        file(COPY "${SRCDIR}/${f}" DESTINATION "${WORKDIR}")
    endif()
endforeach()

execute_process(
    COMMAND "${GENERATOR}" ${GENARGS}
    WORKING_DIRECTORY "${WORKDIR}"
    RESULT_VARIABLE genResult
    OUTPUT_VARIABLE genOutput
    ERROR_VARIABLE genOutput)

if (NOT genResult EQUAL 0)
    message(FATAL_ERROR
        "Generator failed (${genResult}):\n${genOutput}")
endif()

foreach(f ${EXPECT})
    if (NOT EXISTS "${WORKDIR}/${f}")
        message(FATAL_ERROR "The generator did not write ${f}.")
    endif()

    execute_process(
        COMMAND "${CGNSDIFF}" -d "${SRCDIR}/${f}" "${WORKDIR}/${f}"
        RESULT_VARIABLE diffResult
        OUTPUT_VARIABLE diffOutput
        ERROR_VARIABLE diffOutput)

    if (NOT diffResult EQUAL 0)
        message(FATAL_ERROR "cgnsdiff failed on ${f}:\n${diffOutput}")
    endif()

    if (NOT diffOutput STREQUAL "")
        message(FATAL_ERROR
            "${f} is not what its generator now produces. Either the "
            "generator changed and the fixture was not regenerated, or the "
            "fixture was edited by hand. cgnsdiff reports:\n${diffOutput}")
    endif()
endforeach()
