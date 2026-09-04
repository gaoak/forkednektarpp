########################################################################
#
# flatten-source-dir.cmake
#
# Normalise an ExternalProject source directory extracted from an archive
# that has more than one top-level entry.
#
# ExternalProject strips the leading directory of a downloaded archive only
# when that archive holds exactly one top-level item. An archive packed on
# macOS carries an AppleDouble "._<name>" companion beside every entry,
# including the top-level directory, so it has two - and the whole tree is
# copied verbatim, leaving the sources one level below SRC where nothing
# afterwards can find them.
#
# Run with:
#   cmake -DSRC=<source dir> -P flatten-source-dir.cmake
#
# It is a no-op on a correctly stripped tree, so it is safe to run
# unconditionally and safe to run twice.
#
########################################################################

IF (NOT SRC)
    MESSAGE(FATAL_ERROR "flatten-source-dir: SRC must be set.")
ENDIF()

# Note throughout: FILE(GLOB) does not escape '.' when it turns a pattern
# into a regex, so a pattern like "._*" matches almost every name there is.
# AppleDouble entries have to be recognised by testing the name, never by
# globbing for them.
FUNCTION(LIST_ENTRIES DIR OUT_REAL OUT_APPLEDOUBLE)
    FILE(GLOB ENTRIES LIST_DIRECTORIES true "${DIR}/*" "${DIR}/.*")
    SET(REAL)
    SET(APPLEDOUBLE)
    FOREACH(ENTRY IN LISTS ENTRIES)
        GET_FILENAME_COMPONENT(NAME "${ENTRY}" NAME)
        IF (NAME STREQUAL "." OR NAME STREQUAL "..")
            CONTINUE()
        ELSEIF (NAME MATCHES "^\\._")
            LIST(APPEND APPLEDOUBLE "${ENTRY}")
        ELSE()
            LIST(APPEND REAL "${ENTRY}")
        ENDIF()
    ENDFOREACH()
    SET(${OUT_REAL} "${REAL}" PARENT_SCOPE)
    SET(${OUT_APPLEDOUBLE} "${APPLEDOUBLE}" PARENT_SCOPE)
ENDFUNCTION()

# Lift the sources up a level if the extract left them nested. Done before
# the AppleDouble sweep so that a tree left half-processed by an interrupted
# run is still recognisable on the next one.
IF (NOT EXISTS "${SRC}/CMakeLists.txt")
    LIST_ENTRIES("${SRC}" TOP_REAL TOP_APPLEDOUBLE)
    LIST(LENGTH TOP_REAL COUNT)
    IF (NOT COUNT EQUAL 1 OR NOT IS_DIRECTORY "${TOP_REAL}")
        MESSAGE(FATAL_ERROR
            "flatten-source-dir: ${SRC} has no CMakeLists.txt and does not "
            "hold exactly one subdirectory; cannot tell where the sources "
            "are.")
    ENDIF()

    GET_FILENAME_COMPONENT(PARENT "${SRC}" DIRECTORY)
    GET_FILENAME_COMPONENT(NAME "${SRC}" NAME)
    SET(STAGE "${PARENT}/${NAME}-flatten")

    # Move the nested tree aside, discard the wrapper, then put it back under
    # the name the caller expects. Renames only, so this stays cheap on a
    # large tree.
    FILE(REMOVE_RECURSE "${STAGE}")
    FILE(RENAME "${TOP_REAL}" "${STAGE}")
    FILE(REMOVE_RECURSE "${SRC}")
    FILE(RENAME "${STAGE}" "${SRC}")

    MESSAGE(STATUS "flatten-source-dir: lifted sources out of ${TOP_REAL}")
ENDIF()

# Drop the AppleDouble files. They are inert to a build that lists its
# sources, but they sit beside every real file and a source glob would pick
# them up. Walk explicitly rather than globbing, per the note above.
SET(PENDING "${SRC}")
SET(APPLEDOUBLE)
WHILE (PENDING)
    LIST(POP_FRONT PENDING DIR)
    LIST_ENTRIES("${DIR}" REAL FOUND)
    LIST(APPEND APPLEDOUBLE ${FOUND})
    FOREACH(ENTRY IN LISTS REAL)
        IF (IS_DIRECTORY "${ENTRY}" AND NOT IS_SYMLINK "${ENTRY}")
            LIST(APPEND PENDING "${ENTRY}")
        ENDIF()
    ENDFOREACH()
ENDWHILE()

IF (APPLEDOUBLE)
    FILE(REMOVE_RECURSE ${APPLEDOUBLE})
    LIST(LENGTH APPLEDOUBLE COUNT)
    MESSAGE(STATUS "flatten-source-dir: removed ${COUNT} AppleDouble entries")
ENDIF()
