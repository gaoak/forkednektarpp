########################################################################
#
# install-nvpl-archive.cmake
#
# Copy the include/ and lib/ directories of an unpacked NVPL archive into
# the ThirdParty distribution directory. Not every NVPL archive has both
# (nvpl_common ships no headers), so each is copied only if present. FILE(COPY)
# keeps the versioned library symlinks as symlinks.
#
# Run with:
#   cmake -DSRC=<source dir> -DDST=<dist dir> -P install-nvpl-archive.cmake
#
########################################################################

FOREACH(dir include lib)
    IF(IS_DIRECTORY ${SRC}/${dir})
        FILE(COPY ${SRC}/${dir} DESTINATION ${DST})
    ENDIF()
ENDFOREACH()
