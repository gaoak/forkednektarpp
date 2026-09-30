////////////////////////////////////////////////////////////////////////////////
//
//  File: CompressData.h
//
//  For more information, please see: http://www.nektar.info/
//
//  The MIT License
//
//  Copyright (c) 2006 Division of Applied Mathematics, Brown University (USA),
//  Department of Aeronautics, Imperial College London (UK), and Scientific
//  Computing and Imaging Institute, University of Utah (USA).
//
//  Permission is hereby granted, free of charge, to any person obtaining a
//  copy of this software and associated documentation files (the "Software"),
//  to deal in the Software without restriction, including without limitation
//  the rights to use, copy, modify, merge, publish, distribute, sublicense,
//  and/or sell copies of the Software, and to permit persons to whom the
//  Software is furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included
//  in all copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
//  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//  DEALINGS IN THE SOFTWARE.
//
//  Description: Routines for compressing and inflating data
//
////////////////////////////////////////////////////////////////////////////////

#ifndef NEKTAR_LIB_UTILITIES_BASIC_UTILS_COMPRESSDATA_H
#define NEKTAR_LIB_UTILITIES_BASIC_UTILS_COMPRESSDATA_H

#include <LibUtilities/BasicUtils/SessionReader.h>

#include "zlib.h"

// Buffer size for zlib compression/decompression
#define CHUNK 16384

namespace Nektar::LibUtilities
{

enum EndianType
{
    eEndianUnknown,
    eEndianBig,
    eEndianLittle,
    eEndianBigWord,   /* Middle-endian, Honeywell 316 style */
    eEndianLittleWord /* Middle-endian, PDP-11 style */
};

const std::string EndianTypeMap[] = {"UnknownEndian", "BigEndian",
                                     "LittleEndian", "BigWordEndian",
                                     "LittleWordEndian"};

LIB_UTILITIES_EXPORT EndianType Endianness(void);

namespace CompressData
{
LIB_UTILITIES_EXPORT std::string GetCompressString(void);
LIB_UTILITIES_EXPORT std::string GetBitSizeStr(void);

/**
 * @brief Deflate @p in into @p out.
 *
 * @p level is a zlib compression level. The default is what zlib calls default,
 * which is 6, but some applications (e.g. mesh generation) do not require
 * particularly high levels of compression.
 */
template <class T>
int ZlibEncode(std::vector<T> &in, std::string &out,
               int level = Z_DEFAULT_COMPRESSION)
{
    int ret;
    unsigned have;
    std::string buffer;
    buffer.resize(CHUNK);
    z_stream strm;
    unsigned char *input = (unsigned char *)(&in[0]);

    /* allocate deflate state */
    strm.zalloc = Z_NULL;
    strm.zfree  = Z_NULL;
    strm.opaque = Z_NULL;
    ret         = deflateInit(&strm, level);

    ASSERTL0(ret == Z_OK, "Error initializing Zlib.");

    strm.avail_in = (unsigned int)in.size() * sizeof(T) / sizeof(char);
    strm.next_in  = input;

    // Deflate input until output buffer is no longer full.
    do
    {
        strm.avail_out = CHUNK;
        strm.next_out  = (unsigned char *)(&buffer[0]);

        ret = deflate(&strm, Z_FINISH);

        // Deflate can return Z_OK, Z_STREAM_ERROR, Z_BUF_ERROR or
        // Z_STREAM_END. All, except Z_STREAM_ERROR are ok.
        ASSERTL0(ret != Z_STREAM_ERROR, "Zlib stream error");

        have = CHUNK - strm.avail_out;
        out.append(buffer, 0, have);
    } while (strm.avail_out == 0);

    // Check all input was processed.
    ASSERTL0(strm.avail_in == 0, "Not all input was used.");

    // Check stream is complete.
    ASSERTL0(ret == Z_STREAM_END, "Stream not finished");

    // Clean-up and return
    (void)deflateEnd(&strm);
    return Z_OK;
}

/**
 * Convert a string containing compressed binary (i.e. from
 * deflate) into a base 64 string
 */
LIB_UTILITIES_EXPORT void BinaryStrToBase64Str(
    std::string &compressedDataString, std::string &base64string);

/**
 * Compress a vector of NekDouble values into a base64 string.
 */
template <class T>
int ZlibEncodeToBase64Str(std::vector<T> &in, std::string &out64,
                          int level = Z_DEFAULT_COMPRESSION)
{
    std::string out;

    int ok = ZlibEncode(in, out, level);

    BinaryStrToBase64Str(out, out64);

    return ok;
}

/**
 * Decompress a zlib-compressed string into a vector of NekDouble
 * values.
 */
template <class T> int ZlibDecode(std::string &in, std::vector<T> &out)
{
    int ret;
    z_stream strm;

    strm.zalloc   = Z_NULL;
    strm.zfree    = Z_NULL;
    strm.opaque   = Z_NULL;
    strm.avail_in = 0;
    strm.next_in  = Z_NULL;
    ret           = inflateInit(&strm);
    ASSERTL0(ret == Z_OK, "Error initializing zlib decompression.");

    strm.avail_in = (unsigned int)in.size();
    strm.next_in  = (unsigned char *)(&in[0]);

    // Inflate straight into the destination, growing it geometrically. The
    // decompressed size is not known in advance, but going via an
    // intermediate string costs a temporary per chunk, a reallocation of the
    // whole block on every doubling, and one final copy of all of it.
    const size_t perElmt = sizeof(T);
    size_t nElmts        = CHUNK / perElmt > 1 ? CHUNK / perElmt : 1;
    size_t written       = 0;
    out.resize(nElmts);

    do
    {
        size_t capacity = out.size() * perElmt;
        if (capacity == written)
        {
            out.resize(out.size() * 2);
            capacity = out.size() * perElmt;
        }

        strm.avail_out = (unsigned int)(capacity - written);
        strm.next_out  = (unsigned char *)out.data() + written;

        ret = inflate(&strm, Z_NO_FLUSH);

        ASSERTL0(ret != Z_STREAM_ERROR, "Stream error occured.");

        switch (ret)
        {
            case Z_NEED_DICT:
                ret = Z_DATA_ERROR;
                /* Falls through. */
            case Z_DATA_ERROR:
            case Z_MEM_ERROR:
                (void)inflateEnd(&strm);
                out.clear();
                return ret;
        }

        written = capacity - strm.avail_out;
    } while (strm.avail_out == 0);

    (void)inflateEnd(&strm);

    if (ret == Z_STREAM_END)
    {
        // Any trailing bytes that do not make up a whole element are dropped,
        // as they were when this decoded via a string.
        out.resize(written / perElmt);
        return Z_OK;
    }
    else
    {
        out.clear();
        return Z_DATA_ERROR;
    }
}

/**
 * Convert a string containing base 64 (i.e. from xml file)
 * into a binary string
 */
LIB_UTILITIES_EXPORT void Base64StrToBinaryStr(
    std::string &base64string, std::string &compressedDataString);

/**
 * Decompress a base 64 compressed binary string into a vector
 * of NekDouble values.
 */
template <class T>
int ZlibDecodeFromBase64Str(std::string &in64, std::vector<T> &out)
{
    std::string in;
    Base64StrToBinaryStr(in64, in);

    return ZlibDecode(in, out);
}

} // namespace CompressData
} // namespace Nektar::LibUtilities
#endif
