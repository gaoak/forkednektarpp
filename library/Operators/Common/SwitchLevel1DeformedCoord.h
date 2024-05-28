///////////////////////////////////////////////////////////////////////////////
//
// File: SwitchLevel1DeformedCoord.h
//
// For more information, please see: http://www.nektar.info
//
// The MIT License
//
// Copyright (c) 2006 Division of Applied Mathematics, Brown University (USA),
// Department of Aeronautics, Imperial College London (UK), and Scientific
// Computing and Imaging Institute, University of Utah (USA).
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
//
// Description:  Operators which go with the Commone SwitchLevel1.h
// boost preprocessing
//
///////////////////////////////////////////////////////////////////////////////

#define NQ(i) BOOST_PP_TUPLE_ELEM(0, i)
#define NQ_M1(i) BOOST_PP_DEC(BOOST_PP_TUPLE_ELEM(0, i))

if (deformed)
{

    if (dimension == 1)
    {
#define OPERATOR1D_DEF operator1D<LibUtilities::eSegment, true>(inptr, outptr)

        switch (Coordim)
        {
            case 1:
#define OPERATOR1D(r, i)                                                       \
    case NQ(i):                                                                \
        operator1D<LibUtilities::eSegment, true, 1, NQ(i)>(inptr, outptr);     \
        break;

#include "../Common/SwitchLevel1_1D.h"
                break;
            case 2:
#undef OPERATOR1D
#define OPERATOR1D(r, i)                                                       \
    case NQ(i):                                                                \
        operator1D<LibUtilities::eSegment, true, 2, NQ(i)>(inptr, outptr);     \
        break;

#include "../Common/SwitchLevel1_1D.h"

                break;
            case 3:
#undef OPERATOR1D
#define OPERATOR1D(r, i)                                                       \
    case NQ(i):                                                                \
        operator1D<LibUtilities::eSegment, true, 3, NQ(i)>(inptr, outptr);     \
        break;

#include "../Common/SwitchLevel1_1D.h"
                break;
            default:
                NEKERROR(ErrorUtil::efatal, "invalid coordinate dimension");
                break;
        }
    }
    else if (dimension == 2)
    {
        switch (Coordim)
        {
            case 2:

#define OPERATOR2D_DEF_TRI                                                     \
    operator2D<LibUtilities::eTriangle, true>(inptr, outptr)

#define OPERATOR2D_TRI(r, i)                                                   \
    case NQ(i):                                                                \
        operator2D<LibUtilities::eTriangle, true, 2, NQ(i), NQ_M1(i)>(inptr,   \
                                                                      outptr); \
        break;

#define OPERATOR2D_DEF_QUAD                                                    \
    operator2D<LibUtilities::eQuadrilateral, true>(inptr, outptr)

#define OPERATOR2D_QUAD(r, i)                                                  \
    case NQ(i):                                                                \
        operator2D<LibUtilities::eQuadrilateral, true, 2, NQ(i), NQ(i)>(       \
            inptr, outptr);                                                    \
        break;

#include "../Common/SwitchLevel1_2D.h"
                break;
            case 3:
#undef OPERATOR2D_TRI
#define OPERATOR2D_TRI(r, i)                                                   \
    case NQ(i):                                                                \
        operator2D<LibUtilities::eTriangle, true, 3, NQ(i), NQ_M1(i)>(inptr,   \
                                                                      outptr); \
        break;

#undef OPERATOR2D_QUAD
#define OPERATOR2D_QUAD(r, i)                                                  \
    case NQ(i):                                                                \
        operator2D<LibUtilities::eQuadrilateral, true, 3, NQ(i), NQ(i)>(       \
            inptr, outptr);                                                    \
        break;

#include "../Common/SwitchLevel1_2D.h"
                break;
            default:
                NEKERROR(ErrorUtil::efatal, "invalid coordinate dimension");
                break;
        }
    }
    else if (dimension == 3)
    {

#define OPERATOR3D_DEF_HEX                                                     \
    operator3D<LibUtilities::eHexahedron, true>(inptr, outptr)

#define OPERATOR3D_HEX(r, i)                                                   \
    case NQ(i):                                                                \
        operator3D<LibUtilities::eHexahedron, true, NQ(i), NQ(i), NQ(i)>(      \
            inptr, outptr);                                                    \
        break;

#define OPERATOR3D_DEF_TET                                                     \
    operator3D<LibUtilities::eTetrahedron, true>(inptr, outptr)

#define OPERATOR3D_TET(r, i)                                                   \
    case NQ(i):                                                                \
        operator3D<LibUtilities::eTetrahedron, true, NQ(i), NQ_M1(i),          \
                   NQ_M1(i)>(inptr, outptr);                                   \
        break;

#define OPERATOR3D_DEF_PRISM                                                   \
    operator3D<LibUtilities::ePrism, true>(inptr, outptr)

#define OPERATOR3D_PRISM(r, i)                                                 \
    case NQ(i):                                                                \
        operator3D<LibUtilities::ePrism, true, NQ(i), NQ(i), NQ_M1(i)>(        \
            inptr, outptr);                                                    \
        break;

#define OPERATOR3D_DEF_PYR                                                     \
    operator3D<LibUtilities::ePyramid, true>(inptr, outptr)

#define OPERATOR3D_PYR(r, i)                                                   \
    case NQ(i):                                                                \
        operator3D<LibUtilities::ePyramid, true, NQ(i), NQ(i), NQ_M1(i)>(      \
            inptr, outptr);                                                    \
        break;

#include "../Common/SwitchLevel1_3D.h"
    }
}
else
{
    if (dimension == 1)
    {
#undef OPERATOR1D_DEF
#define OPERATOR1D_DEF operator1D<LibUtilities::eSegment, false>(inptr, outptr)

        switch (Coordim)
        {
            case 1:
#undef OPERATOR1D
#define OPERATOR1D(r, i)                                                       \
    case NQ(i):                                                                \
        operator1D<LibUtilities::eSegment, false, 1, NQ(i)>(inptr, outptr);    \
        break;

#include "../Common/SwitchLevel1_1D.h"

                break;
            case 2:
#undef OPERATOR1D
#define OPERATOR1D(r, i)                                                       \
    case NQ(i):                                                                \
        operator1D<LibUtilities::eSegment, false, 2, NQ(i)>(inptr, outptr);    \
        break;

#include "../Common/SwitchLevel1_1D.h"

                break;
            case 3:
#undef OPERATOR1D
#define OPERATOR1D(r, i)                                                       \
    case NQ(i):                                                                \
        operator1D<LibUtilities::eSegment, false, 3, NQ(i)>(inptr, outptr);    \
        break;

#include "../Common/SwitchLevel1_1D.h"
                break;
            default:
                NEKERROR(ErrorUtil::efatal, "invalid coordinate dimension");
                break;
        }
    }
    else if (dimension == 2)
    {
        switch (Coordim)
        {
            case 2:

#undef OPERATOR2D_DEF_TRI
#define OPERATOR2D_DEF_TRI                                                     \
    operator2D<LibUtilities::eTriangle, false>(inptr, outptr)

#undef OPERATOR2D_TRI
#define OPERATOR2D_TRI(r, i)                                                   \
    case NQ(i):                                                                \
        operator2D<LibUtilities::eTriangle, false, 2, NQ(i), NQ_M1(i)>(        \
            inptr, outptr);                                                    \
        break;

#undef OPERATOR2D_DEF_QUAD
#define OPERATOR2D_DEF_QUAD                                                    \
    operator2D<LibUtilities::eQuadrilateral, false>(inptr, outptr)

#undef OPERATOR2D_QUAD
#define OPERATOR2D_QUAD(r, i)                                                  \
    case NQ(i):                                                                \
        operator2D<LibUtilities::eQuadrilateral, false, 2, NQ(i), NQ(i)>(      \
            inptr, outptr);                                                    \
        break;

#include "../Common/SwitchLevel1_2D.h"

                break;
            case 3:
#undef OPERATOR2D_TRI
#define OPERATOR2D_TRI(r, i)                                                   \
    case NQ(i):                                                                \
        operator2D<LibUtilities::eTriangle, false, 3, NQ(i), NQ_M1(i)>(        \
            inptr, outptr);                                                    \
        break;

#undef OPERATOR2D_QUAD
#define OPERATOR2D_QUAD(r, i)                                                  \
    case NQ(i):                                                                \
        operator2D<LibUtilities::eQuadrilateral, false, 3, NQ(i), NQ(i)>(      \
            inptr, outptr);                                                    \
        break;

#include "../Common/SwitchLevel1_2D.h"
                break;
            default:
                NEKERROR(ErrorUtil::efatal, "invalid coordinate dimension");
                break;
        }
    }
    else if (dimension == 3)
    {

#undef OPERATOR3D_DEF_HEX
#define OPERATOR3D_DEF_HEX                                                     \
    operator3D<LibUtilities::eHexahedron, false>(inptr, outptr)

#undef OPERATOR3D_HEX
#define OPERATOR3D_HEX(r, i)                                                   \
    case NQ(i):                                                                \
        operator3D<LibUtilities::eHexahedron, false, NQ(i), NQ(i), NQ(i)>(     \
            inptr, outptr);                                                    \
        break;

#undef OPERATOR3D_DEF_TET
#define OPERATOR3D_DEF_TET                                                     \
    operator3D<LibUtilities::eTetrahedron, false>(inptr, outptr)

#undef OPERATOR3D_TET
#define OPERATOR3D_TET(r, i)                                                   \
    case NQ(i):                                                                \
        operator3D<LibUtilities::eTetrahedron, false, NQ(i), NQ_M1(i),         \
                   NQ_M1(i)>(inptr, outptr);                                   \
        break;

#undef OPERATOR3D_DEF_PRISM
#define OPERATOR3D_DEF_PRISM                                                   \
    operator3D<LibUtilities::ePrism, false>(inptr, outptr)

#undef OPERATOR3D_PRISM
#define OPERATOR3D_PRISM(r, i)                                                 \
    case NQ(i):                                                                \
        operator3D<LibUtilities::ePrism, false, NQ(i), NQ(i), NQ_M1(i)>(       \
            inptr, outptr);                                                    \
        break;

#undef OPERATOR3D_DEF_PYR
#define OPERATOR3D_DEF_PYR                                                     \
    operator3D<LibUtilities::ePyramid, false>(inptr, outptr)

#undef OPERATOR3D_PYR
#define OPERATOR3D_PYR(r, i)                                                   \
    case NQ(i):                                                                \
        operator3D<LibUtilities::ePyramid, false, NQ(i), NQ(i), NQ_M1(i)>(     \
            inptr, outptr);                                                    \
        break;

#include "../Common/SwitchLevel1_3D.h"
    }
}
