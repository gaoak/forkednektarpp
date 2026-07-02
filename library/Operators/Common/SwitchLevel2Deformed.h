///////////////////////////////////////////////////////////////////////////////
//
// File: SwitchLevel2Deformed.h
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
// Description: BwdTrans operators which go with the Commone SwitchLevel2.h
// boost preprocessing
//
///////////////////////////////////////////////////////////////////////////////

/* Switch macros for two level switch over modes and then quadrature
   orders wrapped around a Deformed check */

if (deformed)
{
    // clang-format off
#define OPERATOR1D_DEF                                                         \
    operator1D<LibUtilities::eSegment, true>(inblock, outblock)

#define OPERATOR1D_Q(r, i)                                                     \
    case NQ(i):                                                                \
        operator1D<LibUtilities::eSegment, true, NM(i), NQ(i)>(inblock,        \
                                                               outblock);      \
        break;

#define OPERATOR1D_M(r, i)                                                     \
    case NM(i):                                                                \
        switch (nq0)                                                           \
        {                                                                      \
            BOOST_PP_FOR_##r((NM(i), NM(i), BOOST_PP_MUL(2, NM(i))),           \
                             LEV2TEST1, LEV2UPDATE1, OPERATOR1D_Q)             \
            default :                                                          \
                OPERATOR1D_DEF;                                                \
                break;                                                         \
        }                                                                      \
        break;

#define OPERATOR2D_DEF_TRI                                                     \
    operator2D<LibUtilities::eTriangle, true>(inblock, outblock)

#define OPERATOR2D_Q_TRI(r, i)                                                 \
    case NQ(i):                                                                \
        operator2D<LibUtilities::eTriangle, true, NM(i), NM(i), NQ(i),         \
                   NQ_M1(i)>(inblock, outblock);                               \
        break;

#define OPERATOR2D_M_TRI(r, i)                                                 \
    case NM(i):                                                                \
        switch (nq0)                                                           \
        {                                                                      \
            BOOST_PP_FOR_##r((NM(i), NM_P1(i), BOOST_PP_MUL(2, NM(i))),        \
                             LEV2TEST1, LEV2UPDATE1, OPERATOR2D_Q_TRI)         \
            default :                                                          \
                OPERATOR2D_DEF_TRI;                                            \
                break;                                                         \
        }                                                                      \
        break;

#define OPERATOR2D_DEF_QUAD                                                    \
    operator2D<LibUtilities::eQuadrilateral, true>(inblock, outblock)

#define OPERATOR2D_Q_QUAD(r, i)                                                \
    case NQ(i):                                                                \
        operator2D<LibUtilities::eQuadrilateral, true, NM(i), NM(i), NQ(i),    \
                   NQ(i)>(inblock, outblock);                                  \
        break;

#define OPERATOR2D_M_QUAD(r, i)                                                \
    case NM(i):                                                                \
        switch (nq0)                                                           \
        {                                                                      \
            BOOST_PP_FOR_##r((NM(i), NM(i), BOOST_PP_MUL(2, NM(i))),           \
                             LEV2TEST1, LEV2UPDATE1,                           \
                             OPERATOR2D_Q_QUAD)                                \
            default :                                                          \
                OPERATOR2D_DEF_QUAD;                                           \
                break;                                                         \
        }                                                                      \
        break;

#define OPERATOR3D_DEF_HEX                                                     \
    operator3D<LibUtilities::eHexahedron, true>(inblock, outblock)

#define OPERATOR3D_Q_HEX(r, i)                                                 \
    case NQ(i):                                                                \
        operator3D<LibUtilities::eHexahedron, true, NM(i), NM(i), NM(i),       \
                   NQ(i), NQ(i), NQ(i)>(inblock, outblock);                    \
        break;

#define OPERATOR3D_M_HEX(r, i)                                                 \
    case NM(i):                                                                \
        switch (nq0)                                                           \
        {                                                                      \
            BOOST_PP_FOR_##r((NM(i), NM(i), BOOST_PP_MUL(2, NM(i))),           \
                             LEV2TEST1, LEV2UPDATE1, OPERATOR3D_Q_HEX)         \
            default :                                                          \
                OPERATOR3D_DEF_HEX;                                            \
                break;                                                         \
        }                                                                      \
        break;

#define OPERATOR3D_DEF_TET                                                     \
    operator3D<LibUtilities::eTetrahedron, true>(inblock, outblock)

#define OPERATOR3D_Q_TET(r, i)                                                 \
    case NQ(i):                                                                \
        operator3D<LibUtilities::eTetrahedron, true, NM(i), NM(i), NM(i),      \
                   NQ(i), NQ_M1(i), NQ_M1(i)>(inblock, outblock);              \
        break;

#define OPERATOR3D_M_TET(r, i)                                                 \
    case NM(i):                                                                \
        switch (nq0)                                                           \
        {                                                                      \
            BOOST_PP_FOR_##r((NM(i), NM_P1(i), BOOST_PP_MUL(2, NM(i))),        \
                             LEV2TEST1, LEV2UPDATE1, OPERATOR3D_Q_TET)         \
            default :                                                          \
                OPERATOR3D_DEF_TET;                                            \
                break;                                                         \
        }                                                                      \
        break;

#define OPERATOR3D_DEF_PRISM                                                   \
    operator3D<LibUtilities::ePrism, true>(inblock, outblock)

#define OPERATOR3D_Q_PRISM(r, i)                                               \
    case NQ(i):                                                                \
        operator3D<LibUtilities::ePrism, true, NM(i), NM(i), NM(i), NQ(i),     \
                   NQ(i), NQ_M1(i)>(inblock, outblock);                        \
        break;

#define OPERATOR3D_M_PRISM(r, i)                                               \
    case NM(i):                                                                \
        switch (nq0)                                                           \
        {                                                                      \
            BOOST_PP_FOR_##r((NM(i), NM_P1(i), BOOST_PP_MUL(2, NM(i))),        \
                             LEV2TEST1, LEV2UPDATE1,                           \
                             OPERATOR3D_Q_PRISM)                               \
            default :                                                          \
                OPERATOR3D_DEF_PRISM;                                          \
                break;                                                         \
        }                                                                      \
        break;

#define OPERATOR3D_DEF_PYR                                                     \
    operator3D<LibUtilities::ePyramid, true>(inblock, outblock)

#define OPERATOR3D_Q_PYR(r, i)                                                 \
    case NQ(i):                                                                \
        operator3D<LibUtilities::ePyramid, true, NM(i), NM(i), NM(i), NQ(i),   \
                   NQ(i), NQ_M1(i)>(inblock, outblock);                        \
        break;

#define OPERATOR3D_M_PYR(r, i)                                                 \
    case NM(i):                                                                \
        switch (nq0)                                                           \
        {                                                                      \
            BOOST_PP_FOR_##r((NM(i), NM_P1(i), BOOST_PP_MUL(2, NM(i))),        \
                             LEV2TEST1, LEV2UPDATE1, OPERATOR3D_Q_PYR)         \
            default :                                                          \
                OPERATOR3D_DEF_PYR;                                            \
                break;                                                         \
        }                                                                      \
        break;

#include "../Common/SwitchLevel2.h"
}
else
{
#undef OPERATOR1D_DEF
#undef OPERATOR1D_Q
#undef OPERATOR1D_M

#define OPERATOR1D_DEF                                                         \
    operator1D<LibUtilities::eSegment, false>(inblock, outblock)

#define OPERATOR1D_Q(r, i)                                                     \
    case NQ(i):                                                                \
        operator1D<LibUtilities::eSegment, false, NM(i), NQ(i)>(inblock,       \
                                                                outblock);     \
        break;

#define OPERATOR1D_M(r, i)                                                     \
    case NM(i):                                                                \
        switch (nq0)                                                           \
        {                                                                      \
            BOOST_PP_FOR_##r((NM(i), NM(i), BOOST_PP_MUL(2, NM(i))),           \
                             LEV2TEST1, LEV2UPDATE1, OPERATOR1D_Q)             \
            default :                                                          \
                OPERATOR1D_DEF;                                                \
                break;                                                         \
        }                                                                      \
        break;

#undef OPERATOR2D_DEF_TRI
#undef OPERATOR2D_Q_TRI
#undef OPERATOR2D_M_TRI

#define OPERATOR2D_DEF_TRI                                                     \
    operator2D<LibUtilities::eTriangle, false>(inblock, outblock)

#define OPERATOR2D_Q_TRI(r, i)                                                 \
    case NQ(i):                                                                \
        operator2D<LibUtilities::eTriangle, false, NM(i), NM(i), NQ(i),        \
                   NQ_M1(i)>(inblock, outblock);                               \
        break;

#define OPERATOR2D_M_TRI(r, i)                                                 \
    case NM(i):                                                                \
        switch (nq0)                                                           \
        {                                                                      \
            BOOST_PP_FOR_##r((NM(i), NM_P1(i), BOOST_PP_MUL(2, NM(i))),        \
                             LEV2TEST1, LEV2UPDATE1, OPERATOR2D_Q_TRI)         \
            default :                                                          \
                OPERATOR2D_DEF_TRI;                                            \
                break;                                                         \
        }                                                                      \
        break;

#undef OPERATOR2D_DEF_QUAD
#undef OPERATOR2D_Q_QUAD
#undef OPERATOR2D_M_QUAD

#define OPERATOR2D_DEF_QUAD                                                    \
    operator2D<LibUtilities::eQuadrilateral, false>(inblock, outblock)

#define OPERATOR2D_Q_QUAD(r, i)                                                \
    case NQ(i):                                                                \
        operator2D<LibUtilities::eQuadrilateral, false, NM(i), NM(i), NQ(i),   \
                   NQ(i)>(inblock, outblock);                                  \
        break;

#define OPERATOR2D_M_QUAD(r, i)                                                \
    case NM(i):                                                                \
        switch (nq0)                                                           \
        {                                                                      \
            BOOST_PP_FOR_##r((NM(i), NM(i), BOOST_PP_MUL(2, NM(i))),           \
                             LEV2TEST1, LEV2UPDATE1,                           \
                             OPERATOR2D_Q_QUAD)                                \
            default :                                                          \
                OPERATOR2D_DEF_QUAD;                                           \
                break;                                                         \
        }                                                                      \
        break;

#undef OPERATOR3D_DEF_HEX
#undef OPERATOR3D_Q_HEX
#undef OPERATOR3D_M_HEX

#define OPERATOR3D_DEF_HEX                                                     \
    operator3D<LibUtilities::eHexahedron, false>(inblock, outblock)

#define OPERATOR3D_Q_HEX(r, i)                                                 \
    case NQ(i):                                                                \
        operator3D<LibUtilities::eHexahedron, false, NM(i), NM(i), NM(i),      \
                   NQ(i), NQ(i), NQ(i)>(inblock, outblock);                    \
        break;

#define OPERATOR3D_M_HEX(r, i)                                                 \
    case NM(i):                                                                \
        switch (nq0)                                                           \
        {                                                                      \
            BOOST_PP_FOR_##r((NM(i), NM(i), BOOST_PP_MUL(2, NM(i))),           \
                             LEV2TEST1, LEV2UPDATE1, OPERATOR3D_Q_HEX)         \
            default :                                                          \
                OPERATOR3D_DEF_HEX;                                            \
                break;                                                         \
        }                                                                      \
        break;
#undef OPERATOR3D_DEF_TET
#undef OPERATOR3D_Q_TET
#undef OPERATOR3D_M_TET

#define OPERATOR3D_DEF_TET                                                     \
    operator3D<LibUtilities::eTetrahedron, false>(inblock, outblock)

#define OPERATOR3D_Q_TET(r, i)                                                 \
    case NQ(i):                                                                \
        operator3D<LibUtilities::eTetrahedron, false, NM(i), NM(i), NM(i),     \
                   NQ(i), NQ_M1(i), NQ_M1(i)>(inblock, outblock);              \
        break;

#define OPERATOR3D_M_TET(r, i)                                                 \
    case NM(i):                                                                \
        switch (nq0)                                                           \
        {                                                                      \
            BOOST_PP_FOR_##r((NM(i), NM_P1(i), BOOST_PP_MUL(2, NM(i))),        \
                             LEV2TEST1, LEV2UPDATE1, OPERATOR3D_Q_TET)         \
            default :                                                          \
                OPERATOR3D_DEF_TET;                                            \
                break;                                                         \
            break;                                                             \
        }                                                                      \
        break;

#undef OPERATOR3D_DEF_PRISM
#undef OPERATOR3D_Q_PRISM
#undef OPERATOR3D_M_PRISM

#define OPERATOR3D_DEF_PRISM                                                   \
    operator3D<LibUtilities::ePrism, false>(inblock, outblock)

#define OPERATOR3D_Q_PRISM(r, i)                                               \
    case NQ(i):                                                                \
        operator3D<LibUtilities::ePrism, false, NM(i), NM(i), NM(i), NQ(i),    \
                   NQ(i), NQ_M1(i)>(inblock, outblock);                        \
        break;

#define OPERATOR3D_M_PRISM(r, i)                                               \
    case NM(i):                                                                \
        switch (nq0)                                                           \
        {                                                                      \
            BOOST_PP_FOR_##r((NM(i), NM_P1(i), BOOST_PP_MUL(2, NM(i))),        \
                             LEV2TEST1, LEV2UPDATE1,                           \
                             OPERATOR3D_Q_PRISM)                               \
            default :                                                          \
                OPERATOR3D_DEF_PRISM;                                          \
                break;                                                         \
        }                                                                      \
        break;

#undef OPERATOR3D_DEF_PYR
#undef OPERATOR3D_Q_PYR
#undef OPERATOR3D_M_PYR

#define OPERATOR3D_DEF_PYR                                                     \
    operator3D<LibUtilities::ePyramid, false>(inblock, outblock)

#define OPERATOR3D_Q_PYR(r, i)                                                 \
    case NQ(i):                                                                \
        operator3D<LibUtilities::ePyramid, false, NM(i), NM(i), NM(i), NQ(i),  \
                   NQ(i), NQ_M1(i)>(inblock, outblock);                        \
        break;

#define OPERATOR3D_M_PYR(r, i)                                                 \
    case NM(i):                                                                \
        switch (nq0)                                                           \
        {                                                                      \
            BOOST_PP_FOR_##r((NM(i), NM_P1(i), BOOST_PP_MUL(2, NM(i))),        \
                             LEV2TEST1, LEV2UPDATE1, OPERATOR3D_Q_PYR)         \
            default :                                                          \
                OPERATOR3D_DEF_PYR;                                            \
                break;                                                         \
        }                                                                      \
        break;

// clang-format on
#include "../Common/SwitchLevel2.h"
}
