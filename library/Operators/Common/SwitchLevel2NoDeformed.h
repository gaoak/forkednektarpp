///////////////////////////////////////////////////////////////////////////////
//
// File: SwitchLevel2NoDeformed.h
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
   orders when deformation is not a template parameter */

#define OPERATOR1D_DEF operator1D<LibUtilities::eSegment>(inblock, outblock)

#define OPERATOR1D_Q(r, i)                                                     \
    case NQ(i):                                                                \
        operator1D<LibUtilities::eSegment, NM(i), NQ(i)>(inblock, outblock);   \
        break;

#define OPERATOR1D_M(r, i)                                                     \
    case NM(i):                                                                \
        switch (nq0)                                                           \
        {                                                                      \
            BOOST_PP_FOR_##r((NM(i), NM(i), BOOST_PP_MUL(2, NM(i))),           \
                             LEV2TEST1, LEV2UPDATE1, OPERATOR1D_Q) default     \
                : OPERATOR1D_DEF;                                              \
            break;                                                             \
        }                                                                      \
        break;

#define OPERATOR2D_DEF_TRI                                                     \
    operator2D<LibUtilities::eTriangle>(inblock, outblock)

#define OPERATOR2D_Q_TRI(r, i)                                                 \
    case NQ(i):                                                                \
        operator2D<LibUtilities::eTriangle, NM(i), NM(i), NQ(i), NQ_M1(i)>(    \
            inblock, outblock);                                                \
        break;

#define OPERATOR2D_M_TRI(r, i)                                                 \
    case NM(i):                                                                \
        switch (nq0)                                                           \
        {                                                                      \
            BOOST_PP_FOR_##r((NM(i), NM_P1(i), BOOST_PP_MUL(2, NM(i))),        \
                             LEV2TEST1, LEV2UPDATE1, OPERATOR2D_Q_TRI) default \
                : OPERATOR2D_DEF_TRI;                                          \
            break;                                                             \
        }                                                                      \
        break;

#define OPERATOR2D_DEF_QUAD                                                    \
    operator2D<LibUtilities::eQuadrilateral>(inblock, outblock)

#define OPERATOR2D_Q_QUAD(r, i)                                                \
    case NQ(i):                                                                \
        operator2D<LibUtilities::eQuadrilateral, NM(i), NM(i), NQ(i), NQ(i)>(  \
            inblock, outblock);                                                \
        break;

#define OPERATOR2D_M_QUAD(r, i)                                                \
    case NM(i):                                                                \
        switch (nq0)                                                           \
        {                                                                      \
            BOOST_PP_FOR_##r((NM(i), NM(i), BOOST_PP_MUL(2, NM(i))),           \
                             LEV2TEST1, LEV2UPDATE1,                           \
                             OPERATOR2D_Q_QUAD) default : OPERATOR2D_DEF_QUAD; \
            break;                                                             \
        }                                                                      \
        break;

#define OPERATOR3D_DEF_HEX                                                     \
    operator3D<LibUtilities::eHexahedron>(inblock, outblock)

#define OPERATOR3D_Q_HEX(r, i)                                                 \
    case NQ(i):                                                                \
        operator3D<LibUtilities::eHexahedron, NM(i), NM(i), NM(i), NQ(i),      \
                   NQ(i), NQ(i)>(inblock, outblock);                           \
        break;

#define OPERATOR3D_M_HEX(r, i)                                                 \
    case NM(i):                                                                \
        switch (nq0)                                                           \
        {                                                                      \
            BOOST_PP_FOR_##r((NM(i), NM(i), BOOST_PP_MUL(2, NM(i))),           \
                             LEV2TEST1, LEV2UPDATE1, OPERATOR3D_Q_HEX) default \
                : OPERATOR3D_DEF_HEX;                                          \
            break;                                                             \
        }                                                                      \
        break;

#define OPERATOR3D_DEF_TET                                                     \
    operator3D<LibUtilities::eTetrahedron>(inblock, outblock)

#define OPERATOR3D_Q_TET(r, i)                                                 \
    case NQ(i):                                                                \
        operator3D<LibUtilities::eTetrahedron, NM(i), NM(i), NM(i), NQ(i),     \
                   NQ_M1(i), NQ_M1(i)>(inblock, outblock);                     \
        break;

#define OPERATOR3D_M_TET(r, i)                                                 \
    case NM(i):                                                                \
        switch (nq0)                                                           \
        {                                                                      \
            BOOST_PP_FOR_##r((NM(i), NM_P1(i), BOOST_PP_MUL(2, NM(i))),        \
                             LEV2TEST1, LEV2UPDATE1, OPERATOR3D_Q_TET) default \
                : OPERATOR3D_DEF_TET;                                          \
            break;                                                             \
        }                                                                      \
        break;

#define OPERATOR3D_DEF_PRISM operator3D<LibUtilities::ePrism>(inblock, outblock)

#define OPERATOR3D_Q_PRISM(r, i)                                               \
    case NQ(i):                                                                \
        operator3D<LibUtilities::ePrism, NM(i), NM(i), NM(i), NQ(i), NQ(i),    \
                   NQ_M1(i)>(inblock, outblock);                               \
        break;

#define OPERATOR3D_M_PRISM(r, i)                                               \
    case NM(i):                                                                \
        switch (nq0)                                                           \
        {                                                                      \
            BOOST_PP_FOR_##r((NM(i), NM_P1(i), BOOST_PP_MUL(2, NM(i))),        \
                             LEV2TEST1, LEV2UPDATE1,                           \
                             OPERATOR3D_Q_PRISM) default                       \
                : OPERATOR3D_DEF_PRISM;                                        \
            break;                                                             \
        }                                                                      \
        break;

#define OPERATOR3D_DEF_PYR operator3D<LibUtilities::ePyramid>(inblock, outblock)

#define OPERATOR3D_Q_PYR(r, i)                                                 \
    case NQ(i):                                                                \
        operator3D<LibUtilities::ePyramid, NM(i), NM(i), NM(i), NQ(i), NQ(i),  \
                   NQ_M1(i)>(inblock, outblock);                               \
        break;

#define OPERATOR3D_M_PYR(r, i)                                                 \
    case NM(i):                                                                \
        switch (nq0)                                                           \
        {                                                                      \
            BOOST_PP_FOR_##r((NM(i), NM_P1(i), BOOST_PP_MUL(2, NM(i))),        \
                             LEV2TEST1, LEV2UPDATE1, OPERATOR3D_Q_PYR) default \
                : OPERATOR3D_DEF_PYR;                                          \
            break;                                                             \
        }                                                                      \
        break;

#include "../Common/SwitchLevel2.h"
