///////////////////////////////////////////////////////////////////////////////
//
// File: SwitchLevel1_3D.h
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
// Description: One level switch statement with definable bounds for 3D elmts
//
///////////////////////////////////////////////////////////////////////////////

#include <boost/preprocessor/arithmetic/inc.hpp>
#include <boost/preprocessor/comparison/not_equal.hpp>
#include <boost/preprocessor/repetition/for.hpp>
#include <boost/preprocessor/tuple/elem.hpp>

#include "SwitchLimits.h"

/* The following code sets up a switch where the range goes from MIN3D
   to MAX3D which can be set from CMake Parameter NEKTAR_SWITCH_MIN
   and NEKTAR_SWITCH_MAX. Additional macros need to be provided which
   are OPERATOR3D_HEX, OPERATOR3D_TET, OPERATOR3D_PRISM,
   OPERATOR3D_PYR giving details of the switch case for Tris and Quad
   and OPERATOR3D_DEF_HEX, OPERATOR3D_DEF_TET, OPERATOR3D_DEF_PRISM,
   OPERATOR3D_DEF_PYR which are the default, non-templated function
   call. */

/** this macro tests the values of the tuple 'state' to see if the
    first element, given by BOOST_PP_TUPLE_ELEM(0, state), is not
    equal to the second element plus one, given by
    BOOST_PP_INC(BOOST_PP_TUPLE_ELEM(1, state)) and returns 1 if not
    equal otherwise returns zero. */
#define LEV1TEST(r, state)                                                     \
    BOOST_PP_NOT_EQUAL(BOOST_PP_TUPLE_ELEM(0, state),                          \
                       BOOST_PP_INC(BOOST_PP_TUPLE_ELEM(1, state)))

/** This macro returns an updated tuple where the first element, given
    by BOOST_PP_TUPLE_ELEM(0, state), is incremented by one and the
    second element, given by BOOST_PP_TUPLE_ELEM(1, state) remains the
    same. */
#define LEV1UPDATE(r, state)                                                   \
    (BOOST_PP_INC(BOOST_PP_TUPLE_ELEM(0, state)), BOOST_PP_TUPLE_ELEM(1, state))

/* start of included switch statement */
{
    const int nq0 = m_basisKeys[0].GetNumPoints();
    const int nq1 = m_basisKeys[1].GetNumPoints();
    const int nq2 = m_basisKeys[2].GetNumPoints();
    switch (shapeType)
    {
        case LibUtilities::eHexahedron:
            if (nq0 == nq1 && nq0 == nq2)
            {
                switch (nq0)
                {
                    BOOST_PP_FOR((MIN3D, MAX3D), LEV1TEST, LEV1UPDATE,
                                 OPERATOR3D_HEX);
                    default:
                        OPERATOR3D_DEF_HEX;
                        break;
                }
            }
            break;
        case LibUtilities::eTetrahedron:
            if (nq0 == nq1 + 1 && nq0 == nq2 + 1)
            {
                switch (nq0)
                {
                    BOOST_PP_FOR((MIN3D, MAX3D), LEV1TEST, LEV1UPDATE,
                                 OPERATOR3D_TET);
                    break;
                    default:
                        OPERATOR3D_DEF_TET;
                        break;
                }
            }
            break;
        case LibUtilities::ePrism:
            if (nq0 == nq1 && nq0 == nq2 + 1)
            {
                switch (nq0)
                {
                    BOOST_PP_FOR((MIN3D, MAX3D), LEV1TEST, LEV1UPDATE,
                                 OPERATOR3D_PRISM);
                    default:
                        OPERATOR3D_DEF_PRISM;
                        break;
                }
            }
            break;
        case LibUtilities::ePyramid:
            if (nq0 == nq1 && nq0 == nq2 + 1)
            {
                switch (nq0)
                {
                    BOOST_PP_FOR((MIN3D, MAX3D), LEV1TEST, LEV1UPDATE,
                                 OPERATOR3D_PYR);
                    default:
                        OPERATOR3D_DEF_PYR;
                        break;
                }
            }
            break;
        default:
            std::cout << "ERROR: Unknown shapeType" << std::endl;
            break;
    }
}
