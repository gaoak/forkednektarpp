///////////////////////////////////////////////////////////////////////////////
//
// File: SwitchLevel1_2D.h
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
// Description: One level switch statement with definable bounds for 2D
//
///////////////////////////////////////////////////////////////////////////////

#include <boost/preprocessor/arithmetic/inc.hpp>
#include <boost/preprocessor/comparison/not_equal.hpp>
#include <boost/preprocessor/repetition/for.hpp>
#include <boost/preprocessor/tuple/elem.hpp>

#include "SwitchLimits.h"

/* The following code sets up a switch where the range goes from MIN2D
   to MAX2D which can be set from CMake Parameter NEKTAR_SWITCH_MIN
   and NEKTAR_SWITCH_MAX. Additional macros need to be provided
   which are OPERATOR2D_TRI, OPERATOR2D_QUAD giving details of the
   switch case for Tris and Quad and OPERATOR2D_DEF_TRI,
   OPERATOR2D_DEF_QUAD which are the default, non-templated function
   call. */

/** This macro tests the values of the tuple 'state' to see if the
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
    switch (shapeType)
    {
        case LibUtilities::eTriangle:
            if (nq0 == nq1 + 1)
            {
                switch (nq0)
                {
                    BOOST_PP_FOR((MIN2D, MAX2D), LEV1TEST, LEV1UPDATE,
                                 OPERATOR2D_TRI);
                    default:
                        OPERATOR2D_DEF_TRI;
                        break;
                }
            }
            else
            {
                OPERATOR2D_DEF_TRI;
            }
            break;
        case LibUtilities::eQuadrilateral:
            if (nq0 == nq1)
            {
                switch (nq0)
                {
                    BOOST_PP_FOR((MIN2D, MAX2D), LEV1TEST, LEV1UPDATE,
                                 OPERATOR2D_QUAD);
                    default:
                        OPERATOR2D_DEF_QUAD;
                        break;
                }
            }
            else
            {
                OPERATOR2D_DEF_QUAD;
            }
            break;
        default:
            std::cout << "ERROR: Unknown shapeType" << std::endl;
            break;
    }
}
