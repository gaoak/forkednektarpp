///////////////////////////////////////////////////////////////////////////////
//
// File: SwitchLevel2.h
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
// Description: Two level switch statement with definable bounds
//
///////////////////////////////////////////////////////////////////////////////

#include "SwitchLevel2Defs.h"

if (dimension == 1)
{
    const int nm0 = m_basisKeys[0].GetNumModes();
    const int nq0 = m_basisKeys[0].GetNumPoints();
    switch (nm0)
    {
        /*
          expand switch statement for mm0  from MIN1D to MAX1D with an inner
          switch of quadrature modes from nm0 to 2*nm0 , i.e.
          ..
          case 4:
              switch(nq0)
              {
              ...
                 case 5:
                 operator1D<LibUtilities::eSegment,4,5>(inblock,outblock);
                 break;
                 ...
                 }
        */

        BOOST_PP_FOR((MIN1D, 0, MAX1D), LEV2TEST, LEV2UPDATE, OPERATOR1D_M);
        default:
            OPERATOR1D_DEF;
            break;
    }
}
else if (dimension == 2)
{
    const int nm0 = m_basisKeys[0].GetNumModes();
    const int nq0 = m_basisKeys[0].GetNumPoints();
    const int nm1 = m_basisKeys[1].GetNumModes();
    const int nq1 = m_basisKeys[1].GetNumPoints();
    switch (shapeType)
    {
        case LibUtilities::eTriangle:
            if (nm0 == nm1 && nq0 == nq1 + 1)
            {
                switch (nm0)
                {
                    BOOST_PP_FOR((MIN2D, 0, MAX2D), LEV2TEST, LEV2UPDATE,
                                 OPERATOR2D_M_TRI);
                    default:
                        OPERATOR2D_DEF_TRI;
                        break;
                }
            }
            break;
        case LibUtilities::eQuadrilateral:
            if (nm0 == nm1 && nq0 == nq1)
            {
                switch (nm0)
                {
                    BOOST_PP_FOR((MIN2D, 0, MAX2D), LEV2TEST, LEV2UPDATE,
                                 OPERATOR2D_M_QUAD);
                    default:
                        OPERATOR2D_DEF_QUAD;
                        break;
                }
            }
            break;
        default:
            std::cout << "Operator not defined or not implemented" << std::endl;
            break;
    }
}
else
{
    const int nm0 = m_basisKeys[0].GetNumModes();
    const int nm1 = m_basisKeys[1].GetNumModes();
    const int nm2 = m_basisKeys[2].GetNumModes();
    const int nq0 = m_basisKeys[0].GetNumPoints();
    const int nq1 = m_basisKeys[1].GetNumPoints();
    const int nq2 = m_basisKeys[2].GetNumPoints();
    switch (shapeType)
    {
        case LibUtilities::eHexahedron:
            if (nm0 == nm1 && nm0 == nm2 && nq0 == nq1 && nq0 == nq2)
            {
                switch (nm0)
                {
                    BOOST_PP_FOR((MIN3D, 0, MAX3D), LEV2TEST, LEV2UPDATE,
                                 OPERATOR3D_M_HEX);
                    default:
                        OPERATOR3D_DEF_HEX;
                        break;
                }
            }
            break;
        case LibUtilities::eTetrahedron:
            if (nm0 == nm1 && nm0 == nm2 && nq0 == nq1 + 1 && nq0 == nq2 + 1)
            {
                switch (nm0)
                {
                    BOOST_PP_FOR((MIN3D, 0, MAX3D), LEV2TEST, LEV2UPDATE,
                                 OPERATOR3D_M_TET);
                    break;
                    default:
                        OPERATOR3D_DEF_TET;
                        break;
                }
            }
            break;
        case LibUtilities::ePrism:
            if (nm0 == nm1 && nm0 == nm2 && nq0 == nq1 && nq0 == nq2 + 1)
            {
                switch (nm0)
                {

                    BOOST_PP_FOR((MIN3D, 0, MAX3D), LEV2TEST, LEV2UPDATE,
                                 OPERATOR3D_M_PRISM);
                    default:
                        OPERATOR3D_DEF_PRISM;
                        break;
                }
            }
            break;
        case LibUtilities::ePyramid:
            if (nm0 == nm1 && nm0 == nm2 && nq0 == nq1 && nq0 == nq2 + 1)
            {
                switch (nm0)
                {
                    BOOST_PP_FOR((MIN3D, 0, MAX3D), LEV2TEST, LEV2UPDATE,
                                 OPERATOR3D_M_PYR);
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
