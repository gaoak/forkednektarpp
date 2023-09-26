///////////////////////////////////////////////////////////////////////////////
//
// File: SwitchNodesPoints.h
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
// Description:
//
///////////////////////////////////////////////////////////////////////////////

// This code is the mombo switch statement that is used in mutliple
// operators. It uses preprocessor directives based on the shape
// type and dimension to limit the code inclusion.

// This header is included in the OperatorType.h file. Ideally the
// operator() method would be a template in the Operator base class
// but becasue the operator{1,23}D is both a function and template that
// need to be in the inherited class it is not possible.
if (deformed)
{
    if (dimension == 1)
    {
        const int nq0 = expPtr->GetNumPoints(0);
        switch (Coordim)
        {
            case 1:
                switch (nq0)
                {
                    case 2:
                        operator1D<LibUtilities::eSegment, true, 1, 2>(inptr,
                                                                       outptr);
                        break;
                    case 3:
                        operator1D<LibUtilities::eSegment, true, 1, 3>(inptr,
                                                                       outptr);
                        break;
                    case 4:
                        operator1D<LibUtilities::eSegment, true, 1, 4>(inptr,
                                                                       outptr);
                        break;
                    case 5:
                        operator1D<LibUtilities::eSegment, true, 1, 5>(inptr,
                                                                       outptr);
                        break;
                    case 6:
                        operator1D<LibUtilities::eSegment, true, 1, 6>(inptr,
                                                                       outptr);
                        break;
                    case 7:
                        operator1D<LibUtilities::eSegment, true, 1, 7>(inptr,
                                                                       outptr);
                        break;
                    case 8:
                        operator1D<LibUtilities::eSegment, true, 1, 8>(inptr,
                                                                       outptr);
                        break;
                    case 9:
                        operator1D<LibUtilities::eSegment, true, 1, 9>(inptr,
                                                                       outptr);
                        break;
                    case 10:
                        operator1D<LibUtilities::eSegment, true, 1, 10>(inptr,
                                                                        outptr);
                        break;
                    case 11:
                        operator1D<LibUtilities::eSegment, true, 1, 11>(inptr,
                                                                        outptr);
                        break;
                    case 12:
                        operator1D<LibUtilities::eSegment, true, 1, 12>(inptr,
                                                                        outptr);
                        break;
                    default:
                        operator1D<LibUtilities::eSegment, true>(inptr, outptr);
                        break;
                }
                break;
            case 2:
                switch (nq0)
                {
                    case 2:
                        operator1D<LibUtilities::eSegment, true, 2, 2>(inptr,
                                                                       outptr);
                        break;
                    case 3:
                        operator1D<LibUtilities::eSegment, true, 2, 3>(inptr,
                                                                       outptr);
                        break;
                    case 4:
                        operator1D<LibUtilities::eSegment, true, 2, 4>(inptr,
                                                                       outptr);
                        break;
                    case 5:
                        operator1D<LibUtilities::eSegment, true, 2, 5>(inptr,
                                                                       outptr);
                        break;
                    case 6:
                        operator1D<LibUtilities::eSegment, true, 2, 6>(inptr,
                                                                       outptr);
                        break;
                    case 7:
                        operator1D<LibUtilities::eSegment, true, 2, 7>(inptr,
                                                                       outptr);
                        break;
                    case 8:
                        operator1D<LibUtilities::eSegment, true, 2, 8>(inptr,
                                                                       outptr);
                        break;
                    case 9:
                        operator1D<LibUtilities::eSegment, true, 2, 9>(inptr,
                                                                       outptr);
                        break;
                    case 10:
                        operator1D<LibUtilities::eSegment, true, 2, 10>(inptr,
                                                                        outptr);
                        break;
                    case 11:
                        operator1D<LibUtilities::eSegment, true, 2, 11>(inptr,
                                                                        outptr);
                        break;
                    case 12:
                        operator1D<LibUtilities::eSegment, true, 2, 12>(inptr,
                                                                        outptr);
                        break;
                    default:
                        operator1D<LibUtilities::eSegment, true>(inptr, outptr);
                        break;
                }
                break;
            case 3:
                switch (nq0)
                {
                    case 2:
                        operator1D<LibUtilities::eSegment, true, 3, 2>(inptr,
                                                                       outptr);
                        break;
                    case 3:
                        operator1D<LibUtilities::eSegment, true, 3, 3>(inptr,
                                                                       outptr);
                        break;
                    case 4:
                        operator1D<LibUtilities::eSegment, true, 3, 4>(inptr,
                                                                       outptr);
                        break;
                    case 5:
                        operator1D<LibUtilities::eSegment, true, 3, 5>(inptr,
                                                                       outptr);
                        break;
                    case 6:
                        operator1D<LibUtilities::eSegment, true, 3, 6>(inptr,
                                                                       outptr);
                        break;
                    case 7:
                        operator1D<LibUtilities::eSegment, true, 3, 7>(inptr,
                                                                       outptr);
                        break;
                    case 8:
                        operator1D<LibUtilities::eSegment, true, 3, 8>(inptr,
                                                                       outptr);
                        break;
                    case 9:
                        operator1D<LibUtilities::eSegment, true, 3, 9>(inptr,
                                                                       outptr);
                        break;
                    case 10:
                        operator1D<LibUtilities::eSegment, true, 3, 10>(inptr,
                                                                        outptr);
                        break;
                    case 11:
                        operator1D<LibUtilities::eSegment, true, 3, 11>(inptr,
                                                                        outptr);
                        break;
                    case 12:
                        operator1D<LibUtilities::eSegment, true, 3, 12>(inptr,
                                                                        outptr);
                        break;
                    default:
                        operator1D<LibUtilities::eSegment, true>(inptr, outptr);
                        break;
                }
                break;
            default:
                NEKERROR(ErrorUtil::efatal, "invalid coordinate dimension");
                break;
        }
    }
    else if (dimension == 2)
    {
        const int nq0 = m_basis[0]->GetNumPoints();
        const int nq1 = m_basis[1]->GetNumPoints();
        switch (shapeType)
        {
            case LibUtilities::eTriangle:
                if (nq0 == nq1 + 1)
                {
                    switch (Coordim)
                    {
                        case 2:
                            switch (nq0)
                            {
                                case 3:
                                    operator2D<LibUtilities::eTriangle, true, 2,
                                               3, 2>(inptr, outptr);
                                    break;
                                case 4:
                                    operator2D<LibUtilities::eTriangle, true, 2,
                                               4, 3>(inptr, outptr);
                                    break;
                                case 5:
                                    operator2D<LibUtilities::eTriangle, true, 2,
                                               5, 4>(inptr, outptr);
                                    break;
                                case 6:
                                    operator2D<LibUtilities::eTriangle, true, 2,
                                               6, 5>(inptr, outptr);
                                    break;
                                case 7:
                                    operator2D<LibUtilities::eTriangle, true, 2,
                                               7, 6>(inptr, outptr);
                                    break;
                                case 8:
                                    operator2D<LibUtilities::eTriangle, true, 2,
                                               8, 7>(inptr, outptr);
                                    break;
                                case 9:
                                    operator2D<LibUtilities::eTriangle, true, 2,
                                               9, 8>(inptr, outptr);
                                    break;
                                case 10:
                                    operator2D<LibUtilities::eTriangle, true, 2,
                                               10, 9>(inptr, outptr);
                                    break;
                                case 11:
                                    operator2D<LibUtilities::eTriangle, true, 2,
                                               11, 10>(inptr, outptr);
                                    break;
                                case 12:
                                    operator2D<LibUtilities::eTriangle, true, 2,
                                               12, 11>(inptr, outptr);
                                    break;
                                default:
                                    operator2D<LibUtilities::eTriangle, true>(
                                        inptr, outptr);
                                    break;
                            }
                            break;
                        case 3:
                            switch (nq0)
                            {
                                case 3:
                                    operator2D<LibUtilities::eTriangle, true, 3,
                                               3, 2>(inptr, outptr);
                                    break;
                                case 4:
                                    operator2D<LibUtilities::eTriangle, true, 3,
                                               4, 3>(inptr, outptr);
                                    break;
                                case 5:
                                    operator2D<LibUtilities::eTriangle, true, 3,
                                               5, 4>(inptr, outptr);
                                    break;
                                case 6:
                                    operator2D<LibUtilities::eTriangle, true, 3,
                                               6, 5>(inptr, outptr);
                                    break;
                                case 7:
                                    operator2D<LibUtilities::eTriangle, true, 3,
                                               7, 6>(inptr, outptr);
                                    break;
                                case 8:
                                    operator2D<LibUtilities::eTriangle, true, 3,
                                               8, 7>(inptr, outptr);
                                    break;
                                case 9:
                                    operator2D<LibUtilities::eTriangle, true, 3,
                                               9, 8>(inptr, outptr);
                                    break;
                                case 10:
                                    operator2D<LibUtilities::eTriangle, true, 3,
                                               10, 9>(inptr, outptr);
                                    break;
                                case 11:
                                    operator2D<LibUtilities::eTriangle, true, 3,
                                               11, 10>(inptr, outptr);
                                    break;
                                case 12:
                                    operator2D<LibUtilities::eTriangle, true, 3,
                                               12, 11>(inptr, outptr);
                                    break;
                                default:
                                    operator2D<LibUtilities::eTriangle, true>(
                                        inptr, outptr);
                                    break;
                            }
                            break;
                        default:
                            NEKERROR(ErrorUtil::efatal,
                                     "invalid coordinate dimension");
                            break;
                    }
                }
                else
                {
                    operator2D<LibUtilities::eTriangle, true>(inptr, outptr);
                }
                break;
            case LibUtilities::eQuadrilateral:
                if (nq0 == nq1)
                {
                    switch (Coordim)
                    {
                        case 2:
                            switch (nq0)
                            {
                                case 2:
                                    operator2D<LibUtilities::eQuadrilateral,
                                               true, 2, 2, 2>(inptr, outptr);
                                    break;
                                case 3:
                                    operator2D<LibUtilities::eQuadrilateral,
                                               true, 2, 3, 3>(inptr, outptr);
                                    break;
                                case 4:
                                    operator2D<LibUtilities::eQuadrilateral,
                                               true, 2, 4, 4>(inptr, outptr);
                                    break;
                                case 5:
                                    operator2D<LibUtilities::eQuadrilateral,
                                               true, 2, 5, 5>(inptr, outptr);
                                    break;
                                case 6:
                                    operator2D<LibUtilities::eQuadrilateral,
                                               true, 2, 6, 6>(inptr, outptr);
                                    break;
                                case 7:
                                    operator2D<LibUtilities::eQuadrilateral,
                                               true, 2, 7, 7>(inptr, outptr);
                                    break;
                                case 8:
                                    operator2D<LibUtilities::eQuadrilateral,
                                               true, 2, 8, 8>(inptr, outptr);
                                    break;
                                case 9:
                                    operator2D<LibUtilities::eQuadrilateral,
                                               true, 2, 9, 9>(inptr, outptr);
                                    break;
                                case 10:
                                    operator2D<LibUtilities::eQuadrilateral,
                                               true, 2, 10, 10>(inptr, outptr);
                                    break;
                                case 11:
                                    operator2D<LibUtilities::eQuadrilateral,
                                               true, 2, 11, 11>(inptr, outptr);
                                    break;
                                case 12:
                                    operator2D<LibUtilities::eQuadrilateral,
                                               true, 2, 12, 12>(inptr, outptr);
                                    break;
                                default:
                                    operator2D<LibUtilities::eQuadrilateral,
                                               true>(inptr, outptr);
                                    break;
                            }
                            break;
                        case 3:
                            switch (nq0)
                            {
                                case 2:
                                    operator2D<LibUtilities::eQuadrilateral,
                                               true, 3, 2, 2>(inptr, outptr);
                                    break;
                                case 3:
                                    operator2D<LibUtilities::eQuadrilateral,
                                               true, 3, 3, 3>(inptr, outptr);
                                    break;
                                case 4:
                                    operator2D<LibUtilities::eQuadrilateral,
                                               true, 3, 4, 4>(inptr, outptr);
                                    break;
                                case 5:
                                    operator2D<LibUtilities::eQuadrilateral,
                                               true, 3, 5, 5>(inptr, outptr);
                                    break;
                                case 6:
                                    operator2D<LibUtilities::eQuadrilateral,
                                               true, 3, 6, 6>(inptr, outptr);
                                    break;
                                case 7:
                                    operator2D<LibUtilities::eQuadrilateral,
                                               true, 3, 7, 7>(inptr, outptr);
                                    break;
                                case 8:
                                    operator2D<LibUtilities::eQuadrilateral,
                                               true, 3, 8, 8>(inptr, outptr);
                                    break;
                                case 9:
                                    operator2D<LibUtilities::eQuadrilateral,
                                               true, 3, 9, 9>(inptr, outptr);
                                    break;
                                case 10:
                                    operator2D<LibUtilities::eQuadrilateral,
                                               true, 3, 10, 10>(inptr, outptr);
                                    break;
                                case 11:
                                    operator2D<LibUtilities::eQuadrilateral,
                                               true, 3, 11, 11>(inptr, outptr);
                                    break;
                                case 12:
                                    operator2D<LibUtilities::eQuadrilateral,
                                               true, 3, 12, 12>(inptr, outptr);
                                    break;
                                default:
                                    operator2D<LibUtilities::eQuadrilateral,
                                               true>(inptr, outptr);
                                    break;
                            }
                            break;
                        default:
                            NEKERROR(ErrorUtil::efatal,
                                     "invalid coordinate dimension");
                            break;
                    }
                }
                else
                {
                    operator2D<LibUtilities::eQuadrilateral, true>(inptr,
                                                                   outptr);
                }
                break;
            default:
                std::cout << "Operator not defined or not implemented"
                          << std::endl;
                break;
        }
    }
    else
    {
        const int nq0 = m_basis[0]->GetNumPoints();
        const int nq1 = m_basis[1]->GetNumPoints();
        const int nq2 = m_basis[2]->GetNumPoints();
        switch (shapeType)
        {
            case LibUtilities::eHexahedron:
                if (nq0 == nq1 && nq0 == nq2)
                {
                    switch (nq0)
                    {
                        case 2:
                            operator3D<LibUtilities::eHexahedron, true, 2, 2,
                                       2>(inptr, outptr);
                            break;
                        case 3:
                            operator3D<LibUtilities::eHexahedron, true, 3, 3,
                                       3>(inptr, outptr);
                            break;
                        case 4:
                            operator3D<LibUtilities::eHexahedron, true, 4, 4,
                                       4>(inptr, outptr);
                            break;
                        case 5:
                            operator3D<LibUtilities::eHexahedron, true, 5, 5,
                                       5>(inptr, outptr);
                            break;
                        case 6:
                            operator3D<LibUtilities::eHexahedron, true, 6, 6,
                                       6>(inptr, outptr);
                            break;
                        case 7:
                            operator3D<LibUtilities::eHexahedron, true, 7, 7,
                                       7>(inptr, outptr);
                            break;
                        case 8:
                            operator3D<LibUtilities::eHexahedron, true, 8, 8,
                                       8>(inptr, outptr);
                            break;
                        case 9:
                            operator3D<LibUtilities::eHexahedron, true, 9, 9,
                                       9>(inptr, outptr);
                            break;
                        case 10:
                            operator3D<LibUtilities::eHexahedron, true, 10, 10,
                                       10>(inptr, outptr);
                            break;
                        case 11:
                            operator3D<LibUtilities::eHexahedron, true, 11, 11,
                                       11>(inptr, outptr);
                            break;
                        case 12:
                            operator3D<LibUtilities::eHexahedron, true, 12, 12,
                                       12>(inptr, outptr);
                            break;
                        default:
                            operator3D<LibUtilities::eHexahedron, true>(inptr,
                                                                        outptr);
                            break;
                    }
                }
                break;
            case LibUtilities::eTetrahedron:
                if (nq0 == nq1 + 1 && nq0 == nq2 + 1)
                {
                    switch (nq0)
                    {
                        case 3:
                            operator3D<LibUtilities::eTetrahedron, true, 3, 2,
                                       2>(inptr, outptr);
                            break;
                        case 4:
                            operator3D<LibUtilities::eTetrahedron, true, 4, 3,
                                       3>(inptr, outptr);
                            break;
                        case 5:
                            operator3D<LibUtilities::eTetrahedron, true, 5, 4,
                                       4>(inptr, outptr);
                            break;
                        case 6:
                            operator3D<LibUtilities::eTetrahedron, true, 6, 5,
                                       5>(inptr, outptr);
                            break;
                        case 7:
                            operator3D<LibUtilities::eTetrahedron, true, 7, 6,
                                       6>(inptr, outptr);
                            break;
                        case 8:
                            operator3D<LibUtilities::eTetrahedron, true, 8, 7,
                                       7>(inptr, outptr);
                            break;
                        case 9:
                            operator3D<LibUtilities::eTetrahedron, true, 9, 8,
                                       8>(inptr, outptr);
                            break;
                        case 10:
                            operator3D<LibUtilities::eTetrahedron, true, 10, 9,
                                       9>(inptr, outptr);
                            break;
                        case 11:
                            operator3D<LibUtilities::eTetrahedron, true, 11, 10,
                                       10>(inptr, outptr);
                            break;
                        case 12:
                            operator3D<LibUtilities::eTetrahedron, true, 12, 11,
                                       11>(inptr, outptr);
                            break;
                        default:
                            operator3D<LibUtilities::eTetrahedron, true>(
                                inptr, outptr);
                            break;
                    }
                }
                break;
            case LibUtilities::ePrism:
                if (nq0 == nq1 && nq0 == nq2 + 1)
                {
                    switch (nq0)
                    {
                        case 3:
                            operator3D<LibUtilities::ePrism, true, 3, 3, 2>(
                                inptr, outptr);
                            break;
                        case 4:
                            operator3D<LibUtilities::ePrism, true, 4, 4, 3>(
                                inptr, outptr);
                            break;
                        case 5:
                            operator3D<LibUtilities::ePrism, true, 5, 5, 4>(
                                inptr, outptr);
                            break;
                        case 6:
                            operator3D<LibUtilities::ePrism, true, 6, 6, 5>(
                                inptr, outptr);
                            break;
                        case 7:
                            operator3D<LibUtilities::ePrism, true, 7, 7, 6>(
                                inptr, outptr);
                            break;
                        case 8:
                            operator3D<LibUtilities::ePrism, true, 8, 8, 7>(
                                inptr, outptr);
                            break;
                        case 9:
                            operator3D<LibUtilities::ePrism, true, 9, 9, 8>(
                                inptr, outptr);
                            break;
                        case 10:
                            operator3D<LibUtilities::ePrism, true, 10, 10, 9>(
                                inptr, outptr);
                            break;
                        case 11:
                            operator3D<LibUtilities::ePrism, true, 11, 11, 10>(
                                inptr, outptr);
                            break;
                        case 12:
                            operator3D<LibUtilities::ePrism, true, 12, 12, 11>(
                                inptr, outptr);
                            break;
                        default:
                            operator3D<LibUtilities::ePrism, true>(inptr,
                                                                   outptr);
                            break;
                    }
                }
                break;
            case LibUtilities::ePyramid:
                if (nq0 == nq1 && nq0 == nq2 + 1)
                {
                    switch (nq0)
                    {
                        case 3:
                            operator3D<LibUtilities::ePyramid, true, 3, 3, 2>(
                                inptr, outptr);
                            break;
                        case 4:
                            operator3D<LibUtilities::ePyramid, true, 4, 4, 3>(
                                inptr, outptr);
                            break;
                        case 5:
                            operator3D<LibUtilities::ePyramid, true, 5, 5, 4>(
                                inptr, outptr);
                            break;
                        case 6:
                            operator3D<LibUtilities::ePyramid, true, 6, 6, 5>(
                                inptr, outptr);
                            break;
                        case 7:
                            operator3D<LibUtilities::ePyramid, true, 7, 7, 6>(
                                inptr, outptr);
                            break;
                        case 8:
                            operator3D<LibUtilities::ePyramid, true, 8, 8, 7>(
                                inptr, outptr);
                            break;
                        case 9:
                            operator3D<LibUtilities::ePyramid, true, 9, 9, 8>(
                                inptr, outptr);
                            break;
                        case 10:
                            operator3D<LibUtilities::ePyramid, true, 10, 10, 9>(
                                inptr, outptr);
                            break;
                        case 11:
                            operator3D<LibUtilities::ePyramid, true, 11, 11,
                                       10>(inptr, outptr);
                            break;
                        case 12:
                            operator3D<LibUtilities::ePyramid, true, 12, 12,
                                       11>(inptr, outptr);
                            break;
                        default:
                            operator3D<LibUtilities::ePyramid, true>(inptr,
                                                                     outptr);
                            break;
                    }
                }
                break;
            default:
                std::cout << "ERROR: Unknown shapeType" << std::endl;
                break;
        }
    }
}
else // regular
{
    if (dimension == 1)
    {
        const int nq0 = expPtr->GetNumPoints(0);
        switch (Coordim)
        {
            case 1:
                switch (nq0)
                {
                    case 2:
                        operator1D<LibUtilities::eSegment, false, 1, 2>(inptr,
                                                                        outptr);
                        break;
                    case 3:
                        operator1D<LibUtilities::eSegment, false, 1, 3>(inptr,
                                                                        outptr);
                        break;
                    case 4:
                        operator1D<LibUtilities::eSegment, false, 1, 4>(inptr,
                                                                        outptr);
                        break;
                    case 5:
                        operator1D<LibUtilities::eSegment, false, 1, 5>(inptr,
                                                                        outptr);
                        break;
                    case 6:
                        operator1D<LibUtilities::eSegment, false, 1, 6>(inptr,
                                                                        outptr);
                        break;
                    case 7:
                        operator1D<LibUtilities::eSegment, false, 1, 7>(inptr,
                                                                        outptr);
                        break;
                    case 8:
                        operator1D<LibUtilities::eSegment, false, 1, 8>(inptr,
                                                                        outptr);
                        break;
                    case 9:
                        operator1D<LibUtilities::eSegment, false, 1, 9>(inptr,
                                                                        outptr);
                        break;
                    case 10:
                        operator1D<LibUtilities::eSegment, false, 1, 10>(
                            inptr, outptr);
                        break;
                    case 11:
                        operator1D<LibUtilities::eSegment, false, 1, 11>(
                            inptr, outptr);
                        break;
                    case 12:
                        operator1D<LibUtilities::eSegment, false, 1, 12>(
                            inptr, outptr);
                        break;
                    default:
                        operator1D<LibUtilities::eSegment, false>(inptr,
                                                                  outptr);
                        break;
                }
                break;
            case 2:
                switch (nq0)
                {
                    case 2:
                        operator1D<LibUtilities::eSegment, false, 2, 2>(inptr,
                                                                        outptr);
                        break;
                    case 3:
                        operator1D<LibUtilities::eSegment, false, 2, 3>(inptr,
                                                                        outptr);
                        break;
                    case 4:
                        operator1D<LibUtilities::eSegment, false, 2, 4>(inptr,
                                                                        outptr);
                        break;
                    case 5:
                        operator1D<LibUtilities::eSegment, false, 2, 5>(inptr,
                                                                        outptr);
                        break;
                    case 6:
                        operator1D<LibUtilities::eSegment, false, 2, 6>(inptr,
                                                                        outptr);
                        break;
                    case 7:
                        operator1D<LibUtilities::eSegment, false, 2, 7>(inptr,
                                                                        outptr);
                        break;
                    case 8:
                        operator1D<LibUtilities::eSegment, false, 2, 8>(inptr,
                                                                        outptr);
                        break;
                    case 9:
                        operator1D<LibUtilities::eSegment, false, 2, 9>(inptr,
                                                                        outptr);
                        break;
                    case 10:
                        operator1D<LibUtilities::eSegment, false, 2, 10>(
                            inptr, outptr);
                        break;
                    case 11:
                        operator1D<LibUtilities::eSegment, false, 2, 11>(
                            inptr, outptr);
                        break;
                    case 12:
                        operator1D<LibUtilities::eSegment, false, 2, 12>(
                            inptr, outptr);
                        break;
                    default:
                        operator1D<LibUtilities::eSegment, false>(inptr,
                                                                  outptr);
                        break;
                }
                break;
            case 3:
                switch (nq0)
                {
                    case 2:
                        operator1D<LibUtilities::eSegment, false, 3, 2>(inptr,
                                                                        outptr);
                        break;
                    case 3:
                        operator1D<LibUtilities::eSegment, false, 3, 3>(inptr,
                                                                        outptr);
                        break;
                    case 4:
                        operator1D<LibUtilities::eSegment, false, 3, 4>(inptr,
                                                                        outptr);
                        break;
                    case 5:
                        operator1D<LibUtilities::eSegment, false, 3, 5>(inptr,
                                                                        outptr);
                        break;
                    case 6:
                        operator1D<LibUtilities::eSegment, false, 3, 6>(inptr,
                                                                        outptr);
                        break;
                    case 7:
                        operator1D<LibUtilities::eSegment, false, 3, 7>(inptr,
                                                                        outptr);
                        break;
                    case 8:
                        operator1D<LibUtilities::eSegment, false, 3, 8>(inptr,
                                                                        outptr);
                        break;
                    case 9:
                        operator1D<LibUtilities::eSegment, false, 3, 9>(inptr,
                                                                        outptr);
                        break;
                    case 10:
                        operator1D<LibUtilities::eSegment, false, 3, 10>(
                            inptr, outptr);
                        break;
                    case 11:
                        operator1D<LibUtilities::eSegment, false, 3, 11>(
                            inptr, outptr);
                        break;
                    case 12:
                        operator1D<LibUtilities::eSegment, false, 3, 12>(
                            inptr, outptr);
                        break;
                    default:
                        operator1D<LibUtilities::eSegment, false>(inptr,
                                                                  outptr);
                        break;
                }
                break;
            default:
                NEKERROR(ErrorUtil::efatal, "invalid coordinate dimension");
                break;
        }
    }
    else if (dimension == 2)
    {
        const int nq0 = m_basis[0]->GetNumPoints();
        const int nq1 = m_basis[1]->GetNumPoints();
        switch (shapeType)
        {
            case LibUtilities::eTriangle:
                if (nq0 == nq1 + 1)
                {
                    switch (Coordim)
                    {
                        case 2:
                            switch (nq0)
                            {
                                case 3:
                                    operator2D<LibUtilities::eTriangle, false,
                                               2, 3, 2>(inptr, outptr);
                                    break;
                                case 4:
                                    operator2D<LibUtilities::eTriangle, false,
                                               2, 4, 3>(inptr, outptr);
                                    break;
                                case 5:
                                    operator2D<LibUtilities::eTriangle, false,
                                               2, 5, 4>(inptr, outptr);
                                    break;
                                case 6:
                                    operator2D<LibUtilities::eTriangle, false,
                                               2, 6, 5>(inptr, outptr);
                                    break;
                                case 7:
                                    operator2D<LibUtilities::eTriangle, false,
                                               2, 7, 6>(inptr, outptr);
                                    break;
                                case 8:
                                    operator2D<LibUtilities::eTriangle, false,
                                               2, 8, 7>(inptr, outptr);
                                    break;
                                case 9:
                                    operator2D<LibUtilities::eTriangle, false,
                                               2, 9, 8>(inptr, outptr);
                                    break;
                                case 10:
                                    operator2D<LibUtilities::eTriangle, false,
                                               2, 10, 9>(inptr, outptr);
                                    break;
                                case 11:
                                    operator2D<LibUtilities::eTriangle, false,
                                               2, 11, 10>(inptr, outptr);
                                    break;
                                case 12:
                                    operator2D<LibUtilities::eTriangle, false,
                                               2, 12, 11>(inptr, outptr);
                                    break;
                                default:
                                    operator2D<LibUtilities::eTriangle, false>(
                                        inptr, outptr);
                                    break;
                            }
                            break;
                        case 3:
                            switch (nq0)
                            {
                                case 3:
                                    operator2D<LibUtilities::eTriangle, false,
                                               3, 3, 2>(inptr, outptr);
                                    break;
                                case 4:
                                    operator2D<LibUtilities::eTriangle, false,
                                               3, 4, 3>(inptr, outptr);
                                    break;
                                case 5:
                                    operator2D<LibUtilities::eTriangle, false,
                                               3, 5, 4>(inptr, outptr);
                                    break;
                                case 6:
                                    operator2D<LibUtilities::eTriangle, false,
                                               3, 6, 5>(inptr, outptr);
                                    break;
                                case 7:
                                    operator2D<LibUtilities::eTriangle, false,
                                               3, 7, 6>(inptr, outptr);
                                    break;
                                case 8:
                                    operator2D<LibUtilities::eTriangle, false,
                                               3, 8, 7>(inptr, outptr);
                                    break;
                                case 9:
                                    operator2D<LibUtilities::eTriangle, false,
                                               3, 9, 8>(inptr, outptr);
                                    break;
                                case 10:
                                    operator2D<LibUtilities::eTriangle, false,
                                               3, 10, 9>(inptr, outptr);
                                    break;
                                case 11:
                                    operator2D<LibUtilities::eTriangle, false,
                                               3, 11, 10>(inptr, outptr);
                                    break;
                                case 12:
                                    operator2D<LibUtilities::eTriangle, false,
                                               3, 12, 11>(inptr, outptr);
                                    break;
                                default:
                                    operator2D<LibUtilities::eTriangle, false>(
                                        inptr, outptr);
                                    break;
                            }
                            break;
                        default:
                            NEKERROR(ErrorUtil::efatal,
                                     "invalid coordinate dimension");
                            break;
                    }
                }
                else
                {
                    operator2D<LibUtilities::eTriangle, false>(inptr, outptr);
                }
                break;
            case LibUtilities::eQuadrilateral:
                if (nq0 == nq1)
                {
                    switch (Coordim)
                    {
                        case 2:
                            switch (nq0)
                            {
                                case 2:
                                    operator2D<LibUtilities::eQuadrilateral,
                                               false, 2, 2, 2>(inptr, outptr);
                                    break;
                                case 3:
                                    operator2D<LibUtilities::eQuadrilateral,
                                               false, 2, 3, 3>(inptr, outptr);
                                    break;
                                case 4:
                                    operator2D<LibUtilities::eQuadrilateral,
                                               false, 2, 4, 4>(inptr, outptr);
                                    break;
                                case 5:
                                    operator2D<LibUtilities::eQuadrilateral,
                                               false, 2, 5, 5>(inptr, outptr);
                                    break;
                                case 6:
                                    operator2D<LibUtilities::eQuadrilateral,
                                               false, 2, 6, 6>(inptr, outptr);
                                    break;
                                case 7:
                                    operator2D<LibUtilities::eQuadrilateral,
                                               false, 2, 7, 7>(inptr, outptr);
                                    break;
                                case 8:
                                    operator2D<LibUtilities::eQuadrilateral,
                                               false, 2, 8, 8>(inptr, outptr);
                                    break;
                                case 9:
                                    operator2D<LibUtilities::eQuadrilateral,
                                               false, 2, 9, 9>(inptr, outptr);
                                    break;
                                case 10:
                                    operator2D<LibUtilities::eQuadrilateral,
                                               false, 2, 10, 10>(inptr, outptr);
                                    break;
                                case 11:
                                    operator2D<LibUtilities::eQuadrilateral,
                                               false, 2, 11, 11>(inptr, outptr);
                                    break;
                                case 12:
                                    operator2D<LibUtilities::eQuadrilateral,
                                               false, 2, 12, 12>(inptr, outptr);
                                    break;
                                default:
                                    operator2D<LibUtilities::eQuadrilateral,
                                               false>(inptr, outptr);
                                    break;
                            }
                            break;
                        case 3:
                            switch (nq0)
                            {
                                case 2:
                                    operator2D<LibUtilities::eQuadrilateral,
                                               false, 3, 2, 2>(inptr, outptr);
                                    break;
                                case 3:
                                    operator2D<LibUtilities::eQuadrilateral,
                                               false, 3, 3, 3>(inptr, outptr);
                                    break;
                                case 4:
                                    operator2D<LibUtilities::eQuadrilateral,
                                               false, 3, 4, 4>(inptr, outptr);
                                    break;
                                case 5:
                                    operator2D<LibUtilities::eQuadrilateral,
                                               false, 3, 5, 5>(inptr, outptr);
                                    break;
                                case 6:
                                    operator2D<LibUtilities::eQuadrilateral,
                                               false, 3, 6, 6>(inptr, outptr);
                                    break;
                                case 7:
                                    operator2D<LibUtilities::eQuadrilateral,
                                               false, 3, 7, 7>(inptr, outptr);
                                    break;
                                case 8:
                                    operator2D<LibUtilities::eQuadrilateral,
                                               false, 3, 8, 8>(inptr, outptr);
                                    break;
                                case 9:
                                    operator2D<LibUtilities::eQuadrilateral,
                                               false, 3, 9, 9>(inptr, outptr);
                                    break;
                                case 10:
                                    operator2D<LibUtilities::eQuadrilateral,
                                               false, 3, 10, 10>(inptr, outptr);
                                    break;
                                case 11:
                                    operator2D<LibUtilities::eQuadrilateral,
                                               false, 3, 11, 11>(inptr, outptr);
                                    break;
                                case 12:
                                    operator2D<LibUtilities::eQuadrilateral,
                                               false, 3, 12, 12>(inptr, outptr);
                                    break;
                                default:
                                    operator2D<LibUtilities::eQuadrilateral,
                                               false>(inptr, outptr);
                                    break;
                            }
                            break;
                        default:
                            NEKERROR(ErrorUtil::efatal,
                                     "invalid coordinate dimension");
                            break;
                    }
                }
                else
                {
                    operator2D<LibUtilities::eQuadrilateral, false>(inptr,
                                                                    outptr);
                }
                break;
            default:
                std::cout << "Operator not defined or not implemented"
                          << std::endl;
                break;
        }
    }
    else
    {
        const int nq0 = m_basis[0]->GetNumPoints();
        const int nq1 = m_basis[1]->GetNumPoints();
        const int nq2 = m_basis[2]->GetNumPoints();
        switch (shapeType)
        {
            case LibUtilities::eHexahedron:
                if (nq0 == nq1 && nq0 == nq2)
                {
                    switch (nq0)
                    {
                        case 2:
                            operator3D<LibUtilities::eHexahedron, false, 2, 2,
                                       2>(inptr, outptr);
                            break;
                        case 3:
                            operator3D<LibUtilities::eHexahedron, false, 3, 3,
                                       3>(inptr, outptr);
                            break;
                        case 4:
                            operator3D<LibUtilities::eHexahedron, false, 4, 4,
                                       4>(inptr, outptr);
                            break;
                        case 5:
                            operator3D<LibUtilities::eHexahedron, false, 5, 5,
                                       5>(inptr, outptr);
                            break;
                        case 6:
                            operator3D<LibUtilities::eHexahedron, false, 6, 6,
                                       6>(inptr, outptr);
                            break;
                        case 7:
                            operator3D<LibUtilities::eHexahedron, false, 7, 7,
                                       7>(inptr, outptr);
                            break;
                        case 8:
                            operator3D<LibUtilities::eHexahedron, false, 8, 8,
                                       8>(inptr, outptr);
                            break;
                        case 9:
                            operator3D<LibUtilities::eHexahedron, false, 9, 9,
                                       9>(inptr, outptr);
                            break;
                        case 10:
                            operator3D<LibUtilities::eHexahedron, false, 10, 10,
                                       10>(inptr, outptr);
                            break;
                        case 11:
                            operator3D<LibUtilities::eHexahedron, false, 11, 11,
                                       11>(inptr, outptr);
                            break;
                        case 12:
                            operator3D<LibUtilities::eHexahedron, false, 12, 12,
                                       12>(inptr, outptr);
                            break;
                        default:
                            operator3D<LibUtilities::eHexahedron, false>(
                                inptr, outptr);
                            break;
                    }
                }
                break;
            case LibUtilities::eTetrahedron:
                if (nq0 == nq1 + 1 && nq0 == nq2 + 1)
                {
                    switch (nq0)
                    {
                        case 3:
                            operator3D<LibUtilities::eTetrahedron, false, 3, 2,
                                       2>(inptr, outptr);
                            break;
                        case 4:
                            operator3D<LibUtilities::eTetrahedron, false, 4, 3,
                                       3>(inptr, outptr);
                            break;
                        case 5:
                            operator3D<LibUtilities::eTetrahedron, false, 5, 4,
                                       4>(inptr, outptr);
                            break;
                        case 6:
                            operator3D<LibUtilities::eTetrahedron, false, 6, 5,
                                       5>(inptr, outptr);
                            break;
                        case 7:
                            operator3D<LibUtilities::eTetrahedron, false, 7, 6,
                                       6>(inptr, outptr);
                            break;
                        case 8:
                            operator3D<LibUtilities::eTetrahedron, false, 8, 7,
                                       7>(inptr, outptr);
                            break;
                        case 9:
                            operator3D<LibUtilities::eTetrahedron, false, 9, 8,
                                       8>(inptr, outptr);
                            break;
                        case 10:
                            operator3D<LibUtilities::eTetrahedron, false, 10, 9,
                                       9>(inptr, outptr);
                            break;
                        case 11:
                            operator3D<LibUtilities::eTetrahedron, false, 11,
                                       10, 10>(inptr, outptr);
                            break;
                        case 12:
                            operator3D<LibUtilities::eTetrahedron, false, 12,
                                       11, 11>(inptr, outptr);
                            break;
                        default:
                            operator3D<LibUtilities::eTetrahedron, false>(
                                inptr, outptr);
                            break;
                    }
                }
                break;
            case LibUtilities::ePrism:
                if (nq0 == nq1 && nq0 == nq2 + 1)
                {
                    switch (nq0)
                    {
                        case 3:
                            operator3D<LibUtilities::ePrism, false, 3, 3, 2>(
                                inptr, outptr);
                            break;
                        case 4:
                            operator3D<LibUtilities::ePrism, false, 4, 4, 3>(
                                inptr, outptr);
                            break;
                        case 5:
                            operator3D<LibUtilities::ePrism, false, 5, 5, 4>(
                                inptr, outptr);
                            break;
                        case 6:
                            operator3D<LibUtilities::ePrism, false, 6, 6, 5>(
                                inptr, outptr);
                            break;
                        case 7:
                            operator3D<LibUtilities::ePrism, false, 7, 7, 6>(
                                inptr, outptr);
                            break;
                        case 8:
                            operator3D<LibUtilities::ePrism, false, 8, 8, 7>(
                                inptr, outptr);
                            break;
                        case 9:
                            operator3D<LibUtilities::ePrism, false, 9, 9, 8>(
                                inptr, outptr);
                            break;
                        case 10:
                            operator3D<LibUtilities::ePrism, false, 10, 10, 9>(
                                inptr, outptr);
                            break;
                        case 11:
                            operator3D<LibUtilities::ePrism, false, 11, 11, 10>(
                                inptr, outptr);
                            break;
                        case 12:
                            operator3D<LibUtilities::ePrism, false, 12, 12, 11>(
                                inptr, outptr);
                            break;
                        default:
                            operator3D<LibUtilities::ePrism, false>(inptr,
                                                                    outptr);
                            break;
                    }
                }
                break;
            case LibUtilities::ePyramid:
                if (nq0 == nq1 && nq0 == nq2 + 1)
                {
                    switch (nq0)
                    {
                        case 3:
                            operator3D<LibUtilities::ePyramid, false, 3, 3, 2>(
                                inptr, outptr);
                            break;
                        case 4:
                            operator3D<LibUtilities::ePyramid, false, 4, 4, 3>(
                                inptr, outptr);
                            break;
                        case 5:
                            operator3D<LibUtilities::ePyramid, false, 5, 5, 4>(
                                inptr, outptr);
                            break;
                        case 6:
                            operator3D<LibUtilities::ePyramid, false, 6, 6, 5>(
                                inptr, outptr);
                            break;
                        case 7:
                            operator3D<LibUtilities::ePyramid, false, 7, 7, 6>(
                                inptr, outptr);
                            break;
                        case 8:
                            operator3D<LibUtilities::ePyramid, false, 8, 8, 7>(
                                inptr, outptr);
                            break;
                        case 9:
                            operator3D<LibUtilities::ePyramid, false, 9, 9, 8>(
                                inptr, outptr);
                            break;
                        case 10:
                            operator3D<LibUtilities::ePyramid, false, 10, 10,
                                       9>(inptr, outptr);
                            break;
                        case 11:
                            operator3D<LibUtilities::ePyramid, false, 11, 11,
                                       10>(inptr, outptr);
                            break;
                        case 12:
                            operator3D<LibUtilities::ePyramid, false, 12, 12,
                                       11>(inptr, outptr);
                            break;
                        default:
                            operator3D<LibUtilities::ePyramid, false>(inptr,
                                                                      outptr);
                            break;
                    }
                }
                break;
            default:
                std::cout << "ERROR: Unknown shapeType" << std::endl;
                break;
        }
    }
}