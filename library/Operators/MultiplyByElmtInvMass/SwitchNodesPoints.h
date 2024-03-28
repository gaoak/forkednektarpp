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
if (dimension == 1)
{
    const int nm0 = expPtr->GetBasisNumModes(0);
    const int nq0 = expPtr->GetNumPoints(0);
    switch (nm0)
    {
        case 2:
            switch (nq0)
            {
                case 2:
                    operator1D<LibUtilities::eSegment, 2, 2>(inptr, outptr);
                    break;
                case 3:
                    operator1D<LibUtilities::eSegment, 2, 3>(inptr, outptr);
                    break;
                case 4:
                    operator1D<LibUtilities::eSegment, 2, 4>(inptr, outptr);
                    break;
                default:
                    operator1D<LibUtilities::eSegment>(inptr, outptr);
                    break;
            }
            break;
        case 3:
            switch (nq0)
            {
                case 3:
                    operator1D<LibUtilities::eSegment, 3, 3>(inptr, outptr);
                    break;
                case 4:
                    operator1D<LibUtilities::eSegment, 3, 4>(inptr, outptr);
                    break;
                case 5:
                    operator1D<LibUtilities::eSegment, 3, 5>(inptr, outptr);
                    break;
                case 6:
                    operator1D<LibUtilities::eSegment, 3, 6>(inptr, outptr);
                    break;
                default:
                    operator1D<LibUtilities::eSegment>(inptr, outptr);
                    break;
            }
            break;
        case 4:
            switch (nq0)
            {
                case 4:
                    operator1D<LibUtilities::eSegment, 4, 4>(inptr, outptr);
                    break;
                case 5:
                    operator1D<LibUtilities::eSegment, 4, 5>(inptr, outptr);
                    break;
                case 6:
                    operator1D<LibUtilities::eSegment, 4, 6>(inptr, outptr);
                    break;
                case 7:
                    operator1D<LibUtilities::eSegment, 4, 7>(inptr, outptr);
                    break;
                case 8:
                    operator1D<LibUtilities::eSegment, 4, 8>(inptr, outptr);
                    break;
                default:
                    operator1D<LibUtilities::eSegment>(inptr, outptr);
                    break;
            }
            break;
        case 5:
            switch (nq0)
            {
                case 5:
                    operator1D<LibUtilities::eSegment, 5, 5>(inptr, outptr);
                    break;
                case 6:
                    operator1D<LibUtilities::eSegment, 5, 6>(inptr, outptr);
                    break;
                case 7:
                    operator1D<LibUtilities::eSegment, 5, 7>(inptr, outptr);
                    break;
                case 8:
                    operator1D<LibUtilities::eSegment, 5, 8>(inptr, outptr);
                    break;
                case 9:
                    operator1D<LibUtilities::eSegment, 5, 9>(inptr, outptr);
                    break;
                case 10:
                    operator1D<LibUtilities::eSegment, 5, 10>(inptr, outptr);
                    break;
                default:
                    operator1D<LibUtilities::eSegment>(inptr, outptr);
                    break;
            }
            break;
        case 6:
            switch (nq0)
            {
                case 6:
                    operator1D<LibUtilities::eSegment, 6, 6>(inptr, outptr);
                    break;
                case 7:
                    operator1D<LibUtilities::eSegment, 6, 7>(inptr, outptr);
                    break;
                case 8:
                    operator1D<LibUtilities::eSegment, 6, 8>(inptr, outptr);
                    break;
                case 9:
                    operator1D<LibUtilities::eSegment, 6, 9>(inptr, outptr);
                    break;
                case 10:
                    operator1D<LibUtilities::eSegment, 6, 10>(inptr, outptr);
                    break;
                case 11:
                    operator1D<LibUtilities::eSegment, 6, 11>(inptr, outptr);
                    break;
                case 12:
                    operator1D<LibUtilities::eSegment, 6, 12>(inptr, outptr);
                    break;
                default:
                    operator1D<LibUtilities::eSegment>(inptr, outptr);
                    break;
            }
            break;
        case 7:
            switch (nq0)
            {
                case 7:
                    operator1D<LibUtilities::eSegment, 7, 7>(inptr, outptr);
                    break;
                case 8:
                    operator1D<LibUtilities::eSegment, 7, 8>(inptr, outptr);
                    break;
                case 9:
                    operator1D<LibUtilities::eSegment, 7, 9>(inptr, outptr);
                    break;
                case 10:
                    operator1D<LibUtilities::eSegment, 7, 10>(inptr, outptr);
                    break;
                case 11:
                    operator1D<LibUtilities::eSegment, 7, 11>(inptr, outptr);
                    break;
                case 12:
                    operator1D<LibUtilities::eSegment, 7, 12>(inptr, outptr);
                    break;
                case 13:
                    operator1D<LibUtilities::eSegment, 7, 13>(inptr, outptr);
                    break;
                case 14:
                    operator1D<LibUtilities::eSegment, 7, 14>(inptr, outptr);
                    break;
                default:
                    operator1D<LibUtilities::eSegment>(inptr, outptr);
                    break;
            }
            break;
        case 8:
            switch (nq0)
            {
                case 8:
                    operator1D<LibUtilities::eSegment, 8, 8>(inptr, outptr);
                    break;
                case 9:
                    operator1D<LibUtilities::eSegment, 8, 9>(inptr, outptr);
                    break;
                case 10:
                    operator1D<LibUtilities::eSegment, 8, 10>(inptr, outptr);
                    break;
                case 11:
                    operator1D<LibUtilities::eSegment, 8, 11>(inptr, outptr);
                    break;
                case 12:
                    operator1D<LibUtilities::eSegment, 8, 12>(inptr, outptr);
                    break;
                case 13:
                    operator1D<LibUtilities::eSegment, 8, 13>(inptr, outptr);
                    break;
                case 14:
                    operator1D<LibUtilities::eSegment, 8, 14>(inptr, outptr);
                    break;
                case 15:
                    operator1D<LibUtilities::eSegment, 8, 15>(inptr, outptr);
                    break;
                case 16:
                    operator1D<LibUtilities::eSegment, 8, 16>(inptr, outptr);
                    break;
                default:
                    operator1D<LibUtilities::eSegment>(inptr, outptr);
                    break;
            }
            break;
        default:
            operator1D<LibUtilities::eSegment>(inptr, outptr);
            break;
    }
}
else if (dimension == 2)
{
    const int nm0 = m_basis[0]->GetNumModes();
    const int nm1 = m_basis[1]->GetNumModes();
    const int nq0 = m_basis[0]->GetNumPoints();
    const int nq1 = m_basis[1]->GetNumPoints();
    switch (shapeType)
    {
        case LibUtilities::eTriangle:
            if (nm0 == nm1 && nq0 == nq1 + 1)
            {
                switch (nm0)
                {
                    case 2:
                        switch (nq0)
                        {
                            case 3:
                                operator2D<LibUtilities::eTriangle, 2, 2, 3, 2>(
                                    inptr, outptr);
                                break;
                            case 4:
                                operator2D<LibUtilities::eTriangle, 2, 2, 4, 3>(
                                    inptr, outptr);
                                break;
                            default:
                                operator2D<LibUtilities::eTriangle>(inptr,
                                                                    outptr);
                                break;
                        }
                        break;
                    case 3:
                        switch (nq0)
                        {
                            case 4:
                                operator2D<LibUtilities::eTriangle, 3, 3, 4, 3>(
                                    inptr, outptr);
                                break;
                            case 5:
                                operator2D<LibUtilities::eTriangle, 3, 3, 5, 4>(
                                    inptr, outptr);
                                break;
                            case 6:
                                operator2D<LibUtilities::eTriangle, 3, 3, 6, 5>(
                                    inptr, outptr);
                                break;
                            case 7:
                                operator2D<LibUtilities::eTriangle, 3, 3, 7, 6>(
                                    inptr, outptr);
                                break;
                            default:
                                operator2D<LibUtilities::eTriangle>(inptr,
                                                                    outptr);
                                break;
                        }
                        break;
                    case 4:
                        switch (nq0)
                        {
                            case 5:
                                operator2D<LibUtilities::eTriangle, 4, 4, 5, 4>(
                                    inptr, outptr);
                                break;
                            case 6:
                                operator2D<LibUtilities::eTriangle, 4, 4, 6, 5>(
                                    inptr, outptr);
                                break;
                            case 7:
                                operator2D<LibUtilities::eTriangle, 4, 4, 7, 6>(
                                    inptr, outptr);
                                break;
                            case 8:
                                operator2D<LibUtilities::eTriangle, 4, 4, 8, 7>(
                                    inptr, outptr);
                                break;
                            default:
                                operator2D<LibUtilities::eTriangle>(inptr,
                                                                    outptr);
                                break;
                        }
                        break;
                    case 5:
                        switch (nq0)
                        {
                            case 6:
                                operator2D<LibUtilities::eTriangle, 5, 5, 6, 5>(
                                    inptr, outptr);
                                break;
                            case 7:
                                operator2D<LibUtilities::eTriangle, 5, 5, 7, 6>(
                                    inptr, outptr);
                                break;
                            case 8:
                                operator2D<LibUtilities::eTriangle, 5, 5, 8, 7>(
                                    inptr, outptr);
                                break;
                            case 9:
                                operator2D<LibUtilities::eTriangle, 5, 5, 9, 8>(
                                    inptr, outptr);
                                break;
                            case 10:
                                operator2D<LibUtilities::eTriangle, 5, 5, 10,
                                           9>(inptr, outptr);
                                break;
                            default:
                                operator2D<LibUtilities::eTriangle>(inptr,
                                                                    outptr);
                                break;
                        }
                        break;
                    case 6:
                        switch (nq0)
                        {
                            case 7:
                                operator2D<LibUtilities::eTriangle, 6, 6, 7, 6>(
                                    inptr, outptr);
                                break;
                            case 8:
                                operator2D<LibUtilities::eTriangle, 6, 6, 8, 7>(
                                    inptr, outptr);
                                break;
                            case 9:
                                operator2D<LibUtilities::eTriangle, 6, 6, 9, 8>(
                                    inptr, outptr);
                                break;
                            case 10:
                                operator2D<LibUtilities::eTriangle, 6, 6, 10,
                                           9>(inptr, outptr);
                                break;
                            case 11:
                                operator2D<LibUtilities::eTriangle, 6, 6, 11,
                                           10>(inptr, outptr);
                                break;
                            case 12:
                                operator2D<LibUtilities::eTriangle, 6, 6, 12,
                                           11>(inptr, outptr);
                                break;
                            default:
                                operator2D<LibUtilities::eTriangle>(inptr,
                                                                    outptr);
                                break;
                        }
                        break;
                    case 7:
                        switch (nq0)
                        {
                            case 8:
                                operator2D<LibUtilities::eTriangle, 7, 7, 8, 7>(
                                    inptr, outptr);
                                break;
                            case 9:
                                operator2D<LibUtilities::eTriangle, 7, 7, 9, 8>(
                                    inptr, outptr);
                                break;
                            case 10:
                                operator2D<LibUtilities::eTriangle, 7, 7, 10,
                                           9>(inptr, outptr);
                                break;
                            case 11:
                                operator2D<LibUtilities::eTriangle, 7, 7, 11,
                                           10>(inptr, outptr);
                                break;
                            case 12:
                                operator2D<LibUtilities::eTriangle, 7, 7, 12,
                                           11>(inptr, outptr);
                                break;
                            case 13:
                                operator2D<LibUtilities::eTriangle, 7, 7, 13,
                                           12>(inptr, outptr);
                                break;
                            case 14:
                                operator2D<LibUtilities::eTriangle, 7, 7, 14,
                                           13>(inptr, outptr);
                                break;
                            default:
                                operator2D<LibUtilities::eTriangle>(inptr,
                                                                    outptr);
                                break;
                        }
                        break;
                    case 8:
                        switch (nq0)
                        {
                            case 9:
                                operator2D<LibUtilities::eTriangle, 8, 8, 9, 8>(
                                    inptr, outptr);
                                break;
                            case 10:
                                operator2D<LibUtilities::eTriangle, 8, 8, 10,
                                           9>(inptr, outptr);
                                break;
                            case 11:
                                operator2D<LibUtilities::eTriangle, 8, 8, 11,
                                           10>(inptr, outptr);
                                break;
                            case 12:
                                operator2D<LibUtilities::eTriangle, 8, 8, 12,
                                           11>(inptr, outptr);
                                break;
                            case 13:
                                operator2D<LibUtilities::eTriangle, 8, 8, 13,
                                           12>(inptr, outptr);
                                break;
                            case 14:
                                operator2D<LibUtilities::eTriangle, 8, 8, 14,
                                           13>(inptr, outptr);
                                break;
                            case 15:
                                operator2D<LibUtilities::eTriangle, 8, 8, 15,
                                           14>(inptr, outptr);
                                break;
                            case 16:
                                operator2D<LibUtilities::eTriangle, 8, 8, 16,
                                           15>(inptr, outptr);
                                break;
                            default:
                                operator2D<LibUtilities::eTriangle>(inptr,
                                                                    outptr);
                                break;
                        }
                        break;
                    default:
                        operator2D<LibUtilities::eTriangle>(inptr, outptr);
                        break;
                }
            }
            break;
        case LibUtilities::eQuadrilateral:
            if (nm0 == nm1 && nq0 == nq1)
            {
                switch (nm0)
                {
                    case 2:
                        switch (nq0)
                        {
                            case 2:
                                operator2D<LibUtilities::eQuadrilateral, 2, 2,
                                           2, 2>(inptr, outptr);
                                break;
                            case 3:
                                operator2D<LibUtilities::eQuadrilateral, 2, 2,
                                           3, 3>(inptr, outptr);
                                break;
                            case 4:
                                operator2D<LibUtilities::eQuadrilateral, 2, 2,
                                           4, 4>(inptr, outptr);
                                break;
                            default:
                                operator2D<LibUtilities::eQuadrilateral>(
                                    inptr, outptr);
                                break;
                        }
                        break;
                    case 3:
                        switch (nq0)
                        {
                            case 3:
                                operator2D<LibUtilities::eQuadrilateral, 3, 3,
                                           3, 3>(inptr, outptr);
                                break;
                            case 4:
                                operator2D<LibUtilities::eQuadrilateral, 3, 3,
                                           4, 4>(inptr, outptr);
                                break;
                            case 5:
                                operator2D<LibUtilities::eQuadrilateral, 3, 3,
                                           5, 5>(inptr, outptr);
                                break;
                            case 6:
                                operator2D<LibUtilities::eQuadrilateral, 3, 3,
                                           6, 6>(inptr, outptr);
                                break;
                            default:
                                operator2D<LibUtilities::eQuadrilateral>(
                                    inptr, outptr);
                                break;
                        }
                        break;
                    case 4:
                        switch (nq0)
                        {
                            case 4:
                                operator2D<LibUtilities::eQuadrilateral, 4, 4,
                                           4, 4>(inptr, outptr);
                                break;
                            case 5:
                                operator2D<LibUtilities::eQuadrilateral, 4, 4,
                                           5, 5>(inptr, outptr);
                                break;
                            case 6:
                                operator2D<LibUtilities::eQuadrilateral, 4, 4,
                                           6, 6>(inptr, outptr);
                                break;
                            case 7:
                                operator2D<LibUtilities::eQuadrilateral, 4, 4,
                                           7, 7>(inptr, outptr);
                                break;
                            case 8:
                                operator2D<LibUtilities::eQuadrilateral, 4, 4,
                                           8, 8>(inptr, outptr);
                                break;
                            default:
                                operator2D<LibUtilities::eQuadrilateral>(
                                    inptr, outptr);
                                break;
                        }
                        break;
                    case 5:
                        switch (nq0)
                        {
                            case 5:
                                operator2D<LibUtilities::eQuadrilateral, 5, 5,
                                           5, 5>(inptr, outptr);
                                break;
                            case 6:
                                operator2D<LibUtilities::eQuadrilateral, 5, 5,
                                           6, 6>(inptr, outptr);
                                break;
                            case 7:
                                operator2D<LibUtilities::eQuadrilateral, 5, 5,
                                           7, 7>(inptr, outptr);
                                break;
                            case 8:
                                operator2D<LibUtilities::eQuadrilateral, 5, 5,
                                           8, 8>(inptr, outptr);
                                break;
                            case 9:
                                operator2D<LibUtilities::eQuadrilateral, 5, 5,
                                           9, 9>(inptr, outptr);
                                break;
                            case 10:
                                operator2D<LibUtilities::eQuadrilateral, 5, 5,
                                           10, 10>(inptr, outptr);
                                break;
                            default:
                                operator2D<LibUtilities::eQuadrilateral>(
                                    inptr, outptr);
                                break;
                        }
                        break;
                    case 6:
                        switch (nq0)
                        {
                            case 6:
                                operator2D<LibUtilities::eQuadrilateral, 6, 6,
                                           6, 6>(inptr, outptr);
                                break;
                            case 7:
                                operator2D<LibUtilities::eQuadrilateral, 6, 6,
                                           7, 7>(inptr, outptr);
                                break;
                            case 8:
                                operator2D<LibUtilities::eQuadrilateral, 6, 6,
                                           8, 8>(inptr, outptr);
                                break;
                            case 9:
                                operator2D<LibUtilities::eQuadrilateral, 6, 6,
                                           9, 9>(inptr, outptr);
                                break;
                            case 10:
                                operator2D<LibUtilities::eQuadrilateral, 6, 6,
                                           10, 10>(inptr, outptr);
                                break;
                            case 11:
                                operator2D<LibUtilities::eQuadrilateral, 6, 6,
                                           11, 11>(inptr, outptr);
                                break;
                            case 12:
                                operator2D<LibUtilities::eQuadrilateral, 6, 6,
                                           12, 12>(inptr, outptr);
                                break;
                            default:
                                operator2D<LibUtilities::eQuadrilateral>(
                                    inptr, outptr);
                                break;
                        }
                        break;
                    case 7:
                        switch (nq0)
                        {
                            case 7:
                                operator2D<LibUtilities::eQuadrilateral, 7, 7,
                                           7, 7>(inptr, outptr);
                                break;
                            case 8:
                                operator2D<LibUtilities::eQuadrilateral, 7, 7,
                                           8, 8>(inptr, outptr);
                                break;
                            case 9:
                                operator2D<LibUtilities::eQuadrilateral, 7, 7,
                                           9, 9>(inptr, outptr);
                                break;
                            case 10:
                                operator2D<LibUtilities::eQuadrilateral, 7, 7,
                                           10, 10>(inptr, outptr);
                                break;
                            case 11:
                                operator2D<LibUtilities::eQuadrilateral, 7, 7,
                                           11, 11>(inptr, outptr);
                                break;
                            case 12:
                                operator2D<LibUtilities::eQuadrilateral, 7, 7,
                                           12, 12>(inptr, outptr);
                                break;
                            case 13:
                                operator2D<LibUtilities::eQuadrilateral, 7, 7,
                                           13, 13>(inptr, outptr);
                                break;
                            case 14:
                                operator2D<LibUtilities::eQuadrilateral, 7, 7,
                                           14, 14>(inptr, outptr);
                                break;
                            default:
                                operator2D<LibUtilities::eQuadrilateral>(
                                    inptr, outptr);
                                break;
                        }
                        break;
                    case 8:
                        switch (nq0)
                        {
                            case 8:
                                operator2D<LibUtilities::eQuadrilateral, 8, 8,
                                           8, 8>(inptr, outptr);
                                break;
                            case 9:
                                operator2D<LibUtilities::eQuadrilateral, 8, 8,
                                           9, 9>(inptr, outptr);
                                break;
                            case 10:
                                operator2D<LibUtilities::eQuadrilateral, 8, 8,
                                           10, 10>(inptr, outptr);
                                break;
                            case 11:
                                operator2D<LibUtilities::eQuadrilateral, 8, 8,
                                           11, 11>(inptr, outptr);
                                break;
                            case 12:
                                operator2D<LibUtilities::eQuadrilateral, 8, 8,
                                           12, 12>(inptr, outptr);
                                break;
                            case 13:
                                operator2D<LibUtilities::eQuadrilateral, 8, 8,
                                           13, 13>(inptr, outptr);
                                break;
                            case 14:
                                operator2D<LibUtilities::eQuadrilateral, 8, 8,
                                           14, 14>(inptr, outptr);
                                break;
                            case 15:
                                operator2D<LibUtilities::eQuadrilateral, 8, 8,
                                           15, 15>(inptr, outptr);
                                break;
                            case 16:
                                operator2D<LibUtilities::eQuadrilateral, 8, 8,
                                           16, 16>(inptr, outptr);
                                break;
                            default:
                                operator2D<LibUtilities::eQuadrilateral>(
                                    inptr, outptr);
                                break;
                        }
                        break;
                    default:
                        operator2D<LibUtilities::eQuadrilateral>(inptr, outptr);
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
    const int nm0 = m_basis[0]->GetNumModes();
    const int nm1 = m_basis[1]->GetNumModes();
    const int nm2 = m_basis[2]->GetNumModes();
    const int nq0 = m_basis[0]->GetNumPoints();
    const int nq1 = m_basis[1]->GetNumPoints();
    const int nq2 = m_basis[2]->GetNumPoints();
    switch (shapeType)
    {
        case LibUtilities::eHexahedron:
            if (nm0 == nm1 && nm0 == nm2 && nq0 == nq1 && nq0 == nq2)
            {
                switch (nm0)
                {
                    case 2:
                        switch (nq0)
                        {
                            case 2:
                                operator3D<LibUtilities::eHexahedron, 2, 2, 2,
                                           2, 2, 2>(inptr, outptr);
                                break;
                            case 3:
                                operator3D<LibUtilities::eHexahedron, 2, 2, 2,
                                           3, 3, 3>(inptr, outptr);
                                break;
                            case 4:
                                operator3D<LibUtilities::eHexahedron, 2, 2, 2,
                                           4, 4, 4>(inptr, outptr);
                                break;
                            default:
                                operator3D<LibUtilities::eHexahedron>(inptr,
                                                                      outptr);
                                break;
                        }
                        break;
                    case 3:
                        switch (nq0)
                        {
                            case 3:
                                operator3D<LibUtilities::eHexahedron, 3, 3, 3,
                                           3, 3, 3>(inptr, outptr);
                                break;
                            case 4:
                                operator3D<LibUtilities::eHexahedron, 3, 3, 3,
                                           4, 4, 4>(inptr, outptr);
                                break;
                            case 5:
                                operator3D<LibUtilities::eHexahedron, 3, 3, 3,
                                           5, 5, 5>(inptr, outptr);
                                break;
                            case 6:
                                operator3D<LibUtilities::eHexahedron, 3, 3, 3,
                                           6, 6, 6>(inptr, outptr);
                                break;
                            default:
                                operator3D<LibUtilities::eHexahedron>(inptr,
                                                                      outptr);
                                break;
                        }
                        break;
                    case 4:
                        switch (nq0)
                        {
                            case 4:
                                operator3D<LibUtilities::eHexahedron, 4, 4, 4,
                                           4, 4, 4>(inptr, outptr);
                                break;
                            case 5:
                                operator3D<LibUtilities::eHexahedron, 4, 4, 4,
                                           5, 5, 5>(inptr, outptr);
                                break;
                            case 6:
                                operator3D<LibUtilities::eHexahedron, 4, 4, 4,
                                           6, 6, 6>(inptr, outptr);
                                break;
                            case 7:
                                operator3D<LibUtilities::eHexahedron, 4, 4, 4,
                                           7, 7, 7>(inptr, outptr);
                                break;
                            case 8:
                                operator3D<LibUtilities::eHexahedron, 4, 4, 4,
                                           8, 8, 8>(inptr, outptr);
                                break;
                            default:
                                operator3D<LibUtilities::eHexahedron>(inptr,
                                                                      outptr);
                                break;
                        }
                        break;
                    case 5:
                        switch (nq0)
                        {
                            case 5:
                                operator3D<LibUtilities::eHexahedron, 5, 5, 5,
                                           5, 5, 5>(inptr, outptr);
                                break;
                            case 6:
                                operator3D<LibUtilities::eHexahedron, 5, 5, 5,
                                           6, 6, 6>(inptr, outptr);
                                break;
                            case 7:
                                operator3D<LibUtilities::eHexahedron, 5, 5, 5,
                                           7, 7, 7>(inptr, outptr);
                                break;
                            case 8:
                                operator3D<LibUtilities::eHexahedron, 5, 5, 5,
                                           8, 8, 8>(inptr, outptr);
                                break;
                            case 9:
                                operator3D<LibUtilities::eHexahedron, 5, 5, 5,
                                           9, 9, 9>(inptr, outptr);
                                break;
                            case 10:
                                operator3D<LibUtilities::eHexahedron, 5, 5, 5,
                                           10, 10, 10>(inptr, outptr);
                                break;
                            default:
                                operator3D<LibUtilities::eHexahedron>(inptr,
                                                                      outptr);
                                break;
                        }
                        break;
                    case 6:
                        switch (nq0)
                        {
                            case 6:
                                operator3D<LibUtilities::eHexahedron, 6, 6, 6,
                                           6, 6, 6>(inptr, outptr);
                                break;
                            case 7:
                                operator3D<LibUtilities::eHexahedron, 6, 6, 6,
                                           7, 7, 7>(inptr, outptr);
                                break;
                            case 8:
                                operator3D<LibUtilities::eHexahedron, 6, 6, 6,
                                           8, 8, 8>(inptr, outptr);
                                break;
                            case 9:
                                operator3D<LibUtilities::eHexahedron, 6, 6, 6,
                                           9, 9, 9>(inptr, outptr);
                                break;
                            case 10:
                                operator3D<LibUtilities::eHexahedron, 6, 6, 6,
                                           10, 10, 10>(inptr, outptr);
                                break;
                            case 11:
                                operator3D<LibUtilities::eHexahedron, 6, 6, 6,
                                           11, 11, 11>(inptr, outptr);
                                break;
                            case 12:
                                operator3D<LibUtilities::eHexahedron, 6, 6, 6,
                                           12, 12, 12>(inptr, outptr);
                                break;
                            default:
                                operator3D<LibUtilities::eHexahedron>(inptr,
                                                                      outptr);
                                break;
                        }
                        break;
                    case 7:
                        switch (nq0)
                        {
                            case 7:
                                operator3D<LibUtilities::eHexahedron, 7, 7, 7,
                                           7, 7, 7>(inptr, outptr);
                                break;
                            case 8:
                                operator3D<LibUtilities::eHexahedron, 7, 7, 7,
                                           8, 8, 8>(inptr, outptr);
                                break;
                            case 9:
                                operator3D<LibUtilities::eHexahedron, 7, 7, 7,
                                           9, 9, 9>(inptr, outptr);
                                break;
                            case 10:
                                operator3D<LibUtilities::eHexahedron, 7, 7, 7,
                                           10, 10, 10>(inptr, outptr);
                                break;
                            case 11:
                                operator3D<LibUtilities::eHexahedron, 7, 7, 7,
                                           11, 11, 11>(inptr, outptr);
                                break;
                            case 12:
                                operator3D<LibUtilities::eHexahedron, 7, 7, 7,
                                           12, 12, 12>(inptr, outptr);
                                break;
                            case 13:
                                operator3D<LibUtilities::eHexahedron, 7, 7, 7,
                                           13, 13, 13>(inptr, outptr);
                                break;
                            case 14:
                                operator3D<LibUtilities::eHexahedron, 7, 7, 7,
                                           14, 14, 14>(inptr, outptr);
                                break;
                            default:
                                operator3D<LibUtilities::eHexahedron>(inptr,
                                                                      outptr);
                                break;
                        }
                        break;
                    case 8:
                        switch (nq0)
                        {
                            case 8:
                                operator3D<LibUtilities::eHexahedron, 8, 8, 8,
                                           8, 8, 8>(inptr, outptr);
                                break;
                            case 9:
                                operator3D<LibUtilities::eHexahedron, 8, 8, 8,
                                           9, 9, 9>(inptr, outptr);
                                break;
                            case 10:
                                operator3D<LibUtilities::eHexahedron, 8, 8, 8,
                                           10, 10, 10>(inptr, outptr);
                                break;
                            case 11:
                                operator3D<LibUtilities::eHexahedron, 8, 8, 8,
                                           11, 11, 11>(inptr, outptr);
                                break;
                            case 12:
                                operator3D<LibUtilities::eHexahedron, 8, 8, 8,
                                           12, 12, 12>(inptr, outptr);
                                break;
                            case 13:
                                operator3D<LibUtilities::eHexahedron, 8, 8, 8,
                                           13, 13, 13>(inptr, outptr);
                                break;
                            case 14:
                                operator3D<LibUtilities::eHexahedron, 8, 8, 8,
                                           14, 14, 14>(inptr, outptr);
                                break;
                            case 15:
                                operator3D<LibUtilities::eHexahedron, 8, 8, 8,
                                           15, 15, 15>(inptr, outptr);
                                break;
                            case 16:
                                operator3D<LibUtilities::eHexahedron, 8, 8, 8,
                                           16, 16, 16>(inptr, outptr);
                                break;
                            default:
                                operator3D<LibUtilities::eHexahedron>(inptr,
                                                                      outptr);
                                break;
                        }
                        break;
                    default:
                        operator3D<LibUtilities::eHexahedron>(inptr, outptr);
                        break;
                }
            }
            break;
        case LibUtilities::eTetrahedron:
            if (nm0 == nm1 && nm0 == nm2 && nq0 == nq1 + 1 && nq0 == nq2 + 1)
            {
                switch (nm0)
                {
                    case 2:
                        switch (nq0)
                        {
                            case 3:
                                operator3D<LibUtilities::eTetrahedron, 2, 2, 2,
                                           3, 2, 2>(inptr, outptr);
                                break;
                            case 4:
                                operator3D<LibUtilities::eTetrahedron, 2, 2, 2,
                                           4, 3, 3>(inptr, outptr);
                                break;
                            default:
                                operator3D<LibUtilities::eTetrahedron>(inptr,
                                                                       outptr);
                                break;
                        }
                        break;
                    case 3:
                        switch (nq0)
                        {
                            case 4:
                                operator3D<LibUtilities::eTetrahedron, 3, 3, 3,
                                           4, 3, 3>(inptr, outptr);
                                break;
                            case 5:
                                operator3D<LibUtilities::eTetrahedron, 3, 3, 3,
                                           5, 4, 4>(inptr, outptr);
                                break;
                            case 6:
                                operator3D<LibUtilities::eTetrahedron, 3, 3, 3,
                                           6, 5, 5>(inptr, outptr);
                                break;
                            default:
                                operator3D<LibUtilities::eTetrahedron>(inptr,
                                                                       outptr);
                                break;
                        }
                        break;
                    case 4:
                        switch (nq0)
                        {
                            case 5:
                                operator3D<LibUtilities::eTetrahedron, 4, 4, 4,
                                           5, 4, 4>(inptr, outptr);
                                break;
                            case 6:
                                operator3D<LibUtilities::eTetrahedron, 4, 4, 4,
                                           6, 5, 5>(inptr, outptr);
                                break;
                            case 7:
                                operator3D<LibUtilities::eTetrahedron, 4, 4, 4,
                                           7, 6, 6>(inptr, outptr);
                                break;
                            case 8:
                                operator3D<LibUtilities::eTetrahedron, 4, 4, 4,
                                           8, 7, 7>(inptr, outptr);
                                break;
                            default:
                                operator3D<LibUtilities::eTetrahedron>(inptr,
                                                                       outptr);
                                break;
                        }
                        break;
                    case 5:
                        switch (nq0)
                        {
                            case 6:
                                operator3D<LibUtilities::eTetrahedron, 5, 5, 5,
                                           6, 5, 5>(inptr, outptr);
                                break;
                            case 7:
                                operator3D<LibUtilities::eTetrahedron, 5, 5, 5,
                                           7, 6, 6>(inptr, outptr);
                                break;
                            case 8:
                                operator3D<LibUtilities::eTetrahedron, 5, 5, 5,
                                           8, 7, 7>(inptr, outptr);
                                break;
                            case 9:
                                operator3D<LibUtilities::eTetrahedron, 5, 5, 5,
                                           9, 8, 8>(inptr, outptr);
                                break;
                            case 10:
                                operator3D<LibUtilities::eTetrahedron, 5, 5, 5,
                                           10, 9, 9>(inptr, outptr);
                                break;
                            default:
                                operator3D<LibUtilities::eTetrahedron>(inptr,
                                                                       outptr);
                                break;
                        }
                        break;
                    case 6:
                        switch (nq0)
                        {
                            case 7:
                                operator3D<LibUtilities::eTetrahedron, 6, 6, 6,
                                           7, 6, 6>(inptr, outptr);
                                break;
                            case 8:
                                operator3D<LibUtilities::eTetrahedron, 6, 6, 6,
                                           8, 7, 7>(inptr, outptr);
                                break;
                            case 9:
                                operator3D<LibUtilities::eTetrahedron, 6, 6, 6,
                                           9, 8, 8>(inptr, outptr);
                                break;
                            case 10:
                                operator3D<LibUtilities::eTetrahedron, 6, 6, 6,
                                           10, 9, 9>(inptr, outptr);
                                break;
                            case 11:
                                operator3D<LibUtilities::eTetrahedron, 6, 6, 6,
                                           11, 10, 10>(inptr, outptr);
                                break;
                            case 12:
                                operator3D<LibUtilities::eTetrahedron, 6, 6, 6,
                                           12, 11, 11>(inptr, outptr);
                                break;
                            default:
                                operator3D<LibUtilities::eTetrahedron>(inptr,
                                                                       outptr);
                                break;
                        }
                        break;
                    case 7:
                        switch (nq0)
                        {
                            case 8:
                                operator3D<LibUtilities::eTetrahedron, 7, 7, 7,
                                           8, 7, 7>(inptr, outptr);
                                break;
                            case 9:
                                operator3D<LibUtilities::eTetrahedron, 7, 7, 7,
                                           9, 8, 8>(inptr, outptr);
                                break;
                            case 10:
                                operator3D<LibUtilities::eTetrahedron, 7, 7, 7,
                                           10, 9, 9>(inptr, outptr);
                                break;
                            case 11:
                                operator3D<LibUtilities::eTetrahedron, 7, 7, 7,
                                           11, 10, 10>(inptr, outptr);
                                break;
                            case 12:
                                operator3D<LibUtilities::eTetrahedron, 7, 7, 7,
                                           12, 11, 11>(inptr, outptr);
                                break;
                            case 13:
                                operator3D<LibUtilities::eTetrahedron, 7, 7, 7,
                                           13, 12, 12>(inptr, outptr);
                                break;
                            case 14:
                                operator3D<LibUtilities::eTetrahedron, 7, 7, 7,
                                           14, 13, 13>(inptr, outptr);
                                break;
                            default:
                                operator3D<LibUtilities::eTetrahedron>(inptr,
                                                                       outptr);
                                break;
                        }
                        break;
                    case 8:
                        switch (nq0)
                        {
                            case 9:
                                operator3D<LibUtilities::eTetrahedron, 8, 8, 8,
                                           9, 8, 8>(inptr, outptr);
                                break;
                            case 10:
                                operator3D<LibUtilities::eTetrahedron, 8, 8, 8,
                                           10, 9, 9>(inptr, outptr);
                                break;
                            case 11:
                                operator3D<LibUtilities::eTetrahedron, 8, 8, 8,
                                           11, 10, 10>(inptr, outptr);
                                break;
                            case 12:
                                operator3D<LibUtilities::eTetrahedron, 8, 8, 8,
                                           12, 11, 11>(inptr, outptr);
                                break;
                            case 13:
                                operator3D<LibUtilities::eTetrahedron, 8, 8, 8,
                                           13, 12, 12>(inptr, outptr);
                                break;
                            case 14:
                                operator3D<LibUtilities::eTetrahedron, 8, 8, 8,
                                           14, 13, 13>(inptr, outptr);
                                break;
                            case 15:
                                operator3D<LibUtilities::eTetrahedron, 8, 8, 8,
                                           15, 14, 14>(inptr, outptr);
                                break;
                            case 16:
                                operator3D<LibUtilities::eTetrahedron, 8, 8, 8,
                                           16, 15, 15>(inptr, outptr);
                                break;
                            default:
                                operator3D<LibUtilities::eTetrahedron>(inptr,
                                                                       outptr);
                                break;
                        }
                        break;
                    default:
                        operator3D<LibUtilities::eTetrahedron>(inptr, outptr);
                        break;
                }
            }
            break;
        case LibUtilities::ePrism:
            if (nm0 == nm1 && nm0 == nm2 && nq0 == nq1 && nq0 == nq2 + 1)
            {
                switch (nm0)
                {
                    case 2:
                        switch (nq0)
                        {
                            case 3:
                                operator3D<LibUtilities::ePrism, 2, 2, 2, 3, 3,
                                           2>(inptr, outptr);
                                break;
                            case 4:
                                operator3D<LibUtilities::ePrism, 2, 2, 2, 4, 4,
                                           3>(inptr, outptr);
                                break;
                            default:
                                operator3D<LibUtilities::ePrism>(inptr, outptr);
                                break;
                        }
                        break;
                    case 3:
                        switch (nq0)
                        {
                            case 4:
                                operator3D<LibUtilities::ePrism, 3, 3, 3, 4, 4,
                                           3>(inptr, outptr);
                                break;
                            case 5:
                                operator3D<LibUtilities::ePrism, 3, 3, 3, 5, 5,
                                           4>(inptr, outptr);
                                break;
                            case 6:
                                operator3D<LibUtilities::ePrism, 3, 3, 3, 6, 6,
                                           5>(inptr, outptr);
                                break;
                            default:
                                operator3D<LibUtilities::ePrism>(inptr, outptr);
                                break;
                        }
                        break;
                    case 4:
                        switch (nq0)
                        {
                            case 5:
                                operator3D<LibUtilities::ePrism, 4, 4, 4, 5, 5,
                                           4>(inptr, outptr);
                                break;
                            case 6:
                                operator3D<LibUtilities::ePrism, 4, 4, 4, 6, 6,
                                           5>(inptr, outptr);
                                break;
                            case 7:
                                operator3D<LibUtilities::ePrism, 4, 4, 4, 7, 7,
                                           6>(inptr, outptr);
                                break;
                            case 8:
                                operator3D<LibUtilities::ePrism, 4, 4, 4, 8, 8,
                                           7>(inptr, outptr);
                                break;
                            default:
                                operator3D<LibUtilities::ePrism>(inptr, outptr);
                                break;
                        }
                        break;
                    case 5:
                        switch (nq0)
                        {
                            case 6:
                                operator3D<LibUtilities::ePrism, 5, 5, 5, 6, 6,
                                           5>(inptr, outptr);
                                break;
                            case 7:
                                operator3D<LibUtilities::ePrism, 5, 5, 5, 7, 7,
                                           6>(inptr, outptr);
                                break;
                            case 8:
                                operator3D<LibUtilities::ePrism, 5, 5, 5, 8, 8,
                                           7>(inptr, outptr);
                                break;
                            case 9:
                                operator3D<LibUtilities::ePrism, 5, 5, 5, 9, 9,
                                           8>(inptr, outptr);
                                break;
                            case 10:
                                operator3D<LibUtilities::ePrism, 5, 5, 5, 10,
                                           10, 9>(inptr, outptr);
                                break;
                            default:
                                operator3D<LibUtilities::ePrism>(inptr, outptr);
                                break;
                        }
                        break;
                    case 6:
                        switch (nq0)
                        {
                            case 7:
                                operator3D<LibUtilities::ePrism, 6, 6, 6, 7, 7,
                                           6>(inptr, outptr);
                                break;
                            case 8:
                                operator3D<LibUtilities::ePrism, 6, 6, 6, 8, 8,
                                           7>(inptr, outptr);
                                break;
                            case 9:
                                operator3D<LibUtilities::ePrism, 6, 6, 6, 9, 9,
                                           8>(inptr, outptr);
                                break;
                            case 10:
                                operator3D<LibUtilities::ePrism, 6, 6, 6, 10,
                                           10, 9>(inptr, outptr);
                                break;
                            case 11:
                                operator3D<LibUtilities::ePrism, 6, 6, 6, 11,
                                           11, 10>(inptr, outptr);
                                break;
                            case 12:
                                operator3D<LibUtilities::ePrism, 6, 6, 6, 12,
                                           12, 11>(inptr, outptr);
                                break;
                            default:
                                operator3D<LibUtilities::ePrism>(inptr, outptr);
                                break;
                        }
                        break;
                    case 7:
                        switch (nq0)
                        {
                            case 8:
                                operator3D<LibUtilities::ePrism, 7, 7, 7, 8, 8,
                                           7>(inptr, outptr);
                                break;
                            case 9:
                                operator3D<LibUtilities::ePrism, 7, 7, 7, 9, 9,
                                           8>(inptr, outptr);
                                break;
                            case 10:
                                operator3D<LibUtilities::ePrism, 7, 7, 7, 10,
                                           10, 9>(inptr, outptr);
                                break;
                            case 11:
                                operator3D<LibUtilities::ePrism, 7, 7, 7, 11,
                                           11, 10>(inptr, outptr);
                                break;
                            case 12:
                                operator3D<LibUtilities::ePrism, 7, 7, 7, 12,
                                           12, 11>(inptr, outptr);
                                break;
                            case 13:
                                operator3D<LibUtilities::ePrism, 7, 7, 7, 13,
                                           13, 12>(inptr, outptr);
                                break;
                            case 14:
                                operator3D<LibUtilities::ePrism, 7, 7, 7, 14,
                                           14, 13>(inptr, outptr);
                                break;
                            default:
                                operator3D<LibUtilities::ePrism>(inptr, outptr);
                                break;
                        }
                        break;
                    case 8:
                        switch (nq0)
                        {
                            case 9:
                                operator3D<LibUtilities::ePrism, 8, 8, 8, 9, 9,
                                           8>(inptr, outptr);
                                break;
                            case 10:
                                operator3D<LibUtilities::ePrism, 8, 8, 8, 10,
                                           10, 9>(inptr, outptr);
                                break;
                            case 11:
                                operator3D<LibUtilities::ePrism, 8, 8, 8, 11,
                                           11, 10>(inptr, outptr);
                                break;
                            case 12:
                                operator3D<LibUtilities::ePrism, 8, 8, 8, 12,
                                           12, 11>(inptr, outptr);
                                break;
                            case 13:
                                operator3D<LibUtilities::ePrism, 8, 8, 8, 13,
                                           13, 12>(inptr, outptr);
                                break;
                            case 14:
                                operator3D<LibUtilities::ePrism, 8, 8, 8, 14,
                                           14, 13>(inptr, outptr);
                                break;
                            case 15:
                                operator3D<LibUtilities::ePrism, 8, 8, 8, 15,
                                           15, 14>(inptr, outptr);
                                break;
                            case 16:
                                operator3D<LibUtilities::ePrism, 8, 8, 8, 16,
                                           16, 15>(inptr, outptr);
                                break;
                            default:
                                operator3D<LibUtilities::ePrism>(inptr, outptr);
                                break;
                        }
                        break;
                    default:
                        operator3D<LibUtilities::ePrism>(inptr, outptr);
                        break;
                }
            }
            break;
        case LibUtilities::ePyramid:
            if (nm0 == nm1 && nm0 == nm2 && nq0 == nq1 && nq0 == nq2 + 1)
            {
                switch (nm0)
                {
                    case 2:
                        switch (nq0)
                        {
                            case 3:
                                operator3D<LibUtilities::ePyramid, 2, 2, 2, 3,
                                           3, 2>(inptr, outptr);
                                break;
                            case 4:
                                operator3D<LibUtilities::ePyramid, 2, 2, 2, 4,
                                           4, 3>(inptr, outptr);
                                break;
                            default:
                                operator3D<LibUtilities::ePyramid>(inptr,
                                                                   outptr);
                                break;
                        }
                        break;
                    case 3:
                        switch (nq0)
                        {
                            case 4:
                                operator3D<LibUtilities::ePyramid, 3, 3, 3, 4,
                                           4, 3>(inptr, outptr);
                                break;
                            case 5:
                                operator3D<LibUtilities::ePyramid, 3, 3, 3, 5,
                                           5, 4>(inptr, outptr);
                                break;
                            case 6:
                                operator3D<LibUtilities::ePyramid, 3, 3, 3, 6,
                                           6, 5>(inptr, outptr);
                                break;
                            default:
                                operator3D<LibUtilities::ePyramid>(inptr,
                                                                   outptr);
                                break;
                        }
                        break;
                    case 4:
                        switch (nq0)
                        {
                            case 5:
                                operator3D<LibUtilities::ePyramid, 4, 4, 4, 5,
                                           5, 4>(inptr, outptr);
                                break;
                            case 6:
                                operator3D<LibUtilities::ePyramid, 4, 4, 4, 6,
                                           6, 5>(inptr, outptr);
                                break;
                            case 7:
                                operator3D<LibUtilities::ePyramid, 4, 4, 4, 7,
                                           7, 6>(inptr, outptr);
                                break;
                            case 8:
                                operator3D<LibUtilities::ePyramid, 4, 4, 4, 8,
                                           8, 7>(inptr, outptr);
                                break;
                            default:
                                operator3D<LibUtilities::ePyramid>(inptr,
                                                                   outptr);
                                break;
                        }
                        break;
                    case 5:
                        switch (nq0)
                        {
                            case 6:
                                operator3D<LibUtilities::ePyramid, 5, 5, 5, 6,
                                           6, 5>(inptr, outptr);
                                break;
                            case 7:
                                operator3D<LibUtilities::ePyramid, 5, 5, 5, 7,
                                           7, 6>(inptr, outptr);
                                break;
                            case 8:
                                operator3D<LibUtilities::ePyramid, 5, 5, 5, 8,
                                           8, 7>(inptr, outptr);
                                break;
                            case 9:
                                operator3D<LibUtilities::ePyramid, 5, 5, 5, 9,
                                           9, 8>(inptr, outptr);
                                break;
                            case 10:
                                operator3D<LibUtilities::ePyramid, 5, 5, 5, 10,
                                           10, 9>(inptr, outptr);
                                break;
                            default:
                                operator3D<LibUtilities::ePyramid>(inptr,
                                                                   outptr);
                                break;
                        }
                        break;
                    case 6:
                        switch (nq0)
                        {
                            case 7:
                                operator3D<LibUtilities::ePyramid, 6, 6, 6, 7,
                                           7, 6>(inptr, outptr);
                                break;
                            case 8:
                                operator3D<LibUtilities::ePyramid, 6, 6, 6, 8,
                                           8, 7>(inptr, outptr);
                                break;
                            case 9:
                                operator3D<LibUtilities::ePyramid, 6, 6, 6, 9,
                                           9, 8>(inptr, outptr);
                                break;
                            case 10:
                                operator3D<LibUtilities::ePyramid, 6, 6, 6, 10,
                                           10, 9>(inptr, outptr);
                                break;
                            case 11:
                                operator3D<LibUtilities::ePyramid, 6, 6, 6, 11,
                                           11, 10>(inptr, outptr);
                                break;
                            case 12:
                                operator3D<LibUtilities::ePyramid, 6, 6, 6, 12,
                                           12, 11>(inptr, outptr);
                                break;
                            default:
                                operator3D<LibUtilities::ePyramid>(inptr,
                                                                   outptr);
                                break;
                        }
                        break;
                    case 7:
                        switch (nq0)
                        {
                            case 8:
                                operator3D<LibUtilities::ePyramid, 7, 7, 7, 8,
                                           8, 7>(inptr, outptr);
                                break;
                            case 9:
                                operator3D<LibUtilities::ePyramid, 7, 7, 7, 9,
                                           9, 8>(inptr, outptr);
                                break;
                            case 10:
                                operator3D<LibUtilities::ePyramid, 7, 7, 7, 10,
                                           10, 9>(inptr, outptr);
                                break;
                            case 11:
                                operator3D<LibUtilities::ePyramid, 7, 7, 7, 11,
                                           11, 10>(inptr, outptr);
                                break;
                            case 12:
                                operator3D<LibUtilities::ePyramid, 7, 7, 7, 12,
                                           12, 11>(inptr, outptr);
                                break;
                            case 13:
                                operator3D<LibUtilities::ePyramid, 7, 7, 7, 13,
                                           13, 12>(inptr, outptr);
                                break;
                            case 14:
                                operator3D<LibUtilities::ePyramid, 7, 7, 7, 14,
                                           14, 13>(inptr, outptr);
                                break;
                            default:
                                operator3D<LibUtilities::ePyramid>(inptr,
                                                                   outptr);
                                break;
                        }
                        break;
                    case 8:
                        switch (nq0)
                        {
                            case 9:
                                operator3D<LibUtilities::ePyramid, 8, 8, 8, 9,
                                           9, 8>(inptr, outptr);
                                break;
                            case 10:
                                operator3D<LibUtilities::ePyramid, 8, 8, 8, 10,
                                           10, 9>(inptr, outptr);
                                break;
                            case 11:
                                operator3D<LibUtilities::ePyramid, 8, 8, 8, 11,
                                           11, 10>(inptr, outptr);
                                break;
                            case 12:
                                operator3D<LibUtilities::ePyramid, 8, 8, 8, 12,
                                           12, 11>(inptr, outptr);
                                break;
                            case 13:
                                operator3D<LibUtilities::ePyramid, 8, 8, 8, 13,
                                           13, 12>(inptr, outptr);
                                break;
                            case 14:
                                operator3D<LibUtilities::ePyramid, 8, 8, 8, 14,
                                           14, 13>(inptr, outptr);
                                break;
                            case 15:
                                operator3D<LibUtilities::ePyramid, 8, 8, 8, 15,
                                           15, 14>(inptr, outptr);
                                break;
                            case 16:
                                operator3D<LibUtilities::ePyramid, 8, 8, 8, 16,
                                           16, 15>(inptr, outptr);
                                break;
                            default:
                                operator3D<LibUtilities::ePyramid>(inptr,
                                                                   outptr);
                                break;
                        }
                        break;
                    default:
                        operator3D<LibUtilities::ePyramid>(inptr, outptr);
                        break;
                }
            }
            break;
        default:
            std::cout << "ERROR: Unknown shapeType" << std::endl;
            break;
    }
}