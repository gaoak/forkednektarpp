///////////////////////////////////////////////////////////////////////////////
//
// File: PhysInterp1DScaled.cpp
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
// Description: PhysInterp1DScaled operator implementations
//
///////////////////////////////////////////////////////////////////////////////

#include <Collections/CoalescedGeomData.h>
#include <Collections/MatrixFreeBase.h>
#include <Collections/Operator.h>
#include <LibUtilities/Foundations/Interp.h>
#include <LibUtilities/Foundations/ManagerAccess.h>
#include <MatrixFreeOps/Operator.hpp>

using namespace std;

namespace Nektar::Collections
{

using LibUtilities::eHexahedron;
using LibUtilities::ePrism;
using LibUtilities::ePyramid;
using LibUtilities::eQuadrilateral;
using LibUtilities::eSegment;
using LibUtilities::eTetrahedron;
using LibUtilities::eTriangle;

/**
 * @brief PhysInterp1DScaled help class to calculate the size of the collection
 * that is given as an input and as an output to the PhysInterp1DScaled
 * Operator. The size evaluation takes into account that both the input and the
 * output array belong to the physical space and that the output array can have
 * either a larger, or a smaller size than the input array
 */
class PhysInterp1DScaled_Helper : virtual public Operator
{
public:
    void UpdateFactors(StdRegions::FactorMap factors,
                       [[maybe_unused]] int coll_phys_offset) override
    {
        if (factors == m_factors)
        {
            return;
        }
        m_factors = factors;

        // Set scaling factor for the PhysInterp1DScaled function
        [[maybe_unused]] auto x = factors.find(StdRegions::eFactorConst);
        ASSERTL1(
            x != factors.end(),
            "Constant factor not defined: " +
                std::string(
                    StdRegions::ConstFactorTypeMap[StdRegions::eFactorConst]));
    }

protected:
    PhysInterp1DScaled_Helper()
    {
        NekDouble scale; // declaration of the scaling factor to be used for the
                         // output size
        if (m_factors[StdRegions::eFactorConst] != 0)
        {
            scale = m_factors[StdRegions::eFactorConst];
        }
        else
        {
            scale = 1.5;
        }

        // expect input to be number of elements by the number of quad points
        m_inputSize = m_numElmt * m_stdExp->GetTotPoints();
        // expect input to be number of elements by the number of quad points
        int shape_dimension = m_stdExp->GetShapeDimension();
        m_outputSize        = m_numElmt; // initializing m_outputSize
        for (int i = 0; i < shape_dimension; ++i)
        {
            m_outputSize *= (int)(m_stdExp->GetNumPoints(i) * scale);
        }
    }
    StdRegions::FactorMap
        m_factors; // map for storing the scaling factor of the operator
};

/**
 * @brief PhysInterp1DScaled operator using standard matrix approach. Currently,
 * it is an identical copy of the BwdTrans operator.
 */

class PhysInterp1DScaled_StdMat final : virtual public Operator,
                                        PhysInterp1DScaled_Helper
{
public:
    OPERATOR_CREATE(PhysInterp1DScaled_StdMat)
    ~PhysInterp1DScaled_StdMat() final
    {
    }

    void operator()(const Array<OneD, const NekDouble> &input,
                    Array<OneD, NekDouble> &output,
                    [[maybe_unused]] Array<OneD, NekDouble> &output1,
                    [[maybe_unused]] Array<OneD, NekDouble> &output2,
                    [[maybe_unused]] Array<OneD, NekDouble> &wsp) override
    {
        Blas::Dgemm('N', 'N', m_mat->GetRows(), m_numElmt, m_mat->GetColumns(),
                    1.0, m_mat->GetRawPtr(), m_mat->GetRows(), input.get(),
                    m_stdExp->GetNcoeffs(), 0.0, output.get(),
                    m_stdExp->GetTotPoints());
    }

    void operator()([[maybe_unused]] int dir,
                    [[maybe_unused]] const Array<OneD, const NekDouble> &input,
                    [[maybe_unused]] Array<OneD, NekDouble> &output,
                    [[maybe_unused]] Array<OneD, NekDouble> &wsp) final
    {
        ASSERTL0(false, "Not valid for this operator.");
    }

    void UpdateFactors(StdRegions::FactorMap factors,
                       [[maybe_unused]] int coll_phys_offset) override
    {
        if (factors == m_factors)
        {
            return;
        }

        m_factors = factors;

        // Set scaling factor for the PhysInterp1DScaled function
        [[maybe_unused]] auto x = factors.find(StdRegions::eFactorConst);
        ASSERTL1(
            x != factors.end(),
            "Constant factor not defined: " +
                std::string(
                    StdRegions::ConstFactorTypeMap[StdRegions::eFactorConst]));
    }

protected:
    DNekMatSharedPtr m_mat;
    StdRegions::FactorMap m_factors; // Initially, PhysInterp1DScaled was taking
                                     // as input a single scaling factor
    // This single scaling factor will be expressed into a matrix where each
    // element has the same value so that matrix - vector multiplication
    // optimization by BLAS is implemented

private:
    PhysInterp1DScaled_StdMat(
        vector<StdRegions::StdExpansionSharedPtr> pCollExp,
        CoalescedGeomDataSharedPtr pGeomData, StdRegions::FactorMap factors)
        : Operator(pCollExp, pGeomData, factors), PhysInterp1DScaled_Helper()
    {
    }
};

/**
 * @brief PhysInterp1DScaled operator using matrix free operators.
 */

class PhysInterp1DScaled_MatrixFree final : virtual public Operator,
                                            MatrixFreeOneInOneOut,
                                            PhysInterp1DScaled_Helper
{
public:
    OPERATOR_CREATE(PhysInterp1DScaled_MatrixFree)

    ~PhysInterp1DScaled_MatrixFree() final
    {
    }

    void operator()(const Array<OneD, const NekDouble> &input,
                    Array<OneD, NekDouble> &output0,
                    [[maybe_unused]] Array<OneD, NekDouble> &output1,
                    [[maybe_unused]] Array<OneD, NekDouble> &output2,
                    [[maybe_unused]] Array<OneD, NekDouble> &wsp) final
    {
        if (m_isPadded)
        {
            // copy into padded vector
            Vmath::Vcopy(m_nIn, input, 1, m_input, 1);
            // call op
            (*m_oper)(m_input, m_output);
            // copy out of padded vector
            Vmath::Vcopy(m_nOut, m_output, 1, output0, 1);
        }
        else
        {
            (*m_oper)(input, output0);
        }
    }

    void operator()([[maybe_unused]] int dir,
                    [[maybe_unused]] const Array<OneD, const NekDouble> &input,
                    [[maybe_unused]] Array<OneD, NekDouble> &output,
                    [[maybe_unused]] Array<OneD, NekDouble> &wsp) final
    {
        NEKERROR(ErrorUtil::efatal,
                 "BwdTrans_MatrixFree: Not valid for this operator.");
    }

    void UpdateFactors([[maybe_unused]] StdRegions::FactorMap factors,
                       [[maybe_unused]] int coll_phys_offset) override
    {
        ASSERTL0(false, "Not valid for this operator.");
    }

private:
    std::shared_ptr<MatrixFree::PhysInterp1DScaled> m_oper;

    PhysInterp1DScaled_MatrixFree(
        vector<StdRegions::StdExpansionSharedPtr> pCollExp,
        CoalescedGeomDataSharedPtr pGeomData, StdRegions::FactorMap factors)
        : Operator(pCollExp, pGeomData, factors),
          MatrixFreeOneInOneOut(pCollExp[0]->GetStdExp()->GetNcoeffs(),
                                pCollExp[0]->GetStdExp()->GetTotPoints(),
                                pCollExp.size()),
          PhysInterp1DScaled_Helper()
    {
        // Basis vector.
        const auto dim = pCollExp[0]->GetStdExp()->GetShapeDimension();
        std::vector<LibUtilities::BasisSharedPtr> basis(dim);
        for (auto i = 0; i < dim; ++i)
        {
            basis[i] = pCollExp[0]->GetBasis(i);
        }

        // Get shape type
        auto shapeType = pCollExp[0]->GetStdExp()->DetShapeType();

        // Generate operator string and create operator.
        std::string op_string = "PhysInterp1DScaled";
        op_string += MatrixFree::GetOpstring(shapeType, false);
        auto oper = MatrixFree::GetOperatorFactory().CreateInstance(
            op_string, basis, m_nElmtPad);

        m_oper =
            std::dynamic_pointer_cast<MatrixFree::PhysInterp1DScaled>(oper);
        ASSERTL0(m_oper, "Failed to cast pointer.");
    }
};

/**
 * @brief PhysInterp1DScaled operator using default StdRegions operator
 */

class PhysInterp1DScaled_IterPerExp final : virtual public Operator,
                                            PhysInterp1DScaled_Helper
{
public:
    OPERATOR_CREATE(PhysInterp1DScaled_IterPerExp)

    ~PhysInterp1DScaled_IterPerExp() final
    {
    }

    void operator()(const Array<OneD, const NekDouble> &input,
                    Array<OneD, NekDouble> &output,
                    [[maybe_unused]] Array<OneD, NekDouble> &output1,
                    [[maybe_unused]] Array<OneD, NekDouble> &output2,
                    [[maybe_unused]] Array<OneD, NekDouble> &wsp) override
    {
        const int nCoeffs = m_stdExp->GetNcoeffs();
        const int nPhys   = m_stdExp->GetTotPoints();
        Array<OneD, NekDouble> tmp;

        for (int i = 0; i < m_numElmt; ++i)
        {
            m_stdExp->BwdTrans(input + i * nCoeffs, tmp = output + i * nPhys);
        }
    }

    void operator()([[maybe_unused]] int dir,
                    [[maybe_unused]] const Array<OneD, const NekDouble> &input,
                    [[maybe_unused]] Array<OneD, NekDouble> &output,
                    [[maybe_unused]] Array<OneD, NekDouble> &wsp) final
    {
        ASSERTL0(false, "Not valid for this operator.");
    }

    void UpdateFactors([[maybe_unused]] StdRegions::FactorMap factors,
                       [[maybe_unused]] int coll_phys_offset) override
    {
        m_factors = factors;
    }

protected:
    StdRegions::FactorMap m_factors; // Initially, PhysInterp1DScaled was taking
                                     // as input a single scaling factor
    // This single scaling factor will be expressed into a matrix where each
    // element has the same value so that matrix - vector multiplication
    // optimization by BLAS is implemented

private:
    PhysInterp1DScaled_IterPerExp(
        vector<StdRegions::StdExpansionSharedPtr> pCollExp,
        CoalescedGeomDataSharedPtr pGeomData, StdRegions::FactorMap factors)
        : Operator(pCollExp, pGeomData, factors), PhysInterp1DScaled_Helper()
    {
    }
};

/**
 * @brief PhysInterp1DScaled operator using LocalRegions implementation.
 */
class PhysInterp1DScaled_NoCollection final : virtual public Operator,
                                              PhysInterp1DScaled_Helper
{
public:
    OPERATOR_CREATE(PhysInterp1DScaled_NoCollection)

    ~PhysInterp1DScaled_NoCollection() final
    {
    }

    void operator()(const Array<OneD, const NekDouble> &input,
                    Array<OneD, NekDouble> &output,
                    [[maybe_unused]] Array<OneD, NekDouble> &output1,
                    [[maybe_unused]] Array<OneD, NekDouble> &output2,
                    [[maybe_unused]] Array<OneD, NekDouble> &wsp) override
    {
        int cnt{0};
        int cnt1{0};
        const NekDouble scale = m_factors[StdRegions::eFactorConst];
        int dim{m_expList[0]->GetShapeDimension()}; // same as m_expType
        switch (dim)
        {
            case 1:
            {
                DNekMatSharedPtr I0;
                // the number of points before and after interpolation are the
                // same for each element inside a single collection
                int pt0  = m_expList[0]->GetNumPoints(0);
                int npt0 = (int)pt0 * scale;
                for (int i = 0; i < m_numElmt; ++i)
                {
                    // current points key
                    LibUtilities::PointsKey PointsKey0(
                        pt0, m_expList[i]->GetPointsType(0));
                    // get new points key
                    LibUtilities::PointsKey newPointsKey0(
                        npt0, m_expList[i]->GetPointsType(0));
                    // Interpolate points;
                    I0 = LibUtilities::PointsManager()[PointsKey0]->GetI(
                        newPointsKey0);

                    Blas::Dgemv('N', npt0, pt0, 1.0, I0->GetPtr().get(), npt0,
                                &input[cnt], 1, 0.0, &output[cnt1], 1);
                    cnt += pt0;
                    cnt1 += npt0;
                }
            }
            break;
            case 2:
            {
                DNekMatSharedPtr I0, I1;
                // the number of points before and after interpolation are
                // the same for each element inside a single collection
                int pt0  = m_expList[0]->GetNumPoints(0);
                int pt1  = m_expList[0]->GetNumPoints(1);
                int npt0 = (int)pt0 * scale;
                int npt1 = (int)pt1 * scale;
                // workspace declaration
                Array<OneD, NekDouble> wsp(npt1 * pt0); // fnp0*tnp1

                for (int i = 0; i < m_numElmt; ++i)
                {
                    // current points key
                    LibUtilities::PointsKey PointsKey0(
                        pt0, m_expList[i]->GetPointsType(0));
                    LibUtilities::PointsKey PointsKey1(
                        pt1, m_expList[i]->GetPointsType(1));
                    // get new points key
                    LibUtilities::PointsKey newPointsKey0(
                        npt0, m_expList[i]->GetPointsType(0));
                    LibUtilities::PointsKey newPointsKey1(
                        npt1, m_expList[i]->GetPointsType(1));

                    // Interpolate points;
                    I0 = LibUtilities::PointsManager()[PointsKey0]->GetI(
                        newPointsKey0);
                    I1 = LibUtilities::PointsManager()[PointsKey1]->GetI(
                        newPointsKey1);

                    Blas::Dgemm('N', 'T', pt0, npt1, pt1, 1.0, &input[cnt], pt0,
                                I1->GetPtr().get(), npt1, 0.0, wsp.get(), pt0);

                    Blas::Dgemm('N', 'N', npt0, npt1, pt0, 1.0,
                                I0->GetPtr().get(), npt0, wsp.get(), pt0, 0.0,
                                &output[cnt1], npt0);

                    cnt += pt0 * pt1;
                    cnt1 += npt0 * npt1;
                }
            }
            break;
            case 3:
            {
                DNekMatSharedPtr I0, I1, I2;

                int pt0  = m_expList[0]->GetNumPoints(0);
                int pt1  = m_expList[0]->GetNumPoints(1);
                int pt2  = m_expList[0]->GetNumPoints(2);
                int npt0 = (int)pt0 * scale;
                int npt1 = (int)pt1 * scale;
                int npt2 = (int)pt2 * scale;
                Array<OneD, NekDouble> wsp1(npt0 * npt1 * pt2);
                Array<OneD, NekDouble> wsp2(npt0 * pt1 * pt2);

                for (int i = 0; i < m_numElmt; ++i)
                {
                    // current points key
                    LibUtilities::PointsKey PointsKey0(
                        pt0, m_expList[i]->GetPointsType(0));
                    LibUtilities::PointsKey PointsKey1(
                        pt1, m_expList[i]->GetPointsType(1));
                    LibUtilities::PointsKey PointsKey2(
                        pt2, m_expList[i]->GetPointsType(2));
                    // get new points key
                    LibUtilities::PointsKey newPointsKey0(
                        npt0, m_expList[i]->GetPointsType(0));
                    LibUtilities::PointsKey newPointsKey1(
                        npt1, m_expList[i]->GetPointsType(1));
                    LibUtilities::PointsKey newPointsKey2(
                        npt2, m_expList[i]->GetPointsType(2));

                    I0 = LibUtilities::PointsManager()[PointsKey0]->GetI(
                        newPointsKey0);
                    I1 = LibUtilities::PointsManager()[PointsKey1]->GetI(
                        newPointsKey1);
                    I2 = LibUtilities::PointsManager()[PointsKey2]->GetI(
                        newPointsKey2);

                    // Interpolate points
                    Blas::Dgemm('N', 'N', npt0, pt1 * pt2, pt0, 1.0,
                                I0->GetPtr().get(), npt0, &input[cnt], pt0, 0.0,
                                wsp2.get(), npt0);

                    for (int j = 0; j < pt2; j++)
                    {
                        Blas::Dgemm('N', 'T', npt0, npt1, pt1, 1.0,
                                    wsp2.get() + j * npt0 * pt1, npt0,
                                    I1->GetPtr().get(), npt1, 0.0,
                                    wsp1.get() + j * npt0 * npt1, npt0);
                    }

                    Blas::Dgemm('N', 'T', npt0 * npt1, npt2, pt2, 1.0,
                                wsp1.get(), npt0 * npt1, I2->GetPtr().get(),
                                npt2, 0.0, &output[cnt1], npt0 * npt1);

                    cnt += pt0 * pt1 * pt2;
                    cnt1 += npt0 * npt1 * npt2;
                }
            }
            break;
            default:
            {
                NEKERROR(ErrorUtil::efatal, "This expansion is not set for the "
                                            "PhysInterp1DScaled operator.");
            }
            break;
        }
    }
    void operator()([[maybe_unused]] int dir,
                    [[maybe_unused]] const Array<OneD, const NekDouble> &input,
                    [[maybe_unused]] Array<OneD, NekDouble> &output,
                    [[maybe_unused]] Array<OneD, NekDouble> &wsp) final
    {
        ASSERTL0(false, "Not valid for this operator.");
    }

    void UpdateFactors([[maybe_unused]] StdRegions::FactorMap factors,
                       [[maybe_unused]] int coll_phys_offset) override
    {
        PhysInterp1DScaled_Helper::UpdateFactors(factors, coll_phys_offset);
        m_factors = factors;
    }

protected:
    vector<StdRegions::StdExpansionSharedPtr> m_expList;
    StdRegions::FactorMap m_factors; // Initially, PhysInterp1DScaled was taking
                                     // as input a single scaling factor
    // This single scaling factor will be expressed into a matrix where each
    // element has the same value so that matrix - vector multiplication
    // optimization by BLAS is implemented

private:
    PhysInterp1DScaled_NoCollection(
        vector<StdRegions::StdExpansionSharedPtr> pCollExp,
        CoalescedGeomDataSharedPtr pGeomData, StdRegions::FactorMap factors)
        : Operator(pCollExp, pGeomData, factors), PhysInterp1DScaled_Helper()
    {
        m_expList = pCollExp;
        m_factors = factors;
    }
};

/// Factory initialisation for the PhysInterp1DScaled_NoCollection operators
OperatorKey PhysInterp1DScaled_NoCollection::m_typeArr[] = {
    GetOperatorFactory().RegisterCreatorFunction(
        OperatorKey(eSegment, ePhysInterp1DScaled, eNoCollection, false),
        PhysInterp1DScaled_NoCollection::create,
        "PhysInterp1DScaled_NoCollection_Seg"),
    GetOperatorFactory().RegisterCreatorFunction(
        OperatorKey(eTriangle, ePhysInterp1DScaled, eNoCollection, false),
        PhysInterp1DScaled_NoCollection::create,
        "PhysInterp1DScaled_NoCollection_Tri"),
    GetOperatorFactory().RegisterCreatorFunction(
        OperatorKey(eTriangle, ePhysInterp1DScaled, eNoCollection, true),
        PhysInterp1DScaled_NoCollection::create,
        "PhysInterp1DScaled_NoCollection_NodalTri"),
    GetOperatorFactory().RegisterCreatorFunction(
        OperatorKey(eQuadrilateral, ePhysInterp1DScaled, eNoCollection, false),
        PhysInterp1DScaled_NoCollection::create,
        "PhysInterp1DScaled_NoCollection_Quad"),
    GetOperatorFactory().RegisterCreatorFunction(
        OperatorKey(eTetrahedron, ePhysInterp1DScaled, eNoCollection, false),
        PhysInterp1DScaled_NoCollection::create,
        "PhysInterp1DScaled_NoCollection_Tet"),
    GetOperatorFactory().RegisterCreatorFunction(
        OperatorKey(eTetrahedron, ePhysInterp1DScaled, eNoCollection, true),
        PhysInterp1DScaled_NoCollection::create,
        "PhysInterp1DScaled_NoCollection_NodalTet"),
    GetOperatorFactory().RegisterCreatorFunction(
        OperatorKey(ePyramid, ePhysInterp1DScaled, eNoCollection, false),
        PhysInterp1DScaled_NoCollection::create,
        "PhysInterp1DScaled_NoCollection_Pyr"),
    GetOperatorFactory().RegisterCreatorFunction(
        OperatorKey(ePrism, ePhysInterp1DScaled, eNoCollection, false),
        PhysInterp1DScaled_NoCollection::create,
        "PhysInterp1DScaled_NoCollection_Prism"),
    GetOperatorFactory().RegisterCreatorFunction(
        OperatorKey(ePrism, ePhysInterp1DScaled, eNoCollection, true),
        PhysInterp1DScaled_NoCollection::create,
        "PhysInterp1DScaled_NoCollection_NodalPrism"),
    GetOperatorFactory().RegisterCreatorFunction(
        OperatorKey(eHexahedron, ePhysInterp1DScaled, eNoCollection, false),
        PhysInterp1DScaled_NoCollection::create,
        "PhysInterp1DScaled_NoCollection_Hex"),
};

} // namespace Nektar::Collections
