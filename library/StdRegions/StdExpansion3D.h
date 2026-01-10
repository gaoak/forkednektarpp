///////////////////////////////////////////////////////////////////////////////
//
// File: StdExpansion3D.h
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
// Description: Daughter of StdExpansion. This class contains routine
// which are common to 3D expansion. Typically this inolves physiocal
// space operations.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef STDEXP3D_H
#define STDEXP3D_H

#include <StdRegions/StdExpansion.h>

namespace Nektar::StdRegions
{
class StdExpansion3D;
typedef std::shared_ptr<StdExpansion3D> StdExpansion3DSharedPtr;

class StdExpansion3D : virtual public StdExpansion
{

public:
    STD_REGIONS_EXPORT StdExpansion3D(int numcoeffs,
                                      const LibUtilities::BasisKey &Ba,
                                      const LibUtilities::BasisKey &Bb,
                                      const LibUtilities::BasisKey &Bc);
    STD_REGIONS_EXPORT StdExpansion3D()                        = default;
    STD_REGIONS_EXPORT StdExpansion3D(const StdExpansion3D &T) = default;
    STD_REGIONS_EXPORT ~StdExpansion3D() override              = default;

    STD_REGIONS_EXPORT void IProductWRTBaseKernel(
        const Array<OneD, const NekDouble> &base0,
        const Array<OneD, const NekDouble> &base1,
        const Array<OneD, const NekDouble> &base2,
        const Array<OneD, const NekDouble> &inarray,
        Array<OneD, NekDouble> &outarray, const Array<OneD, NekDouble> &jac,
        const bool Deformed, [[maybe_unused]] bool CollDir0 = false,
        [[maybe_unused]] bool CollDir1 = false,
        [[maybe_unused]] bool CollDir2 = false);

    /** \brief return the number of edges in 3D expansion
     */
    int GetNedges() const
    {
        return v_GetNedges();
    }

    /** \brief This function returns the number of expansion coefficients
     *  belonging to the \a i-th edge
     *
     *  This function is a wrapper around the virtual function
     *  \a v_GetEdgeNcoeffs()
     *
     *  \param i specifies which edge
     *  \return returns the number of expansion coefficients belonging to
     *  the \a i-th edge
     */
    int GetEdgeNcoeffs(const int i) const
    {
        return v_GetEdgeNcoeffs(i);
    }

    void GetEdgeInteriorToElementMap(const int tid,
                                     Array<OneD, unsigned int> &maparray,
                                     Array<OneD, int> &signarray,
                                     Orientation traceOrient = eForwards)
    {
        v_GetEdgeInteriorToElementMap(tid, maparray, signarray, traceOrient);
    }

protected:
    /** \brief This function evaluates the expansion at a single
     *  (arbitrary) point of the domain
     *
     *
     *  Based on the value of the expansion at the quadrature points,
     *  this function calculates the value of the expansion at an
     *  arbitrary single points (with coordinates \f$ \mathbf{x_c}\f$
     *  given by the pointer \a coords). This operation, equivalent to
     *  \f[ u(\mathbf{x_c})  = \sum_p \phi_p(\mathbf{x_c}) \hat{u}_p \f]
     *  is evaluated using Lagrangian interpolants through the quadrature
     *  points:
     *  \f[ u(\mathbf{x_c}) = \sum_p h_p(\mathbf{x_c}) u_p\f]
     *
     *  This function requires that the physical value array
     *  \f$\mathbf{u}\f$ (implemented as the attribute #phys)
     *  is set.
     *
     *  \param coords the coordinates of the single point
     *  \return returns the value of the expansion at the single point
     */
    STD_REGIONS_EXPORT NekDouble
    v_PhysEvaluate(const Array<OneD, const NekDouble> &coords,
                   const Array<OneD, const NekDouble> &physvals) override;

    STD_REGIONS_EXPORT NekDouble
    v_PhysEvaluateInterp(const Array<OneD, DNekMatSharedPtr> &I,
                         const Array<OneD, const NekDouble> &physvals) override;

    STD_REGIONS_EXPORT virtual void v_IProductWRTBaseKernel(
        const Array<OneD, const NekDouble> &base0,
        const Array<OneD, const NekDouble> &base1,
        const Array<OneD, const NekDouble> &base2,
        const Array<OneD, const NekDouble> &inarray,
        Array<OneD, NekDouble> &outarray, const Array<OneD, NekDouble> &jac,
        const bool Deformed, [[maybe_unused]] bool CollDir0 = false,
        [[maybe_unused]] bool CollDir1 = false,
        [[maybe_unused]] bool CollDir2 = false) = 0;

    STD_REGIONS_EXPORT void v_LaplacianMatrixOp_MatFree(
        const Array<OneD, const NekDouble> &inarray,
        Array<OneD, NekDouble> &outarray,
        const StdRegions::StdMatrixKey &mkey) override;

    STD_REGIONS_EXPORT void v_HelmholtzMatrixOp_MatFree(
        const Array<OneD, const NekDouble> &inarray,
        Array<OneD, NekDouble> &outarray,
        const StdRegions::StdMatrixKey &mkey) override;

    STD_REGIONS_EXPORT NekDouble
    v_Integral(const Array<OneD, const NekDouble> &inarray) override;

    STD_REGIONS_EXPORT virtual int v_GetNedges(void) const;
    STD_REGIONS_EXPORT virtual int v_GetEdgeNcoeffs(const int i) const;

    /**
     * Performs tensor product evaluation in 3D to evaluate the physical
     * and derivative values in each direction at input coordinate
     * @param coord using input physical values at quadrature points
     * @param inarray. Returns via reference the derivatives.

     * @param coord Global coordinate
     * @param inarray Phys values
     * @param out_d0 Return by reference parameter for 0th derivative
     * @param out_d1 Return by reference parameter for 1st derivative
     * @param out_d2 Return by reference parameter for 2nd derivative
     * @return Physical value at @param coord
     */
    STD_REGIONS_EXPORT inline NekDouble BaryTensorDeriv(
        const Array<OneD, NekDouble> &coord,
        const Array<OneD, const NekDouble> &inarray,
        std::array<NekDouble, 3> &firstOrderDerivs)
    {
        const int nq0 = m_base[0]->GetNumPoints();
        const int nq1 = m_base[1]->GetNumPoints();
        const int nq2 = m_base[2]->GetNumPoints();

        const NekDouble *ptr = &inarray[0];
        Array<OneD, NekDouble> deriv0(nq1 * nq2, 0.0);
        Array<OneD, NekDouble> phys0(nq1 * nq2, 0.0);
        Array<OneD, NekDouble> deriv0phys1(nq1, 0.0);
        Array<OneD, NekDouble> phys0deriv1(nq1, 0.0);
        Array<OneD, NekDouble> phys0phys1(nq1, 0.0);

        for (int j = 0; j < nq1 * nq2; ++j, ptr += nq0)
        {
            phys0[j] =
                StdExpansion::BaryEvaluate<0, true>(coord[0], ptr, deriv0[j]);
        }

        for (int j = 0; j < nq2; ++j)
        {
            deriv0phys1[j] = StdExpansion::BaryEvaluate<1, false>(
                coord[1], &deriv0[j * nq1]);
        }
        firstOrderDerivs[0] =
            StdExpansion::BaryEvaluate<2, false>(coord[2], &deriv0phys1[0]);

        for (int j = 0; j < nq2; ++j)
        {
            phys0phys1[j] = StdExpansion::BaryEvaluate<1, true>(
                coord[1], &phys0[j * nq1], phys0deriv1[j]);
        }
        firstOrderDerivs[1] =
            StdExpansion::BaryEvaluate<2, false>(coord[2], &phys0deriv1[0]);

        return StdExpansion::BaryEvaluate<2, true>(coord[2], &phys0phys1[0],
                                                   firstOrderDerivs[2]);
    }

    STD_REGIONS_EXPORT virtual void v_GetEdgeInteriorToElementMap(
        const int tid, Array<OneD, unsigned int> &maparray,
        Array<OneD, int> &signarray, Orientation traceOrient = eForwards);

    STD_REGIONS_EXPORT void v_GetTraceToElementMap(
        const int tid, Array<OneD, unsigned int> &maparray,
        Array<OneD, int> &signarray, Orientation traceOrient, int P,
        int Q) override;

    STD_REGIONS_EXPORT void v_GenStdMatBwdDeriv(const int dir,
                                                DNekMatSharedPtr &mat) override;

    STD_REGIONS_EXPORT void v_PhysInterp(
        std::shared_ptr<StdExpansion> fromExp,
        const Array<OneD, const NekDouble> &fromData,
        Array<OneD, NekDouble> &toData) override;

private:
    int v_GetShapeDimension() const final
    {
        return 3;
    }
};

STD_REGIONS_EXPORT LibUtilities::BasisKey EvaluateTriFaceBasisKey(
    const int facedir, const LibUtilities::BasisSharedPtr &faceDirBasis,
    bool UseGLL = false);

STD_REGIONS_EXPORT LibUtilities::BasisKey EvaluateQuadFaceBasisKey(
    const int facedir, const LibUtilities::BasisSharedPtr &faceDirBasis);
} // namespace Nektar::StdRegions

#endif // STDEXP3D_H
