///////////////////////////////////////////////////////////////////////////////
//
// File: StdQuadExp.h
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
// Description: Header field for Quadrilateral routines built upon
// StdExpansion2D
//
///////////////////////////////////////////////////////////////////////////////

#ifndef NEKTAR_LIB_STDREGIONS_STDQUADEXP_H
#define NEKTAR_LIB_STDREGIONS_STDQUADEXP_H

#include <StdRegions/StdExpansion2D.h>
#include <StdRegions/StdSegExp.h>

namespace Nektar::StdRegions
{
class StdQuadExp : virtual public StdExpansion2D
{
public:
    STD_REGIONS_EXPORT StdQuadExp(const LibUtilities::BasisKey &Ba,
                                  const LibUtilities::BasisKey &Bb);
    STD_REGIONS_EXPORT StdQuadExp(const StdQuadExp &T) = default;
    STD_REGIONS_EXPORT ~StdQuadExp() override          = default;

protected:
    //-------------------------------
    // Integration Methods
    //-------------------------------
    STD_REGIONS_EXPORT NekDouble
    v_Integral(const Array<OneD, const NekDouble> &inarray) override;

    //-------------------------------
    // Differentiation Methods
    //-------------------------------
    /** \brief Calculate the 2D derivative in the local
     *  tensor/collapsed coordinate at the physical points
     *
     *  This function is independent of the expansion basis and can
     *  therefore be defined for all tensor product distribution of
     *  quadrature points in a generic manner.  The key operations are:
     *
     *  - \f$ \frac{d}{d\eta_1} \rightarrow {\bf D^T_0 u } \f$ \n
     *  - \f$ \frac{d}{d\eta_2} \rightarrow {\bf D_1 u } \f$
     *
     *  \param inarray array of physical points to be differentiated
     *  \param  outarray_d0 the resulting array of derivative in the
     *  \f$\eta_1\f$ direction will be stored in outarray_d0 as output
     *  of the function
     *  \param outarray_d1 the resulting array of derivative in the
     *  \f$\eta_2\f$ direction will be stored in outarray_d1 as output
     *  of the function
     *
     *  Recall that:
     *  \f$
     *  \hspace{1cm} \begin{array}{llll}
     *  \mbox{Shape}    & \mbox{Cartesian coordinate range} &
     *  \mbox{Collapsed coord.}      &
     *  \mbox{Collapsed coordinate definition}\\
     *  \mbox{Quadrilateral}  & -1 \leq \xi_1,\xi_2 \leq  1
     *  & -1 \leq \eta_1,\eta_2 \leq 1
     *  & \eta_1 = \xi_1, \eta_2 = \xi_2\\
     *  \mbox{Triangle}  & -1 \leq \xi_1,\xi_2; \xi_1+\xi_2 \leq  0
     *  & -1 \leq \eta_1,\eta_2 \leq 1
     *  & \eta_1 = \frac{2(1+\xi_1)}{(1-\xi_2)}-1, \eta_2 = \xi_2 \\
     *  \end{array} \f$
     */
    STD_REGIONS_EXPORT void PhysTensorDeriv(
        const Array<OneD, const NekDouble> &inarray,
        Array<OneD, NekDouble> &outarray_d0,
        Array<OneD, NekDouble> &outarray_d1);
    STD_REGIONS_EXPORT void v_PhysDeriv(
        const Array<OneD, const NekDouble> &inarray,
        Array<OneD, NekDouble> &out_d0, Array<OneD, NekDouble> &out_d1,
        Array<OneD, NekDouble> &out_d2 = NullNekDouble1DArray) override;
    STD_REGIONS_EXPORT void v_PhysDeriv(
        const int dir, const Array<OneD, const NekDouble> &inarray,
        Array<OneD, NekDouble> &outarray) override;
    STD_REGIONS_EXPORT void v_StdPhysDeriv(
        const Array<OneD, const NekDouble> &inarray,
        Array<OneD, NekDouble> &out_d0, Array<OneD, NekDouble> &out_d1,
        Array<OneD, NekDouble> &out_d2 = NullNekDouble1DArray) override;

    //---------------------------------------
    // Transforms
    //---------------------------------------
    STD_REGIONS_EXPORT void v_BwdTrans(
        const Array<OneD, const NekDouble> &inarray,
        Array<OneD, NekDouble> &outarray) override;
    STD_REGIONS_EXPORT void v_FwdTrans(
        const Array<OneD, const NekDouble> &inarray,
        Array<OneD, NekDouble> &outarray) override;
    STD_REGIONS_EXPORT void v_FwdTransBndConstrained(
        const Array<OneD, const NekDouble> &inarray,
        Array<OneD, NekDouble> &outarray) override;

    //---------------------------------------
    // Inner product functions
    //---------------------------------------
    STD_REGIONS_EXPORT void v_IProductWRTBase(
        const Array<OneD, const NekDouble> &inarray,
        Array<OneD, NekDouble> &outarray) override;
    STD_REGIONS_EXPORT void v_IProductWRTBaseKernel(
        const Array<OneD, const NekDouble> &base0,
        const Array<OneD, const NekDouble> &base1,
        const Array<OneD, const NekDouble> &inarray,
        Array<OneD, NekDouble> &outarray, const Array<OneD, NekDouble> &jac,
        const bool Deformed, [[maybe_unused]] const bool CollDir0 = false,
        [[maybe_unused]] const bool CollDir1 = false) override;
    STD_REGIONS_EXPORT void v_IProductWRTDerivBase(
        const int dir, const Array<OneD, const NekDouble> &inarray,
        Array<OneD, NekDouble> &outarray) override;

    //---------------------------------------
    // Evaluation functions
    //---------------------------------------
    STD_REGIONS_EXPORT void v_LocCoordToLocCollapsed(
        const Array<OneD, const NekDouble> &xi,
        Array<OneD, NekDouble> &eta) override;
    STD_REGIONS_EXPORT void v_LocCollapsedToLocCoord(
        const Array<OneD, const NekDouble> &eta,
        Array<OneD, NekDouble> &xi) override;
    STD_REGIONS_EXPORT void v_FillMode(const int mode,
                                       Array<OneD, NekDouble> &array) override;

    //---------------------------
    // Helper functions
    //---------------------------
    STD_REGIONS_EXPORT int v_GetNverts() const final;
    STD_REGIONS_EXPORT int v_GetNtraces() const final;
    STD_REGIONS_EXPORT int v_GetTraceNcoeffs(const int i) const final;
    STD_REGIONS_EXPORT int v_GetTraceIntNcoeffs(const int i) const final;
    STD_REGIONS_EXPORT int v_GetTraceNumPoints(const int i) const final;
    STD_REGIONS_EXPORT int v_NumBndryCoeffs() const final;
    STD_REGIONS_EXPORT int v_NumDGBndryCoeffs() const final;
    STD_REGIONS_EXPORT int v_CalcNumberOfCoefficients(
        const std::vector<unsigned int> &nummodes, int &modes_offset) override;
    STD_REGIONS_EXPORT const LibUtilities::BasisKey v_GetTraceBasisKey(
        const int i, const int j, bool UseGLL = false) const final;
    STD_REGIONS_EXPORT LibUtilities::ShapeType v_DetShapeType() const final;
    STD_REGIONS_EXPORT bool v_IsBoundaryInteriorExpansion() const override;
    STD_REGIONS_EXPORT void v_GetCoords(
        Array<OneD, NekDouble> &coords_0, Array<OneD, NekDouble> &coords_1,
        Array<OneD, NekDouble> &coords_2) override;
    STD_REGIONS_EXPORT NekDouble v_PhysEvaluateBasis(
        const Array<OneD, const NekDouble> &coords, int mode) override;
    STD_REGIONS_EXPORT NekDouble
    v_PhysEvalFirstDeriv(const Array<OneD, NekDouble> &coord,
                         const Array<OneD, const NekDouble> &inarray,
                         std::array<NekDouble, 3> &firstOrderDerivs) override;

    //--------------------------
    // Mappings
    //--------------------------
    STD_REGIONS_EXPORT void v_GetBoundaryMap(
        Array<OneD, unsigned int> &outarray) override;
    STD_REGIONS_EXPORT void v_GetInteriorMap(
        Array<OneD, unsigned int> &outarray) override;
    STD_REGIONS_EXPORT int v_GetVertexMap(
        int localVertexId, bool useCoeffPacking = false) override;
    STD_REGIONS_EXPORT void v_GetTraceCoeffMap(
        const unsigned int traceid,
        Array<OneD, unsigned int> &maparray) override;
    STD_REGIONS_EXPORT void v_GetTraceInteriorToElementMap(
        const int eid, Array<OneD, unsigned int> &maparray,
        Array<OneD, int> &signarray,
        const Orientation edgeOrient = eForwards) override;

    //---------------------------------------
    // Wrapper functions
    //---------------------------------------
    STD_REGIONS_EXPORT DNekMatSharedPtr
    v_GenMatrix(const StdMatrixKey &mkey) override;
    STD_REGIONS_EXPORT DNekMatSharedPtr
    v_CreateStdMatrix(const StdMatrixKey &mkey) override;

    //---------------------------------------
    // Operator evaluation functions
    //---------------------------------------
    STD_REGIONS_EXPORT void v_MassMatrixOp(
        const Array<OneD, const NekDouble> &inarray,
        Array<OneD, NekDouble> &outarray, const StdMatrixKey &mkey) override;
    STD_REGIONS_EXPORT void v_LaplacianMatrixOp(
        const Array<OneD, const NekDouble> &inarray,
        Array<OneD, NekDouble> &outarray, const StdMatrixKey &mkey) override;
    STD_REGIONS_EXPORT void v_LaplacianMatrixOp(
        const int k1, const int k2, const Array<OneD, const NekDouble> &inarray,
        Array<OneD, NekDouble> &outarray, const StdMatrixKey &mkey) override;
    STD_REGIONS_EXPORT void v_WeakDerivMatrixOp(
        const int i, const Array<OneD, const NekDouble> &inarray,
        Array<OneD, NekDouble> &outarray, const StdMatrixKey &mkey) override;
    STD_REGIONS_EXPORT void v_HelmholtzMatrixOp(
        const Array<OneD, const NekDouble> &inarray,
        Array<OneD, NekDouble> &outarray, const StdMatrixKey &mkey) override;
    STD_REGIONS_EXPORT void v_SVVLaplacianFilter(
        Array<OneD, NekDouble> &array, const StdMatrixKey &mkey) override;
    STD_REGIONS_EXPORT void v_ExponentialFilter(
        Array<OneD, NekDouble> &array, const NekDouble alpha,
        const NekDouble exponent, const NekDouble cutoff) override;
    STD_REGIONS_EXPORT void v_ReduceOrderCoeffs(
        int numMin, const Array<OneD, const NekDouble> &inarray,
        Array<OneD, NekDouble> &outarray) override;
    STD_REGIONS_EXPORT void v_MultiplyByStdQuadratureMetric(
        const Array<OneD, const NekDouble> &inarray,
        Array<OneD, NekDouble> &outarray) override;

    //---------------------------------------
    // Output interpolation functions
    //---------------------------------------
    STD_REGIONS_EXPORT void v_GetSimplexEquiSpacedConnectivity(
        Array<OneD, int> &conn, bool standard = true) override;
};
typedef std::shared_ptr<StdQuadExp> StdQuadExpSharedPtr;

} // namespace Nektar::StdRegions

#endif // STDQUADEXP_H
