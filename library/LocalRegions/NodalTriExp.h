///////////////////////////////////////////////////////////////////////////////
//
// File: NodalTriExp.h
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
// Description: Header for NodalTriExp routines
//
///////////////////////////////////////////////////////////////////////////////

#ifndef NODALTRIEXP_H
#define NODALTRIEXP_H

#include <SpatialDomains/TriGeom.h>
#include <StdRegions/StdNodalTriExp.h>

#include <LocalRegions/Expansion2D.h>
#include <LocalRegions/LocalRegionsDeclspec.h>
#include <LocalRegions/MatrixKey.h>

namespace Nektar::LocalRegions
{

class NodalTriExp final : virtual public StdRegions::StdNodalTriExp,
                          virtual public Expansion2D
{
public:
    /** \brief Constructor using BasisKey class for quadrature
        points and order definition */
    LOCAL_REGIONS_EXPORT NodalTriExp(
        const LibUtilities::BasisKey &Ba, const LibUtilities::BasisKey &Bb,
        const LibUtilities::PointsType Ntype,
        const SpatialDomains::TriGeomSharedPtr &geom);

    LOCAL_REGIONS_EXPORT NodalTriExp(const NodalTriExp &T);

    LOCAL_REGIONS_EXPORT ~NodalTriExp() override = default;

protected:
    //-------------------------------
    // Integration Methods
    //-------------------------------
    LOCAL_REGIONS_EXPORT NekDouble
    v_Integral(const Array<OneD, const NekDouble> &inarray) override;

    //----------------------------
    // Differentiation Methods
    //----------------------------
    LOCAL_REGIONS_EXPORT void v_PhysDeriv(
        const Array<OneD, const NekDouble> &inarray,
        Array<OneD, NekDouble> &out_d0, Array<OneD, NekDouble> &out_d1,
        [[maybe_unused]] Array<OneD, NekDouble> &out_d2 =
            NullNekDouble1DArray) override;
    LOCAL_REGIONS_EXPORT void v_PhysDeriv(
        const int dir, const Array<OneD, const NekDouble> &inarray,
        Array<OneD, NekDouble> &outarray) override;

    //---------------------------------------
    // Transforms
    //---------------------------------------
    LOCAL_REGIONS_EXPORT void v_FwdTrans(
        const Array<OneD, const NekDouble> &inarray,
        Array<OneD, NekDouble> &outarray) override;

    //---------------------------------------
    // Inner product functions
    //---------------------------------------
    LOCAL_REGIONS_EXPORT void v_IProductWRTBase(
        const Array<OneD, const NekDouble> &inarray,
        Array<OneD, NekDouble> &outarray) override;
    LOCAL_REGIONS_EXPORT void v_IProductWRTDerivBase(
        const int dir, const Array<OneD, const NekDouble> &inarray,
        Array<OneD, NekDouble> &outarray) override;
    LOCAL_REGIONS_EXPORT void v_AlignVectorToCollapsedDir(
        const int dir, const Array<OneD, const NekDouble> &inarray,
        Array<OneD, Array<OneD, NekDouble>> &outarray) override;

    //---------------------------------------
    // Evaluation functions
    //---------------------------------------
    LOCAL_REGIONS_EXPORT StdRegions::StdExpansionSharedPtr v_GetStdExp(
        void) const override;
    LOCAL_REGIONS_EXPORT StdRegions::StdExpansionSharedPtr v_GetLinStdExp(
        void) const override;
    LOCAL_REGIONS_EXPORT void v_GetCoords(
        Array<OneD, NekDouble> &coords_0,
        Array<OneD, NekDouble> &coords_1 = NullNekDouble1DArray,
        Array<OneD, NekDouble> &coords_2 = NullNekDouble1DArray) override;
    LOCAL_REGIONS_EXPORT void v_GetCoord(
        const Array<OneD, const NekDouble> &lcoord,
        Array<OneD, NekDouble> &coord) override;
    LOCAL_REGIONS_EXPORT NekDouble
    v_PhysEvaluate(const Array<OneD, const NekDouble> &coord,
                   const Array<OneD, const NekDouble> &physvals) override;
    LOCAL_REGIONS_EXPORT NekDouble
    v_PhysEvaluate(const Array<OneD, NekDouble> &coord,
                   const Array<OneD, const NekDouble> &inarray,
                   std::array<NekDouble, 3> &firstOrderDerivs) override;
    LOCAL_REGIONS_EXPORT void v_ComputeTraceNormal(const int edge) override;

    //---------------------------------------
    // Helper functions
    //---------------------------------------
    LOCAL_REGIONS_EXPORT void v_GetTracePhysVals(
        const int edge, const StdRegions::StdExpansionSharedPtr &EdgeExp,
        const Array<OneD, const NekDouble> &inarray,
        Array<OneD, NekDouble> &outarray,
        StdRegions::Orientation orient) override;

    //---------------------------------------
    // Matrix creation functions
    //---------------------------------------
    LOCAL_REGIONS_EXPORT DNekMatSharedPtr
    v_GenMatrix(const StdRegions::StdMatrixKey &mkey) override;
    LOCAL_REGIONS_EXPORT DNekMatSharedPtr
    v_CreateStdMatrix(const StdRegions::StdMatrixKey &mkey) override;
    LOCAL_REGIONS_EXPORT DNekScalMatSharedPtr
    v_GetLocMatrix(const MatrixKey &mkey) override;
    LOCAL_REGIONS_EXPORT DNekScalBlkMatSharedPtr
    v_GetLocStaticCondMatrix(const MatrixKey &mkey) override;
    LOCAL_REGIONS_EXPORT void v_DropLocMatrix(const MatrixKey &mkey) override;
    LOCAL_REGIONS_EXPORT void v_MassMatrixOp(
        const Array<OneD, const NekDouble> &inarray,
        Array<OneD, NekDouble> &outarray,
        const StdRegions::StdMatrixKey &mkey) override;
    LOCAL_REGIONS_EXPORT void v_LaplacianMatrixOp(
        const Array<OneD, const NekDouble> &inarray,
        Array<OneD, NekDouble> &outarray,
        const StdRegions::StdMatrixKey &mkey) override;
    LOCAL_REGIONS_EXPORT void v_LaplacianMatrixOp(
        const int k1, const int k2, const Array<OneD, const NekDouble> &inarray,
        Array<OneD, NekDouble> &outarray,
        const StdRegions::StdMatrixKey &mkey) override;
    LOCAL_REGIONS_EXPORT void v_WeakDerivMatrixOp(
        const int i, const Array<OneD, const NekDouble> &inarray,
        Array<OneD, NekDouble> &outarray,
        const StdRegions::StdMatrixKey &mkey) override;
    LOCAL_REGIONS_EXPORT void v_HelmholtzMatrixOp(
        const Array<OneD, const NekDouble> &inarray,
        Array<OneD, NekDouble> &outarray,
        const StdRegions::StdMatrixKey &mkey) override;

private:
    LibUtilities::NekManager<MatrixKey, DNekScalMat, MatrixKey::opLess>
        m_matrixManager;
    LibUtilities::NekManager<MatrixKey, DNekScalBlkMat, MatrixKey::opLess>
        m_staticCondMatrixManager;
};

typedef std::shared_ptr<NodalTriExp> NodalTriExpSharedPtr;
typedef std::vector<NodalTriExpSharedPtr> NodalTriExpVector;

} // namespace Nektar::LocalRegions

#endif // NODALTRIEXP_H
