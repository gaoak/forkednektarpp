///////////////////////////////////////////////////////////////////////////////
//
// File: StdExpansion2D.cpp
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
// which are common to 2D expansion. Typically this inolves physiocal
// space operations.
//
///////////////////////////////////////////////////////////////////////////////

#include <LibUtilities/Foundations/Interp.h>
#include <StdRegions/StdExpansion2D.h>

#ifdef max
#undef max
#endif

namespace Nektar::StdRegions
{

StdExpansion2D::StdExpansion2D(
    [[maybe_unused]] int numcoeffs,
    [[maybe_unused]] const LibUtilities::BasisKey &Ba,
    [[maybe_unused]] const LibUtilities::BasisKey &Bb)
{
}

//----------------------------
// Differentiation Methods
//----------------------------
void StdExpansion2D::v_PhysDeriv(const int dir,
                                 const Array<OneD, const NekDouble> &inarray,
                                 Array<OneD, NekDouble> &outarray)
{
    switch (dir)
    {
        case 0:
        {
            v_PhysDeriv(inarray, outarray, NullNekDouble1DArray,
                        NullNekDouble1DArray);
            break;
        }
        case 1:
        {
            v_PhysDeriv(inarray, NullNekDouble1DArray, outarray,
                        NullNekDouble1DArray);
            break;
        }
        default:
        {
            ASSERTL1(false, "input dir is out of range");
            break;
        }
    }
}

NekDouble StdExpansion2D::v_PhysEvaluate(
    const Array<OneD, const NekDouble> &coords,
    const Array<OneD, const NekDouble> &physvals)
{
    ASSERTL2(coords[0] > -1 - NekConstants::kNekZeroTol, "coord[0] < -1");
    ASSERTL2(coords[0] < 1 + NekConstants::kNekZeroTol, "coord[0] >  1");
    ASSERTL2(coords[1] > -1 - NekConstants::kNekZeroTol, "coord[1] < -1");
    ASSERTL2(coords[1] < 1 + NekConstants::kNekZeroTol, "coord[1] >  1");

    Array<OneD, NekDouble> coll(2);
    LocCoordToLocCollapsed(coords, coll);

    const int nq0 = m_base[0]->GetNumPoints();
    const int nq1 = m_base[1]->GetNumPoints();

    Array<OneD, NekDouble> wsp(nq1);
    for (int i = 0; i < nq1; ++i)
    {
        wsp[i] = StdExpansion::BaryEvaluate<0>(coll[0], &physvals[0] + i * nq0);
    }

    return StdExpansion::BaryEvaluate<1>(coll[1], &wsp[0]);
}

NekDouble StdExpansion2D::v_PhysEvaluateInterp(
    const Array<OneD, DNekMatSharedPtr> &I,
    const Array<OneD, const NekDouble> &physvals)
{
    NekDouble val;
    int i;
    int nq0 = m_base[0]->GetNumPoints();
    int nq1 = m_base[1]->GetNumPoints();
    Array<OneD, NekDouble> wsp1(nq1);

    // interpolate first coordinate direction
    for (i = 0; i < nq1; ++i)
    {
        wsp1[i] =
            Vmath::Dot(nq0, &(I[0]->GetPtr())[0], 1, &physvals[i * nq0], 1);
    }

    // interpolate in second coordinate direction
    val = Vmath::Dot(nq1, I[1]->GetPtr(), 1, wsp1, 1);

    return val;
}

//////////////////////////////
// Integration Methods
//////////////////////////////

NekDouble StdExpansion2D::Integral(const Array<OneD, const NekDouble> &inarray,
                                   const Array<OneD, const NekDouble> &w0,
                                   const Array<OneD, const NekDouble> &w1)
{
    int i;
    NekDouble Int = 0.0;
    int nquad0    = m_base[0]->GetNumPoints();
    int nquad1    = m_base[1]->GetNumPoints();
    Array<OneD, NekDouble> tmp(nquad0 * nquad1);

    // multiply by integration constants
    for (i = 0; i < nquad1; ++i)
    {
        Vmath::Vmul(nquad0, &inarray[0] + i * nquad0, 1, w0.data(), 1,
                    &tmp[0] + i * nquad0, 1);
    }

    for (i = 0; i < nquad0; ++i)
    {
        Vmath::Vmul(nquad1, &tmp[0] + i, nquad0, w1.data(), 1, &tmp[0] + i,
                    nquad0);
    }
    Int = Vmath::Vsum(nquad0 * nquad1, tmp, 1);

    return Int;
}

void StdExpansion2D::IProductWRTBaseKernel(
    const Array<OneD, const NekDouble> &base0,
    const Array<OneD, const NekDouble> &base1,
    const Array<OneD, const NekDouble> &inarray,
    Array<OneD, NekDouble> &outarray, const Array<OneD, NekDouble> &jac,
    const bool Deformed, const bool CollDir0, const bool CollDir1)
{
    v_IProductWRTBaseKernel(base0, base1, inarray, outarray, jac, Deformed,
                            CollDir0, CollDir1);
}

void StdExpansion2D::v_GenStdMatBwdDeriv(const int dir, DNekMatSharedPtr &mat)
{
    ASSERTL1((dir == 0) || (dir == 1), "Invalid direction.");

    const int nq0 = m_base[0]->GetNumPoints();
    const int nq1 = m_base[1]->GetNumPoints();
    const int nq  = nq0 * nq1;

    Array<OneD, NekDouble> in(nq, 0.0);
    Array<OneD, NekDouble> out(m_ncoeffs);
    Array<OneD, NekDouble> one(1, 1.0);

    for (int i = 0; i < nq; i++)
    {
        int l = i % nq0;
        int m = (i / nq0);

        // initialise with inverse of weights t
        in[i] = 1.0 / (m_weights[0][l] * m_weights[1][m]);

        // do standard iproduct
        if (dir == 0)
        {
            v_IProductWRTBaseKernel(m_base[0]->GetDbdata(),
                                    m_base[1]->GetBdata(), in, out, one, false,
                                    false, m_base[1]->Collocation());
        }
        else if (dir == 1)
        {
            v_IProductWRTBaseKernel(m_base[0]->GetBdata(),
                                    m_base[1]->GetDbdata(), in, out, one, false,
                                    m_base[0]->Collocation(), false);
        }
        in[i] = 0.0;

        for (int j = 0; j < m_ncoeffs; j++)
        {
            (*mat)(j, i) = out[j];
        }
    }
}

void StdExpansion2D::v_LaplacianMatrixOp_MatFree(
    const Array<OneD, const NekDouble> &inarray,
    Array<OneD, NekDouble> &outarray, const StdRegions::StdMatrixKey &mkey)
{
    if (mkey.GetNVarCoeff() == 0 &&
        !mkey.ConstFactorExists(StdRegions::eFactorCoeffD00) &&
        !mkey.ConstFactorExists(StdRegions::eFactorSVVCutoffRatio))
    {
        using std::max;

        // This implementation is only valid when there are no
        // coefficients associated to the Laplacian operator
        int nquad0  = m_base[0]->GetNumPoints();
        int nquad1  = m_base[1]->GetNumPoints();
        int nqtot   = nquad0 * nquad1;
        int nmodes0 = m_base[0]->GetNumModes();
        int nmodes1 = m_base[1]->GetNumModes();
        int wspsize =
            max(max(max(nqtot, m_ncoeffs), nquad1 * nmodes0), nquad0 * nmodes1);

        // Allocate temporary storage
        Array<OneD, NekDouble> wsp0(4 * wspsize);    // size wspsize
        Array<OneD, NekDouble> wsp1(wsp0 + wspsize); // size 3*wspsize

        if (!(m_base[0]->Collocation() && m_base[1]->Collocation()))
        {
            // LAPLACIAN MATRIX OPERATION
            // wsp0 = u       = B   * u_hat
            // wsp1 = du_dxi1 = D_xi1 * wsp0 = D_xi1 * u
            // wsp2 = du_dxi2 = D_xi2 * wsp0 = D_xi2 * u
            BwdTrans(inarray, wsp0);
            LaplacianMatrixOp_MatFree_Kernel(wsp0, outarray, wsp1);
        }
        else
        {
            LaplacianMatrixOp_MatFree_Kernel(inarray, outarray, wsp1);
        }
    }
    else
    {
        StdExpansion::LaplacianMatrixOp_MatFree_GenericImpl(inarray, outarray,
                                                            mkey);
    }
}

void StdExpansion2D::v_HelmholtzMatrixOp_MatFree(
    const Array<OneD, const NekDouble> &inarray,
    Array<OneD, NekDouble> &outarray, const StdRegions::StdMatrixKey &mkey)
{
    if (mkey.GetNVarCoeff() == 0 &&
        !mkey.ConstFactorExists(StdRegions::eFactorCoeffD00) &&
        !mkey.ConstFactorExists(StdRegions::eFactorSVVCutoffRatio))
    {
        using std::max;

        int nquad0  = m_base[0]->GetNumPoints();
        int nquad1  = m_base[1]->GetNumPoints();
        int nqtot   = nquad0 * nquad1;
        int nmodes0 = m_base[0]->GetNumModes();
        int nmodes1 = m_base[1]->GetNumModes();
        int wspsize =
            max(max(max(nqtot, m_ncoeffs), nquad1 * nmodes0), nquad0 * nmodes1);
        NekDouble lambda = mkey.GetConstFactor(StdRegions::eFactorLambda);

        // Allocate temporary storage
        Array<OneD, NekDouble> wsp0(5 * wspsize);        // size wspsize
        Array<OneD, NekDouble> wsp1(wsp0 + wspsize);     // size wspsize
        Array<OneD, NekDouble> wsp2(wsp0 + 2 * wspsize); // size 3*wspsize

        if (!(m_base[0]->Collocation() && m_base[1]->Collocation()))
        {
            // MASS MATRIX OPERATION
            // The following is being calculated:
            // wsp0     = B   * u_hat = u
            // wsp1     = W   * wsp0
            // outarray = B^T * wsp1  = B^T * W * B * u_hat = M * u_hat
            BwdTrans(inarray, wsp0);
            IProductWRTBase(wsp0, outarray);
            LaplacianMatrixOp_MatFree_Kernel(wsp0, wsp1, wsp2);
        }
        else
        {
            MultiplyByQuadratureMetric(inarray, outarray);
            LaplacianMatrixOp_MatFree_Kernel(inarray, wsp1, wsp2);
        }

        // outarray = lambda * outarray + wsp1
        //          = (lambda * M + L ) * u_hat
        Vmath::Svtvp(m_ncoeffs, lambda, &outarray[0], 1, &wsp1[0], 1,
                     &outarray[0], 1);
    }
    else
    {
        StdExpansion::HelmholtzMatrixOp_MatFree_GenericImpl(inarray, outarray,
                                                            mkey);
    }
}

void StdExpansion2D::v_GetTraceCoeffMap(
    [[maybe_unused]] const unsigned int traceid,
    [[maybe_unused]] Array<OneD, unsigned int> &maparray)
{
    ASSERTL0(false,
             "This method must be defined at the individual shape level");
}

/** \brief Determine the mapping to re-orientate the
    coefficients along the element trace (assumed to align
    with the standard element) into the orientation of the
    local trace given by edgeOrient.
 */
void StdExpansion2D::v_GetElmtTraceToTraceMap(
    const unsigned int eid, Array<OneD, unsigned int> &maparray,
    Array<OneD, int> &signarray, Orientation edgeOrient, int P,
    [[maybe_unused]] int Q)
{
    unsigned int i;

    int dir;
    // determine basis direction for edge.
    if (DetShapeType() == LibUtilities::eTriangle)
    {
        dir = (eid == 0) ? 0 : 1;
    }
    else
    {
        dir = eid % 2;
    }

    int numModes = m_base[dir]->GetNumModes();

    // P is the desired length of the map
    P = (P == -1) ? numModes : P;

    // decalare maparray
    if (maparray.size() != P)
    {
        maparray = Array<OneD, unsigned int>(P);
    }

    // fill default mapping as increasing index
    for (i = 0; i < P; ++i)
    {
        maparray[i] = i;
    }

    if (signarray.size() != P)
    {
        signarray = Array<OneD, int>(P, 1);
    }
    else
    {
        std::fill(signarray.data(), signarray.data() + P, 1);
    }

    // Zero signmap and set maparray to zero if
    // elemental modes are not as large as trace modes
    for (i = numModes; i < P; ++i)
    {
        signarray[i] = 0.0;
        maparray[i]  = maparray[0];
    }

    if (edgeOrient == eBackwards)
    {
        const LibUtilities::BasisType bType = GetBasisType(dir);

        if ((bType == LibUtilities::eModified_A) ||
            (bType == LibUtilities::eModified_B))
        {
            std::swap(maparray[0], maparray[1]);

            for (i = 3; i < std::min(P, numModes); i += 2)
            {
                signarray[i] *= -1;
            }
        }
        else if (bType == LibUtilities::eGLL_Lagrange ||
                 bType == LibUtilities::eGauss_Lagrange)
        {
            ASSERTL1(P == numModes, "Different trace space edge dimension "
                                    "and element edge dimension not currently "
                                    "possible for GLL-Lagrange bases");

            std::reverse(maparray.data(), maparray.data() + P);
        }
        else
        {
            ASSERTL0(false, "Mapping not defined for this type of basis");
        }
    }
}

void StdExpansion2D::v_GetTraceToElementMap(const int eid,
                                            Array<OneD, unsigned int> &maparray,
                                            Array<OneD, int> &signarray,
                                            Orientation edgeOrient, int P,
                                            int Q)
{
    Array<OneD, unsigned int> map1, map2;
    v_GetTraceCoeffMap(eid, map1);
    v_GetElmtTraceToTraceMap(eid, map2, signarray, edgeOrient, P, Q);

    if (maparray.size() != map2.size())
    {
        maparray = Array<OneD, unsigned int>(map2.size());
    }

    for (int i = 0; i < map2.size(); ++i)
    {
        maparray[i] = map1[map2[i]];
    }
}

void StdExpansion2D::v_PhysInterp(std::shared_ptr<StdExpansion> fromExp,
                                  const Array<OneD, const NekDouble> &fromData,
                                  Array<OneD, NekDouble> &toData)
{

    LibUtilities::Interp2D(fromExp->GetBasis(0)->GetPointsKey(),
                           fromExp->GetBasis(1)->GetPointsKey(), fromData,
                           m_base[0]->GetPointsKey(), m_base[1]->GetPointsKey(),
                           toData);
}

} // namespace Nektar::StdRegions
