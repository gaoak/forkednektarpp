#include "../Operator.hpp"
#include "Operators/OperatorIProductWRTBase.hpp"

namespace Nektar::Operators::detail
{
template <typename TData>
void IProductWRTBaseSumFacSegKernel(
    const TData *&inptr, TData *&outptr,
    std::shared_ptr<Nektar::LocalRegions::Expansion> expPtr,
    Array<OneD, TData> m_jac, int const numElmts, size_t &jac_idx)
{

    auto const isDeformed = expPtr->GetMetricInfo()->GetGtype();
    auto const nquad0     = expPtr->GetNumPoints(0);
    auto const nmodes0    = expPtr->GetBasisNumModes(0);
    auto const weights    = expPtr->GetBasis(0)->GetW();
    bool const colldir0   = expPtr->GetBasis(0)->Collocation();
    auto const base0      = expPtr->GetBasis(0)->GetBdata();

    // Allocation workspace to store W*f to the heap
    auto const wspSize = numElmts * nquad0;
    double *wsp;
    wsp = new double[wspSize];

    // Pre-multiply inptr with jacobian and store results in wsp
    // wsp = J * \hat{f}
    if (isDeformed == SpatialDomains::eDeformed)
    {
        // wsp = Jac * inptr
        Vmath::Vmul(numElmts * nquad0, &m_jac[jac_idx], 1, inptr, 1, wsp, 1);

        jac_idx += numElmts * nquad0;
    }
    else
    {
        // wsp = Jac * inptr
        // Looping through elements
        for (int e = 0; e < numElmts; ++e)
        {
            Vmath::Smul(nquad0, m_jac[jac_idx], inptr + e * nquad0, 1,
                        wsp + e * nquad0, 1);

            jac_idx += 1;
        }
    }

    // Multiplies wsp with B^T
    // outptr = B^T * W * wsp
    for (int e = 0; e < numElmts; ++e)
    {
        if (colldir0)
        {
            Vmath::Vmul(nquad0, wsp + e * nquad0, 1, &weights[0], 1,
                        outptr + e * nquad0, 1);
        }
        else
        {
            // wsp = W * inptr
            Vmath::Vmul(nquad0, &weights[0], 1, wsp + e * nquad0, 1,
                        wsp + e * nquad0, 1);

            // outptr = B^T * wsp;
            Blas::Dgemv('T', nquad0, nmodes0, 1.0, base0.get(), nquad0,
                        &wsp[0 + e * nquad0], 1, 0.0, &outptr[0 + e * nmodes0],
                        1);
        }
    }

    // Deallocating memory
    delete[] wsp;
}

template <typename TData>
void IProductWRTBaseSumFacQuadKernel(
    const TData *&inptr, TData *&outptr,
    std::shared_ptr<Nektar::LocalRegions::Expansion> expPtr,
    Array<OneD, TData> m_jac, int const numElmts, size_t &jac_idx)
{
    auto const isDeformed = expPtr->GetMetricInfo()->GetGtype();

    auto const nquad0 = expPtr->GetNumPoints(0);
    auto const nquad1 = expPtr->GetNumPoints(1);

    auto const nmodes0 = expPtr->GetBasisNumModes(0);
    auto const nmodes1 = expPtr->GetBasisNumModes(1);

    auto const weights0 = expPtr->GetBasis(0)->GetW();
    auto const weights1 = expPtr->GetBasis(1)->GetW();

    bool const colldir0 = expPtr->GetBasis(0)->Collocation();
    bool const colldir1 = expPtr->GetBasis(1)->Collocation();

    auto const base0 = expPtr->GetBasis(0)->GetBdata();
    auto const base1 = expPtr->GetBasis(1)->GetBdata();

    // Allocation workspace to store W*f to the heap
    auto const wspSize = numElmts * nquad0 * nquad1;
    auto const tmpSize = numElmts * nmodes0 * nquad1;
    double *wsp;
    double *tmp;
    wsp = new double[wspSize];
    tmp = new double[tmpSize];

    // Pre-multiply inptr with jacobian and store results in wsp
    // wsp = J * \hat{f}
    if (isDeformed == SpatialDomains::eDeformed)
    {
        // wsp = Jac * inptr
        Vmath::Vmul(numElmts * nquad0 * nquad1, &m_jac[jac_idx], 1, inptr, 1,
                    wsp, 1);

        jac_idx += numElmts * nquad0 * nquad1;
    }
    else
    {
        // wsp = Jac * inptr
        // Looping through elements
        for (int e = 0; e < numElmts; ++e)
        {
            Vmath::Smul(nquad0 * nquad1, m_jac[jac_idx],
                        inptr + e * (nquad0 * nquad1), 1,
                        wsp + e * (nquad0 * nquad1), 1);
            jac_idx += 1;
        }
    }

    // Multiplies wsp with B^T
    // outptr = B^T * W * wsp
    for (int e = 0; e < numElmts; ++e)
    {
        // Pre-multiply integration weights
        // Performs [W1][Fij]
        for (int j = 0; j < nquad1; ++j)
        {
            Vmath::Vmul(nquad0, &weights0[0], 1,
                        wsp + j * nquad0 + e * (nquad1 * nquad0), 1,
                        wsp + j * nquad0 + e * (nquad1 * nquad0), 1);
        }

        // Performs [Fij][W2]
        for (int i = 0; i < nquad0; ++i)
        {
            Vmath::Vmul(nquad1, &weights1[0], 1,
                        wsp + i + e * (nquad1 * nquad0), nquad0,
                        wsp + i + e * (nquad1 * nquad0), nquad0);
        }

        if (colldir0 && colldir1)
        {
            Vmath::Vcopy(nmodes0 * nmodes1, wsp + e * (nquad1 * nquad0), 1,
                         outptr + e * (nquad1 * nquad0), 1);
        }
        else if (colldir0)
        {
            // [f] * [B]
            Blas::Dgemm('N', 'N', nmodes0, nmodes1, nquad1, 1.0,
                        &wsp[0 + e * (nmodes0 * nquad1)], nmodes0, base1.get(),
                        nquad1, 0.0, &outptr[0 + e * (nmodes0 * nmodes1)],
                        nmodes0);
        }
        else if (colldir1)
        {
            // [B^T] * [f]
            Blas::Dgemm('T', 'N', nmodes0, nquad1, nquad0, 1.0, base0.get(),
                        nquad0, &wsp[0 + e * (nquad1 * nquad0)], nquad0, 0.0,
                        &outptr[0 + e * (nmodes0 * nquad1)], nmodes0);
        }
        else
        {

            // [B^T] * [f]
            Blas::Dgemm('T', 'N', nmodes0, nquad1, nquad0, 1.0, base0.get(),
                        nquad0, &wsp[0 + e * (nquad1 * nquad0)], nquad0, 0.0,
                        &tmp[0 + e * (nmodes0 * nquad1)], nmodes0);

            // [f] * [B]
            Blas::Dgemm('N', 'N', nmodes0, nmodes1, nquad1, 1.0,
                        &tmp[0 + e * (nmodes0 * nquad1)], nmodes0, base1.get(),
                        nquad1, 0.0, &outptr[0 + e * (nmodes0 * nmodes1)],
                        nmodes0);
        }
    }

    delete[] wsp;
    delete[] tmp;
}

template <typename TData>
void IProductWRTBaseSumFacHexKernel(
    const TData *&inptr, TData *&outptr,
    std::shared_ptr<Nektar::LocalRegions::Expansion> expPtr,
    Array<OneD, TData> m_jac, int const numElmts, size_t &jac_idx)
{
    auto const isDeformed = expPtr->GetMetricInfo()->GetGtype();

    auto const nquad0 = expPtr->GetNumPoints(0);
    auto const nquad1 = expPtr->GetNumPoints(1);
    auto const nquad2 = expPtr->GetNumPoints(2);

    auto const nmodes0  = expPtr->GetBasisNumModes(0);
    auto const nmodes1  = expPtr->GetBasisNumModes(1);
    auto const nmodes2  = expPtr->GetBasisNumModes(2);
    auto const totModes = nmodes0 * nmodes1 * nmodes2;

    auto const weights0 = expPtr->GetBasis(0)->GetW();
    auto const weights1 = expPtr->GetBasis(1)->GetW();
    auto const weights2 = expPtr->GetBasis(2)->GetW();

    bool const colldir0 = expPtr->GetBasis(0)->Collocation();
    bool const colldir1 = expPtr->GetBasis(1)->Collocation();
    bool const colldir2 = expPtr->GetBasis(2)->Collocation();

    auto const base0 = expPtr->GetBasis(0)->GetBdata();
    auto const base1 = expPtr->GetBasis(1)->GetBdata();
    auto const base2 = expPtr->GetBasis(2)->GetBdata();

    // Allocation workspace to store W*f to the heap
    auto const totPoints = nquad0 * nquad1 * nquad2;
    auto const wspSize   = numElmts * totPoints;

    auto const totPoints1 = nquad1 * nquad2 * nmodes0;
    auto const wspSize1   = numElmts * totPoints1;

    auto const totPoints2 = nquad2 * nmodes0 * nmodes1;
    auto const wspSize2   = numElmts * totPoints2;

    double *wsp;
    double *wsp1;
    double *wsp2;
    wsp  = new double[wspSize];
    wsp1 = new double[wspSize1];
    wsp2 = new double[wspSize2];

    // Pre-multiply inptr with jacobian and store results in wsp
    // wsp = J * \hat{f}
    if (isDeformed == SpatialDomains::eDeformed)
    {
        // wsp = Jac * inptr
        Vmath::Vmul(numElmts * nquad0 * nquad1 * nquad2, &m_jac[jac_idx], 1,
                    inptr, 1, wsp, 1);
        jac_idx += numElmts * nquad0 * nquad1 * nquad2;
    }
    else
    {
        // Looping through elements
        for (int e = 0; e < numElmts; ++e)
        {
            Vmath::Smul(totPoints, m_jac[jac_idx], inptr + e * totPoints, 1,
                        wsp + e * totPoints, 1);
            jac_idx += 1;
        }
    }

    // Multiplies wsp with B^T
    // outptr = B^T * W * wsp
    for (int e = 0; e < numElmts; ++e)
    {
        // Pre-multiply integration weights0
        int stride = 0;
        for (int k = 0; k < nquad2; ++k)
        {
            for (int j = 0; j < nquad1; ++j)
            {
                stride = j * nquad0 + k * nquad1 * nquad0 + e * totPoints;
                Vmath::Vmul(nquad0, &weights0[0], 1, wsp + stride, 1,
                            wsp + stride, 1);
            }
        }

        // Pre-multiply integration weights1
        for (int k = 0; k < nquad2; ++k)
        {
            for (int i = 0; i < nquad0; ++i)
            {
                stride = i + k * nquad1 * nquad0 + e * totPoints;
                Vmath::Vmul(nquad1, &weights1[0], 1, wsp + stride, nquad0,
                            wsp + stride, nquad0);
            }
        }

        // Pre-multiply integration weights2
        for (int j = 0; j < nquad1; ++j)
        {
            for (int i = 0; i < nquad0; ++i)
            {
                stride = i + j * nquad0 + e * totPoints;
                Vmath::Vmul(nquad2, &weights2[0], 1, wsp + stride,
                            nquad0 * nquad1, wsp + stride, nquad0 * nquad1);
            }
        }

        // Checks for collocated bases and performs SumFac
        if (colldir0)
        {
            for (int i = 0; i < nmodes0; ++i)
            {
                Vmath::Vcopy(nquad1 * nquad2, &wsp[e * totPoints] + i, nquad0,
                             wsp1 + e * totPoints1 + nquad1 * nquad2 * i, 1);
            }
        }
        else
        {
            Blas::Dgemm('T', 'N', nquad1 * nquad2, nmodes0, nquad0, 1.0,
                        &wsp[e * totPoints], nquad0, base0.get(), nquad0, 0.0,
                        &wsp1[e * totPoints1], nquad1 * nquad2);
        }

        if (colldir1)
        {
            for (int i = 0; i < nmodes1; ++i)
            {
                Vmath::Vcopy(nquad2 * nmodes0, wsp1 + e * totPoints1 + i,
                             nquad1,
                             wsp2 + e * totPoints2 + nquad2 * nmodes0 * i, 1);
            }
        }
        else
        {
            // Sum-fac w.r.t base1
            Blas::Dgemm('T', 'N', nquad2 * nmodes0, nmodes1, nquad1, 1.0,
                        &wsp1[e * totPoints1], nquad1, base1.get(), nquad1, 0.0,
                        &wsp2[e * totPoints2], nmodes0 * nquad2);
        }

        if (colldir2)
        {
            for (int i = 0; i < nmodes2; ++i)
            {
                Vmath::Vcopy(nmodes0 * nmodes1, wsp2 + e * totPoints2 + i,
                             nquad2,
                             &outptr[e * totModes] + nmodes0 * nmodes1 * i, 1);
            }
        }
        else
        {
            // Sum-fac w.r.t base2
            Blas::Dgemm('T', 'N', nmodes0 * nmodes1, nmodes2, nquad2, 1.0,
                        &wsp2[e * totPoints2], nquad2, base2.get(), nquad2, 0.0,
                        &outptr[e * totModes], nmodes0 * nmodes1);
        }
    }

    delete[] wsp;
    delete[] wsp1;
    delete[] wsp2;
}
} // namespace Nektar::Operators::detail
