///////////////////////////////////////////////////////////////////////////////
//
// File: NekFFTW.cpp
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
// Description: Wrapper around FFTW library
//
///////////////////////////////////////////////////////////////////////////////

#include <fftw3.h>

#include <LibUtilities/BasicUtils/SharedArray.hpp>
#include <LibUtilities/BasicUtils/VmathArray.hpp>
#include <LibUtilities/FFT/NekFFTW.h>

namespace Nektar::LibUtilities
{

template <>
std::string NekFFTWImpl<double>::className =
    GetNektarFFTFactory().RegisterCreatorFunction("NekFFTW",
                                                  NekFFTWImpl<double>::create);

#ifdef NEKTAR_HAVE_FFTW_FLOAT
template <>
std::string NekFFTWImpl<float>::className =
    GetNektarFFTFloatFactory().RegisterCreatorFunction(
        "NekFFTW", NekFFTWImpl<float>::create);
#endif

template <typename TData>
NekFFTWImpl<TData>::NekFFTWImpl(int N) : NektarFFT<TData>(N)
{
    m_wsp = Array<OneD, TData>(this->m_N);

    if constexpr (std::is_same_v<TData, double>)
    {
        m_plan_forward  = fftw_plan_r2r_1d(this->m_N, nullptr, nullptr,
                                           FFTW_R2HC, FFTW_ESTIMATE);
        m_plan_backward = fftw_plan_r2r_1d(this->m_N, nullptr, nullptr,
                                           FFTW_HC2R, FFTW_ESTIMATE);
    }
    else
    {
        m_plan_forward  = fftwf_plan_r2r_1d(this->m_N, nullptr, nullptr,
                                            FFTW_R2HC, FFTW_ESTIMATE);
        m_plan_backward = fftwf_plan_r2r_1d(this->m_N, nullptr, nullptr,
                                            FFTW_HC2R, FFTW_ESTIMATE);
    }

    m_FFTW_w     = Array<OneD, TData>(this->m_N);
    m_FFTW_w_inv = Array<OneD, TData>(this->m_N);

    m_FFTW_w[0] = TData(1) / static_cast<TData>(this->m_N);
    m_FFTW_w[1] = TData(0);

    m_FFTW_w_inv[0] = TData(1);
    m_FFTW_w_inv[1] = TData(0);

    for (int i = 2; i < this->m_N; i++)
    {
        m_FFTW_w[i]     = m_FFTW_w[0] * TData(2);
        m_FFTW_w_inv[i] = m_FFTW_w_inv[0] / TData(2);
    }
}

template <typename TData> NekFFTWImpl<TData>::~NekFFTWImpl()
{
    if constexpr (std::is_same_v<TData, double>)
    {
        fftw_destroy_plan((fftw_plan)m_plan_forward);
        fftw_destroy_plan((fftw_plan)m_plan_backward);
    }
    else
    {
        fftwf_destroy_plan((fftwf_plan)m_plan_forward);
        fftwf_destroy_plan((fftwf_plan)m_plan_backward);
    }
}

template <typename TData>
void NekFFTWImpl<TData>::v_FFTFwdTrans(TData *inarray, TData *outarray)
{
    const int halfN = this->m_N / 2;

    if constexpr (std::is_same_v<TData, double>)
    {
        fftw_execute_r2r((fftw_plan)m_plan_forward, inarray, m_wsp.data());
    }
    else
    {
        fftwf_execute_r2r((fftwf_plan)m_plan_forward, inarray, m_wsp.data());
    }

    // Reshuffle from half-complex to Nektar++ coefficient layout.
    outarray[1] = m_FFTW_w[1] * m_wsp[halfN];

    if constexpr (std::is_same_v<TData, double>)
    {
        Vmath::Vmul(halfN, m_wsp.data(), 1, m_FFTW_w.data(), 2, outarray, 2);
    }
    else
    {
        for (int i = 0; i < halfN; ++i)
        {
            outarray[2 * i] = m_wsp[i] * m_FFTW_w[2 * i];
        }
    }

    for (int i = 0; i < halfN - 1; i++)
    {
        outarray[(this->m_N - 1) - 2 * i] =
            m_FFTW_w[(this->m_N - 1) - 2 * i] * m_wsp[halfN + 1 + i];
    }
}

template <typename TData>
void NekFFTWImpl<TData>::v_FFTBwdTrans(TData *inarray, TData *outarray)
{
    const int halfN = this->m_N / 2;

    // Reshuffle from Nektar++ coefficient layout to half-complex.
    m_wsp[halfN] = m_FFTW_w_inv[1] * inarray[1];

    if constexpr (std::is_same_v<TData, double>)
    {
        Vmath::Vmul(halfN, inarray, 2, m_FFTW_w_inv.data(), 2, m_wsp.data(), 1);
    }
    else
    {
        for (int i = 0; i < halfN; ++i)
        {
            m_wsp[i] = inarray[2 * i] * m_FFTW_w_inv[2 * i];
        }
    }

    for (int i = 0; i < (halfN - 1); i++)
    {
        m_wsp[halfN + 1 + i] = m_FFTW_w_inv[(this->m_N - 1) - 2 * i] *
                               inarray[(this->m_N - 1) - 2 * i];
    }

    if constexpr (std::is_same_v<TData, double>)
    {
        fftw_execute_r2r((fftw_plan)m_plan_backward, m_wsp.data(), outarray);
    }
    else
    {
        fftwf_execute_r2r((fftwf_plan)m_plan_backward, m_wsp.data(), outarray);
    }
}

// Explicit instantiations.
template class NekFFTWImpl<double>;
#ifdef NEKTAR_HAVE_FFTW_FLOAT
template class NekFFTWImpl<float>;
#endif

} // namespace Nektar::LibUtilities
