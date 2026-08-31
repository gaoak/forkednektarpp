///////////////////////////////////////////////////////////////////////////////
//
// File: NekFFTW.h
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
// Description: Header file for the wrapper around FFTW library
//
///////////////////////////////////////////////////////////////////////////////

#ifndef NEKTAR_LIB_UTILIITIES_FFT_NEKFFTW_H
#define NEKTAR_LIB_UTILIITIES_FFT_NEKFFTW_H

#include <LibUtilities/FFT/NektarFFT.h>

#include <LibUtilities/BasicConst/NektarUnivConsts.hpp>
#include <LibUtilities/BasicUtils/NekManager.hpp>
#include <LibUtilities/BasicUtils/SharedArray.hpp>
#include <LibUtilities/Memory/NekMemoryManager.hpp>

namespace Nektar::LibUtilities
{

/**
 * FFTW-backed NektarFFT implementation. TData must be double or float,
 * the corresponding fftw_plan / fftwf_plan API is selected via if constexpr.
 *
 * The public aliases NekFFTW and NekFFTWFloat refer to the double and float
 * specialisations respectively.
 */
template <typename TData> class NekFFTWImpl : public NektarFFT<TData>
{
public:
    static std::shared_ptr<NektarFFT<TData>> create(int N)
    {
        return MemoryManager<NekFFTWImpl<TData>>::AllocateSharedPtr(N);
    }

    static std::string className;

    NekFFTWImpl(int N);
    ~NekFFTWImpl() override;

protected:
    Array<OneD, TData> m_FFTW_w;
    Array<OneD, TData> m_FFTW_w_inv;
    Array<OneD, TData> m_wsp;

    void *m_plan_backward;
    void *m_plan_forward;

    void v_FFTFwdTrans(TData *inarray, TData *outarray) override;
    void v_FFTBwdTrans(TData *inarray, TData *outarray) override;
};

// Backward-compatible aliases.
using NekFFTW      = NekFFTWImpl<double>;
using NekFFTWFloat = NekFFTWImpl<float>;

using NekFFTWSharedPtr = std::shared_ptr<NekFFTW>;

} // namespace Nektar::LibUtilities
#endif // NEKTAR_LIB_UTILIITIES_FFT_NEKFFTW_H
