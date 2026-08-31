///////////////////////////////////////////////////////////////////////////////
//
// File: NektarFFT.h
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
// Description: Header file for the Fast Fourier Transform class in Nektar++
//
///////////////////////////////////////////////////////////////////////////////

#ifndef NEKTAR_LIB_UTILIITIES_FFT_NEKTARFFT_H
#define NEKTAR_LIB_UTILIITIES_FFT_NEKTARFFT_H

#include <LibUtilities/BasicConst/NektarUnivTypeDefs.hpp>
#include <LibUtilities/BasicUtils/NekFactory.hpp>
#include <LibUtilities/BasicUtils/SharedArray.hpp>
#include <LibUtilities/LibUtilitiesDeclspec.h>

namespace Nektar::LibUtilities
{

template <typename TData> class NektarFFT;

// Backward-compatible shared pointer for the double specialisation.
using NektarFFTSharedPtr = std::shared_ptr<NektarFFT<double>>;

// Convenience aliases for the two supported precisions.
using NektarFFTDouble = NektarFFT<double>;
using NektarFFTFloat  = NektarFFT<float>;

// Factory types, one per precision.
using NektarFFTFactory      = NekFactory<std::string, NektarFFT<double>, int>;
using NektarFFTFloatFactory = NekFactory<std::string, NektarFFT<float>, int>;

LIB_UTILITIES_EXPORT NektarFFTFactory &GetNektarFFTFactory();
LIB_UTILITIES_EXPORT NektarFFTFloatFactory &GetNektarFFTFloatFactory();

/**
 * Base class for FFT implementations. Templated on the real scalar type TData
 * (double or float). Derived classes implement v_FFTFwdTrans and
 * v_FFTBwdTrans.
 */
template <typename TData> class NektarFFT
{
public:
    NektarFFT(int N) : m_N(N)
    {
    }

    virtual ~NektarFFT() = default;

    /**
     * m_N is the number of real points in the transform.
     */
    int m_N;

    // Array overloads: dispatch to the raw-pointer virtuals.

    void FFTFwdTrans(Array<OneD, TData> &phys, Array<OneD, TData> &coef)
    {
        v_FFTFwdTrans(phys.data(), coef.data());
    }

    void FFTFwdTrans(TData *phys, TData *coef)
    {
        v_FFTFwdTrans(phys, coef);
    }

    void FFTBwdTrans(Array<OneD, TData> &coef, Array<OneD, TData> &phys)
    {
        v_FFTBwdTrans(coef.data(), phys.data());
    }

    void FFTBwdTrans(TData *coef, TData *phys)
    {
        v_FFTBwdTrans(coef, phys);
    }

protected:
    virtual void v_FFTFwdTrans(TData *phys, TData *coef) = 0;
    virtual void v_FFTBwdTrans(TData *coef, TData *phys) = 0;
};

} // namespace Nektar::LibUtilities
// end of namespace Nektar
#endif // NEKTAR_LIB_UTILIITIES_FFT_NEKTARFFT_H
