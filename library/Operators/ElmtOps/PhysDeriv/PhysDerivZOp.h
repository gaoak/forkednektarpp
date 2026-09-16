///////////////////////////////////////////////////////////////////////////////
//
// File: PhysDerivZOp.h
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
// Description: Abstract base and factory for the z-derivative backend used by
// PhysDerivOp. Concrete implementations are in PhysDerivZOpHost.h (always
// built) and PhysDerivZOpDevice.h (NEKTAR_ENABLE_CUDA only).
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "LibUtilities/BasicUtils/Field/Field.hpp"
#include "Operators/ElmtOps/PhysDeriv/PhysDerivBlockOp.hpp"

#include <MultiRegions/ExpList.h>

namespace Nektar::Operators
{

/// \brief Abstract base for the homogeneous z-derivative backend used by
/// PhysDerivOp.
///
/// Subclasses compute only the z-FFT. The xy derivatives are applied by
/// PhysDerivOp::v_Apply before it calls Launch().
///
/// Concrete implementations:
///   - PhysDerivZOpHost (PhysDerivZOpHost.h) - always built, FFTW path.
///   - PhysDerivZOpDevice (PhysDerivZOpDevice.h) - cuFFT path, requires
///     NEKTAR_ENABLE_CUDA.
///   - PhysDerivZOpDeviceDx (PhysDerivZOpDevice.h) - cuFFTDx path, requires
///     NEKTAR_ENABLE_CUDA and NEKTAR_USE_CUFFTDX.
///
/// \tparam TData Floating-point type (float or double).
template <typename TData> class PhysDerivZOpBase
{
public:
    virtual ~PhysDerivZOpBase() = default;

    /// \brief Initialise backend state. Called once after Create() when
    ///        nhomo > 1.
    /// \param beta  Wavenumber factor 2*pi/Lz.
    void Init(TData beta)
    {
        v_Init(beta);
    }

    /// \brief Compute the z-derivative. Called only when nhomo > 1, after
    ///        the xy derivatives have been applied.
    ///
    /// \param in   Input field.
    /// \param out  Output field.
    void Launch(LibUtilities::Field<TData, FieldState::Phys> &in,
                LibUtilities::Field<TData, FieldState::Phys> &out)
    {
        v_Launch(in, out);
    }

protected:
    virtual void v_Init(TData beta) = 0;

    virtual void v_Launch(
        LibUtilities::Field<TData, FieldState::Phys> &in,
        LibUtilities::Field<TData, FieldState::Phys> &out) = 0;

public:
    /// \brief Factory: create a z-op backend matching execStr.
    ///
    /// \param expansionList Used by the host backend to obtain the
    ///                      homogeneous FFT and transposition objects.
    /// \param execStr       Backend selector; "" resolves it from the
    ///                      session. "Device" takes the CUDA path (cuFFTDx
    ///                      when NEKTAR_USE_CUFFTDX, otherwise cuFFT);
    ///                      everything else -- including "Device" on a
    ///                      non-CUDA device build -- takes the host FFTW
    ///                      path. Never returns null.
    static std::shared_ptr<PhysDerivZOpBase<TData>> Create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::string &execStr);
};

} // namespace Nektar::Operators

// Always-built host backend.
#include "PhysDerivZOpHost.h"

// CUDA-only GPU backends. On HIP/SYCL this include is present but the file's
// NEKTAR_ENABLE_CUDA guard keeps its content empty, so no CUDA types leak out.
#if defined(NEKTAR_ENABLE_DEVICE)
#include "PhysDerivZOpDevice.h"
#endif

namespace Nektar::Operators
{

template <typename TData>
std::shared_ptr<PhysDerivZOpBase<TData>> PhysDerivZOpBase<TData>::Create(
    const MultiRegions::ExpListSharedPtr &expansionList,
    const std::string &execStr)
{
    auto session = expansionList->GetSession();

    std::string execStr0 =
        (execStr == "") ? Operator<TData>::GetOpExecSpace(session) : execStr;

    // "Device" takes a CUDA z-FFT where the build provides one; a device
    // build with no z-FFT of its own (HIP/SYCL) falls through to the host
    // path below.
    if (execStr0 == "Device")
    {
#if defined(NEKTAR_ENABLE_CUDA)
#if defined(NEKTAR_USE_CUFFTDX)
        return std::make_shared<PhysDerivZOpDeviceDx<TData>>();
#else
        return std::make_shared<PhysDerivZOpDevice<TData>>();
#endif // NEKTAR_USE_CUFFTDX
#endif // NEKTAR_ENABLE_CUDA
    }

    // Serial/AVX, and any device build without its own z-FFT. Reading the
    // blocks through GetPtr<HostSpace> pulls the xy results back from the
    // device first, so this stays correct -- just not fast.
    return std::make_shared<PhysDerivZOpHost<TData>>(expansionList);
}

} // namespace Nektar::Operators
