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
/// Each subclass owns the FULL per-call pipeline: xy derivatives via
/// blockOp[blk]->Apply(), then the z-FFT when nhomo > 1. This lets
/// PhysDerivOp::v_Apply make one unconditional call to m_zOp->Launch()
/// with no execution-space branches.
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

    /// \brief Execute the full xy+z pipeline for one Apply() call.
    ///
    /// Implementations must call blockOp[blk]->Apply() for all blocks
    /// (xy derivatives), and additionally compute dz when nhomo > 1.
    ///
    /// \param blockOp   Per-block xy operator list.
    /// \param in        Input field.
    /// \param out       Output field.
    /// \param nhomo     Number of homogeneous planes (1 = no z-FFT).
    /// \param blockNXY  Points per plane for each block.
    void Launch(std::vector<std::shared_ptr<PhysDerivBlockOp<TData>>> &blockOp,
                LibUtilities::Field<TData, FieldState::Phys> &in,
                LibUtilities::Field<TData, FieldState::Phys> &out,
                unsigned int nhomo, const std::vector<int> &blockNXY)
    {
        v_Launch(blockOp, in, out, nhomo, blockNXY);
    }

protected:
    virtual void v_Init(TData beta) = 0;

    virtual void v_Launch(
        std::vector<std::shared_ptr<PhysDerivBlockOp<TData>>> &blockOp,
        LibUtilities::Field<TData, FieldState::Phys> &in,
        LibUtilities::Field<TData, FieldState::Phys> &out, unsigned int nhomo,
        const std::vector<int> &blockNXY) = 0;

public:
    /// \brief Factory: create a z-op backend matching execStr.
    ///
    /// \param execStr       Backend selector: "" / "Serial" / "Host" always
    ///                      built; "Device" requires NEKTAR_ENABLE_CUDA;
    ///                      "DeviceDx" additionally requires
    ///                      NEKTAR_USE_CUFFTDX. Unknown or unbuilt strings
    ///                      trigger ASSERTL0.
    /// \param expansionList Used by the host backend to obtain the
    ///                      homogeneous FFT and transposition objects.
    static std::shared_ptr<PhysDerivZOpBase<TData>> Create(
        const std::string &execStr,
        const MultiRegions::ExpListSharedPtr &expansionList);
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
    const std::string &execStr,
    const MultiRegions::ExpListSharedPtr &expansionList)
{
    if (execStr.empty() || execStr == "Serial" || execStr == "AVX")
    {
        return std::make_shared<PhysDerivZOpHost<TData>>(expansionList);
    }

#if defined(NEKTAR_ENABLE_CUDA)
    if (execStr == "Device")
    {
        return std::make_shared<PhysDerivZOpDevice<TData>>();
    }

#if defined(NEKTAR_USE_CUFFTDX)
    if (execStr == "DeviceDx")
    {
        return std::make_shared<PhysDerivZOpDeviceDx<TData>>();
    }
#else
    if (execStr == "DeviceDx")
    {
        ASSERTL0(false, "PhysDerivZOp: execStr \"DeviceDx\" requires "
                        "NEKTAR_USE_CUFFTDX.");
    }
#endif // NEKTAR_USE_CUFFTDX
#endif // NEKTAR_ENABLE_CUDA

    return nullptr;
}

} // namespace Nektar::Operators
