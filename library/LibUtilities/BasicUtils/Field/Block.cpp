///////////////////////////////////////////////////////////////////////////////
//
// File: Block.cpp
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
// Description:
//
///////////////////////////////////////////////////////////////////////////////

#include <LibUtilities/BasicUtils/Field/Field.hpp>

namespace Nektar::LibUtilities
{

template <typename TData, FieldState TState>
template <typename MemSpace, typename MemAccess>
typename const_if<std::is_same_v<MemAccess, ReadOnly>, TData>::type *BlockAccessor<
    TData, TState>::GetPtr(const unsigned int streamID)
{
    // If not yet allocated, allocate contiguous host OR device memory
    // accross all MemoryRegion objects from m_field. Note: m_field is a
    // pointer to a Field object from which the current BlockAccessor object
    // belong to.
    Field<TData, TState>::template AllocateFieldStorage<MemSpace>(m_field);

    return m_memory_region.template GetPtr<MemSpace, MemAccess>(streamID);
}

template const uint8_t *BlockAccessor<uint8_t, FieldState::Phys>::GetPtr<
    NektarSpaces::HostSpace, ReadOnly>(const unsigned int streamID);
template uint8_t *BlockAccessor<uint8_t, FieldState::Phys>::GetPtr<
    NektarSpaces::HostSpace, WriteOnly>(const unsigned int streamID);
template uint8_t *BlockAccessor<uint8_t, FieldState::Phys>::GetPtr<
    NektarSpaces::HostSpace, ReadWrite>(const unsigned int streamID);
template const uint8_t *BlockAccessor<uint8_t, FieldState::Coeff>::GetPtr<
    NektarSpaces::HostSpace, ReadOnly>(const unsigned int streamID);
template uint8_t *BlockAccessor<uint8_t, FieldState::Coeff>::GetPtr<
    NektarSpaces::HostSpace, WriteOnly>(const unsigned int streamID);
template uint8_t *BlockAccessor<uint8_t, FieldState::Coeff>::GetPtr<
    NektarSpaces::HostSpace, ReadWrite>(const unsigned int streamID);
template const float *BlockAccessor<float, FieldState::Phys>::GetPtr<
    NektarSpaces::HostSpace, ReadOnly>(const unsigned int streamID);
template float *BlockAccessor<float, FieldState::Phys>::GetPtr<
    NektarSpaces::HostSpace, WriteOnly>(const unsigned int streamID);
template float *BlockAccessor<float, FieldState::Phys>::GetPtr<
    NektarSpaces::HostSpace, ReadWrite>(const unsigned int streamID);
template const float *BlockAccessor<float, FieldState::Coeff>::GetPtr<
    NektarSpaces::HostSpace, ReadOnly>(const unsigned int streamID);
template float *BlockAccessor<float, FieldState::Coeff>::GetPtr<
    NektarSpaces::HostSpace, WriteOnly>(const unsigned int streamID);
template float *BlockAccessor<float, FieldState::Coeff>::GetPtr<
    NektarSpaces::HostSpace, ReadWrite>(const unsigned int streamID);
template const double *BlockAccessor<double, FieldState::Phys>::GetPtr<
    NektarSpaces::HostSpace, ReadOnly>(const unsigned int streamID);
template double *BlockAccessor<double, FieldState::Phys>::GetPtr<
    NektarSpaces::HostSpace, WriteOnly>(const unsigned int streamID);
template double *BlockAccessor<double, FieldState::Phys>::GetPtr<
    NektarSpaces::HostSpace, ReadWrite>(const unsigned int streamID);
template const double *BlockAccessor<double, FieldState::Coeff>::GetPtr<
    NektarSpaces::HostSpace, ReadOnly>(const unsigned int streamID);
template double *BlockAccessor<double, FieldState::Coeff>::GetPtr<
    NektarSpaces::HostSpace, WriteOnly>(const unsigned int streamID);
template double *BlockAccessor<double, FieldState::Coeff>::GetPtr<
    NektarSpaces::HostSpace, ReadWrite>(const unsigned int streamID);
#if defined(NEKTAR_ENABLE_DEVICE)
template const uint8_t *BlockAccessor<uint8_t, FieldState::Phys>::GetPtr<
    NektarSpaces::DeviceSpace, ReadOnly>(const unsigned int streamID);
template uint8_t *BlockAccessor<uint8_t, FieldState::Phys>::GetPtr<
    NektarSpaces::DeviceSpace, WriteOnly>(const unsigned int streamID);
template uint8_t *BlockAccessor<uint8_t, FieldState::Phys>::GetPtr<
    NektarSpaces::DeviceSpace, ReadWrite>(const unsigned int streamID);
template const uint8_t *BlockAccessor<uint8_t, FieldState::Coeff>::GetPtr<
    NektarSpaces::DeviceSpace, ReadOnly>(const unsigned int streamID);
template uint8_t *BlockAccessor<uint8_t, FieldState::Coeff>::GetPtr<
    NektarSpaces::DeviceSpace, WriteOnly>(const unsigned int streamID);
template uint8_t *BlockAccessor<uint8_t, FieldState::Coeff>::GetPtr<
    NektarSpaces::DeviceSpace, ReadWrite>(const unsigned int streamID);
template const float *BlockAccessor<float, FieldState::Phys>::GetPtr<
    NektarSpaces::DeviceSpace, ReadOnly>(const unsigned int streamID);
template float *BlockAccessor<float, FieldState::Phys>::GetPtr<
    NektarSpaces::DeviceSpace, WriteOnly>(const unsigned int streamID);
template float *BlockAccessor<float, FieldState::Phys>::GetPtr<
    NektarSpaces::DeviceSpace, ReadWrite>(const unsigned int streamID);
template const float *BlockAccessor<float, FieldState::Coeff>::GetPtr<
    NektarSpaces::DeviceSpace, ReadOnly>(const unsigned int streamID);
template float *BlockAccessor<float, FieldState::Coeff>::GetPtr<
    NektarSpaces::DeviceSpace, WriteOnly>(const unsigned int streamID);
template float *BlockAccessor<float, FieldState::Coeff>::GetPtr<
    NektarSpaces::DeviceSpace, ReadWrite>(const unsigned int streamID);
template const double *BlockAccessor<double, FieldState::Phys>::GetPtr<
    NektarSpaces::DeviceSpace, ReadOnly>(const unsigned int streamID);
template double *BlockAccessor<double, FieldState::Phys>::GetPtr<
    NektarSpaces::DeviceSpace, WriteOnly>(const unsigned int streamID);
template double *BlockAccessor<double, FieldState::Phys>::GetPtr<
    NektarSpaces::DeviceSpace, ReadWrite>(const unsigned int streamID);
template const double *BlockAccessor<double, FieldState::Coeff>::GetPtr<
    NektarSpaces::DeviceSpace, ReadOnly>(const unsigned int streamID);
template double *BlockAccessor<double, FieldState::Coeff>::GetPtr<
    NektarSpaces::DeviceSpace, WriteOnly>(const unsigned int streamID);
template double *BlockAccessor<double, FieldState::Coeff>::GetPtr<
    NektarSpaces::DeviceSpace, ReadWrite>(const unsigned int streamID);
#endif

} // namespace Nektar::LibUtilities
