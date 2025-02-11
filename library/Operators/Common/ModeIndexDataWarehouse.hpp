///////////////////////////////////////////////////////////////////////////////
//
// File: ModeIndexDataWarehouse.hpp
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

#pragma once

#include "Operators/Common/NekDataWarehouse.hpp"

namespace Nektar::Operators
{

class ModeIndexCreator;

class ModeIndexKey : public BaseKey
{
public:
    using creator = ModeIndexCreator;

    ~ModeIndexKey() override = default;
    ModeIndexKey(const LocalRegions::ExpansionSharedPtr expPtr,
                 const unsigned int mode)
        : m_expPtr(expPtr), m_mode(mode)
    {
        hash_combine(m_hash, m_expPtr, m_mode, m_name);
    }

    LocalRegions::ExpansionSharedPtr m_expPtr;
    unsigned int m_mode;
    typedef unsigned int m_data_type;

private:
    inline static const std::string m_name = "ModeIndexKey";
};

class ModeIndexCreator : public DataCreatorClass
{
public:
    ~ModeIndexCreator() override = default;

    template <typename MemSpace, typename TData>
    MemoryRegion<TData> Create(const ModeIndexKey &modeIndexKey,
                               const unsigned int alignment)
    {
        const auto expPtr = modeIndexKey.m_expPtr;
        const auto mode   = modeIndexKey.m_mode;

        switch (expPtr->DetShapeType())
        {
            case LibUtilities::Tri:
            {
                const auto nm0          = expPtr->GetBasisNumModes(0);
                const auto nm1          = expPtr->GetBasisNumModes(1);
                const unsigned int nm01 = (2u * nm1 - nm0 + 1u) * nm0 / 2u;

                auto index = MemoryRegion<TData>::template Create<MemSpace>(
                    nm01, alignment);
                auto ptr =
                    index.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
                for (unsigned int p = 0, mode_pq = 0; p < nm0; p++)
                {
                    for (unsigned int q = 0; q < nm1 - p; q++, mode_pq++)
                    {
                        if (mode == 0)
                        {
                            ptr[mode_pq] = p;
                        }
                        else if (mode == 1)
                        {
                            ptr[mode_pq] = q;
                        }
                        else
                        {
                            NEKERROR(ErrorUtil::efatal,
                                     "invalid data requested.");
                        }
                    }
                }

                return index;
            }
            break;
            case LibUtilities::Tet:
            {
                const auto nmTot        = expPtr->GetNcoeffs();
                const auto nm0          = expPtr->GetBasisNumModes(0);
                const auto nm1          = expPtr->GetBasisNumModes(1);
                const auto nm2          = expPtr->GetBasisNumModes(2);
                const unsigned int nm01 = (2u * nm1 - nm0 + 1u) * nm0 / 2u;
                if (mode == 0 || mode == 3)
                {
                    auto index = MemoryRegion<TData>::template Create<MemSpace>(
                        nm01, alignment);
                    auto ptr = index.template GetPtr<NektarSpaces::HostSpace,
                                                     WriteOnly>();
                    for (unsigned int p = 0, mode_pq = 0; p < nm0; p++)
                    {
                        for (unsigned int q = 0; q < nm1 - p; q++, mode_pq++)
                        {
                            if (mode == 0)
                            {
                                ptr[mode_pq] = p;
                            }
                            else if (mode == 3)
                            {
                                ptr[mode_pq] = q;
                            }
                        }
                    }

                    return index;
                }
                else if (mode == 1 || mode == 2)
                {
                    auto index = MemoryRegion<TData>::template Create<MemSpace>(
                        nmTot, alignment);
                    auto ptr = index.template GetPtr<NektarSpaces::HostSpace,
                                                     WriteOnly>();
                    for (unsigned int p = 0, mode_pq = 0, mode_pqr = 0; p < nm0;
                         p++)
                    {
                        for (unsigned int q = 0; q < nm1 - p; q++, mode_pq++)
                        {
                            for (unsigned int r = 0; r < nm2 - p - q;
                                 r++, mode_pqr++)
                            {
                                if (mode == 1)
                                {
                                    ptr[mode_pqr] = p;
                                }
                                else if (mode == 2)
                                {
                                    ptr[mode_pqr] = q;
                                }
                            }
                        }
                    }

                    return index;
                }
                else
                {
                    NEKERROR(ErrorUtil::efatal, "invalid data requested.");
                    return MemoryRegion<TData>::template Create<MemSpace>(
                        0, alignment);
                }
            }
            break;
            case LibUtilities::Prism:
            {
                const auto nmTot = expPtr->GetNcoeffs();
                const auto nm0   = expPtr->GetBasisNumModes(0);
                const auto nm1   = expPtr->GetBasisNumModes(1);
                const auto nm2   = expPtr->GetBasisNumModes(2);

                auto index = MemoryRegion<TData>::template Create<MemSpace>(
                    nmTot, alignment);
                auto ptr =
                    index.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
                for (unsigned int p = 0, mode_pqr = 0; p < nm0; p++)
                {
                    for (unsigned int q = 0u; q < nm1; q++)
                    {
                        for (unsigned int r = 0u; r < nm2 - p; r++, mode_pqr++)
                        {
                            if (mode == 0)
                            {
                                ptr[mode_pqr] = p;
                            }
                            else if (mode == 1)
                            {
                                ptr[mode_pqr] = q;
                            }
                            else if (mode == 2)
                            {
                                ptr[mode_pqr] = r;
                            }
                            else
                            {
                                NEKERROR(ErrorUtil::efatal,
                                         "invalid data requested.");
                            }
                        }
                    }
                }

                return index;
            }
            break;
            case LibUtilities::Pyr:
            {
                const auto nmTot = expPtr->GetNcoeffs();
                const auto nm0   = expPtr->GetBasisNumModes(0);
                const auto nm1   = expPtr->GetBasisNumModes(1);
                const auto nm2   = expPtr->GetBasisNumModes(2);

                auto index = MemoryRegion<TData>::template Create<MemSpace>(
                    nmTot, alignment);
                auto ptr =
                    index.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
                for (unsigned int p = 0, mode_pqr = 0; p < nm0; p++)
                {
                    for (unsigned int q = 0u; q < nm1; q++)
                    {
                        for (unsigned int r = 0; r < nm2 - std::max(p, q);
                             r++, mode_pqr++)
                        {
                            if (mode == 0)
                            {
                                ptr[mode_pqr] = p;
                            }
                            else if (mode == 1)
                            {
                                ptr[mode_pqr] = q;
                            }
                            else
                            {
                                NEKERROR(ErrorUtil::efatal,
                                         "invalid data requested.");
                            }
                        }
                    }
                }

                return index;
            }
            break;
            default:
                NEKERROR(ErrorUtil::efatal, "invalid data requested.");
                return MemoryRegion<TData>::template Create<MemSpace>(
                    0, alignment);
                break;
        }
    }

    inline static const std::string m_name = "ModeIndexCreator";
};

} // namespace Nektar::Operators
