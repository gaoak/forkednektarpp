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

#if defined(_MSC_VER)
#undef max
#undef min
#endif

namespace Nektar::Operators
{

class ModeIndexCreator;

class ModeIndexKey : public BaseKey
{
    friend class ModeIndexCreator;

public:
    using creator = ModeIndexCreator;
    typedef unsigned int value_type;

    ~ModeIndexKey() override = default;

    ModeIndexKey(const LibUtilities::ShapeType shapeType,
                 const unsigned int nm0, const unsigned int nm1,
                 const unsigned int mode)
        : m_shapeType(shapeType), m_nm0(nm0), m_nm1(nm1), m_mode(mode)
    {
        hash_combine(m_hash, m_shapeType, m_nm0, m_nm1, m_nm2, m_mode,
                     "ModeIndexKey");
    }

    ModeIndexKey(const LibUtilities::ShapeType shapeType,
                 const unsigned int nm0, const unsigned int nm1,
                 const unsigned int nm2, const unsigned int mode)
        : m_shapeType(shapeType), m_nm0(nm0), m_nm1(nm1), m_nm2(nm2),
          m_mode(mode)
    {
        hash_combine(m_hash, m_shapeType, m_nm0, m_nm1, m_nm2, m_mode,
                     "ModeIndexKey");
    }

private:
    LibUtilities::ShapeType m_shapeType;
    unsigned int m_nm0 = 0;
    unsigned int m_nm1 = 0;
    unsigned int m_nm2 = 0;
    unsigned int m_mode;
};

class ModeIndexCreator : public DataCreatorClass
{
public:
    ~ModeIndexCreator() override = default;

    using value_type = ModeIndexKey::value_type;

    template <typename MemSpace>
    MemoryRegion<value_type> Create(const ModeIndexKey &modeIndexKey,
                                    const unsigned int alignment)
    {
        const auto shapeType = modeIndexKey.m_shapeType;

        switch (shapeType)
        {
            case LibUtilities::Tri:
            case LibUtilities::NodalTri:
            {
                const auto mode = modeIndexKey.m_mode;
                const auto nm0  = modeIndexKey.m_nm0;
                const auto nm1  = modeIndexKey.m_nm1;
                const auto nm01 =
                    LibUtilities::GetNumberOfCoefficients(shapeType, nm0, nm1);

                auto index = MemoryRegion<value_type>::Create(nm01, alignment);
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
            case LibUtilities::NodalTet:
            {
                const auto mode  = modeIndexKey.m_mode;
                const auto nm0   = modeIndexKey.m_nm0;
                const auto nm1   = modeIndexKey.m_nm1;
                const auto nm2   = modeIndexKey.m_nm2;
                const auto nmTot = LibUtilities::GetNumberOfCoefficients(
                    shapeType, nm0, nm1, nm2);
                const unsigned int nm01 = (2u * nm1 - nm0 + 1u) * nm0 / 2u;
                if (mode == 0 || mode == 3)
                {
                    auto index =
                        MemoryRegion<value_type>::Create(nm01, alignment);
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
                    auto index =
                        MemoryRegion<value_type>::Create(nmTot, alignment);
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
                    return MemoryRegion<value_type>::Create(0, alignment);
                }
            }
            break;
            case LibUtilities::Prism:
            case LibUtilities::NodalPrism:
            {
                const auto mode  = modeIndexKey.m_mode;
                const auto nm0   = modeIndexKey.m_nm0;
                const auto nm1   = modeIndexKey.m_nm1;
                const auto nm2   = modeIndexKey.m_nm2;
                const auto nmTot = LibUtilities::GetNumberOfCoefficients(
                    shapeType, nm0, nm1, nm2);

                auto index = MemoryRegion<value_type>::Create(nmTot, alignment);
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
                const auto mode  = modeIndexKey.m_mode;
                const auto nm0   = modeIndexKey.m_nm0;
                const auto nm1   = modeIndexKey.m_nm1;
                const auto nm2   = modeIndexKey.m_nm2;
                const auto nmTot = LibUtilities::GetNumberOfCoefficients(
                    shapeType, nm0, nm1, nm2);

                auto index = MemoryRegion<value_type>::Create(nmTot, alignment);
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
                return MemoryRegion<value_type>::Create(0, alignment);
                break;
        }
    }

    inline static const std::string m_name = "ModeIndexCreator";
};

} // namespace Nektar::Operators
