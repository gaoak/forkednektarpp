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

#include "Operators/Common/DataWarehouse/NekDataWarehouse.hpp"

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
    LibUtilities::MemoryRegion<value_type> Create(
        const ModeIndexKey &modeIndexKey);

    inline static const std::string m_name = "ModeIndexCreator";
};

} // namespace Nektar::Operators
