///////////////////////////////////////////////////////////////////////////////
//
// File: ImplicitSDCOp.hpp
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

#include "Operators/TimeOps/SDC/SDCOp.hpp"
#include <LibUtilities/Foundations/Points.h>
#include <LibUtilities/Polylib/Polylib.h>

namespace Nektar::Operators
{

// ImplicitSDC base class
// Defines the apply operator to enforce apply parameter types
template <typename TData> class ImplicitSDCOp : public SDCOp<TData>
{
public:
    static std::shared_ptr<ImplicitSDCOp<TData>> Create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const unsigned int &order = 0, const std::string &variant = "",
        const std::vector<TData> freeParams = std::vector<TData>{},
        const std::string &execStr          = "")
    {
        return std::dynamic_pointer_cast<ImplicitSDCOp<TData>>(
            TimeOp<TData>::Create(expansionList, name, order, variant,
                                  freeParams, execStr));
    }

    static inline const std::string name = "ImplicitSDC";

protected:
    ImplicitSDCOp(const MultiRegions::ExpListSharedPtr &expansionList)
        : SDCOp<TData>(expansionList)
    {
    }

    ~ImplicitSDCOp() override = default;
};

} // namespace Nektar::Operators
