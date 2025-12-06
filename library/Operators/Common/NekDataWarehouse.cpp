///////////////////////////////////////////////////////////////////////////////
//
// File: NekDataWarehouse.cpp
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
// Description: DataWarehouse pattern class for Nektar
//
///////////////////////////////////////////////////////////////////////////////

#include "Operators/Common/NekDataWarehouse.hpp"

#include <MultiRegions/ExpListHomogeneous1D.h>
#include <MultiRegions/ExpListHomogeneous2D.h>

namespace Nektar::Operators
{

// Helper function
Collections::Collection GetCollection(
    MultiRegions::ExpListSharedPtr expansionList, unsigned int block_idx)
{
    MultiRegions::ExpListSharedPtr tmp;
    auto explistHomo1D =
        std::dynamic_pointer_cast<MultiRegions::ExpListHomogeneous1D>(
            expansionList);
    auto explistHomo2D =
        std::dynamic_pointer_cast<MultiRegions::ExpListHomogeneous2D>(
            expansionList);
    if (explistHomo1D)
    {
        tmp = explistHomo1D->GetPlane(0);
    }
    else if (explistHomo2D)
    {
        tmp = explistHomo2D->GetLine(0);
    }
    else
    {
        tmp = expansionList;
    }

    return tmp->GetCollections()[block_idx];
}

} // namespace Nektar::Operators
