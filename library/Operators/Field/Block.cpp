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

#include <MultiRegions/ExpList.h>
#include <MultiRegions/ExpListHomogeneous1D.h>
#include <MultiRegions/ExpListHomogeneous2D.h>

#include <Operators/Field/Block.hpp>

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

Collections::CollectionVector GetCollection(
    MultiRegions::ExpListSharedPtr expansionList)
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
        return explistHomo1D->GetPlane(0)->GetCollections();
    }
    else if (explistHomo2D)
    {
        return explistHomo2D->GetLine(0)->GetCollections();
    }
    else
    {
        return expansionList->GetCollections();
    }
}

/**
 * @brief Get the BlockAttributes for a given field state from an ExpList.
 * This method basically captures identical elements that are contiguously
 * stored in the ExpList and group them into blocks. Padding elements will
 * also be set based on given vector width.
 *
 * @param state     Field state to query.
 * @param explist   Expansion list to query.
 * @param interleave_width Vector width to use for the field.
 * @return std::vector<BlockAttributes>
 */
template <typename TData, FieldState TState>
std::vector<BlockAttributes<TState>> GetBlockAttributes(
    const MultiRegions::ExpListSharedPtr explist,
    const unsigned interleave_width)
{
    // Use maximum vector width for back-ends interoperability.
    const auto vector_width = NektarSpaces::max_vector_width<TData>::value;

    std::vector<BlockAttributes<TState>> blockAttr;

    auto colls = GetCollection(explist);
    for (auto &coll : colls)
    {
        auto expPtr               = coll.GetExpVector()[0];
        const size_t num_elements = coll.GetExpVector().size();
        const unsigned int ndata  = TState == FieldState::Phys
                                        ? expPtr->GetTotPoints()
                                        : expPtr->GetNcoeffs();
        size_t num_elements_with_padding =
            ((num_elements + vector_width - 1) / vector_width) * vector_width;
        blockAttr.push_back(
            {num_elements, num_elements_with_padding, ndata, interleave_width});
    }

    return blockAttr;
}

template std::vector<BlockAttributes<FieldState::Phys>> GetBlockAttributes<
    double, FieldState::Phys>(const MultiRegions::ExpListSharedPtr explist,
                              const unsigned interleave_width);
template std::vector<BlockAttributes<FieldState::Coeff>> GetBlockAttributes<
    double, FieldState::Coeff>(const MultiRegions::ExpListSharedPtr explist,
                               const unsigned interleave_width);

template std::vector<BlockAttributes<FieldState::Phys>> GetBlockAttributes<
    float, FieldState::Phys>(const MultiRegions::ExpListSharedPtr explist,
                             const unsigned interleave_width);
template std::vector<BlockAttributes<FieldState::Coeff>> GetBlockAttributes<
    float, FieldState::Coeff>(const MultiRegions::ExpListSharedPtr explist,
                              const unsigned interleave_width);

template std::vector<BlockAttributes<FieldState::Phys>> GetBlockAttributes<
    int, FieldState::Phys>(const MultiRegions::ExpListSharedPtr explist,
                           const unsigned interleave_width);
template std::vector<BlockAttributes<FieldState::Coeff>> GetBlockAttributes<
    int, FieldState::Coeff>(const MultiRegions::ExpListSharedPtr explist,
                            const unsigned interleave_width);

template std::vector<BlockAttributes<FieldState::Phys>> GetBlockAttributes<
    unsigned int, FieldState::Phys>(
    const MultiRegions::ExpListSharedPtr explist,
    const unsigned interleave_width);
template std::vector<BlockAttributes<FieldState::Coeff>> GetBlockAttributes<
    unsigned int, FieldState::Coeff>(
    const MultiRegions::ExpListSharedPtr explist,
    const unsigned interleave_width);

template std::vector<BlockAttributes<FieldState::Phys>> GetBlockAttributes<
    size_t, FieldState::Phys>(const MultiRegions::ExpListSharedPtr explist,
                              const unsigned interleave_width);
template std::vector<BlockAttributes<FieldState::Coeff>> GetBlockAttributes<
    size_t, FieldState::Coeff>(const MultiRegions::ExpListSharedPtr explist,
                               const unsigned interleave_width);

} // namespace Nektar::Operators
