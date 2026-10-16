///////////////////////////////////////////////////////////////////////////////
//
// File: LocalToGlobalDataWarehouseDef.hpp
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
// Description: Creation of the local-to-global assembly tables.
//
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <algorithm>
#include <limits>
#include <typeindex>

#include <LibUtilities/BasicUtils/Field/Field.hpp>
#include <MultiRegions/DataWarehouse/LocalToGlobalDataWarehouse.hpp>

#include <MultiRegions/ContField.h>

namespace Nektar::MultiRegions
{

std::vector<AssemblyMapCGSharedPtr> &LocalToGlobalDataCreator::GetAssemblyMap(
    const std::vector<std::string> &components)
{
    size_t hash = 0;
    for (const auto &component : components)
    {
        hash_combine(hash, component);
    }

    auto it = m_assemblyMaps.find(hash);
    if (it != m_assemblyMaps.end())
    {
        return it->second;
    }

    auto session = this->m_expansionList->GetSession();
    auto graph   = this->m_expansionList->GetGraph();
    std::vector<AssemblyMapCGSharedPtr> assemblyMap;

    if (components.empty())
    {
        auto contfield =
            std::dynamic_pointer_cast<ContField>(this->m_expansionList);
        assemblyMap.push_back(contfield->GetLocalToGlobalMap());
    }
    else
    {
        auto expContField =
            std::dynamic_pointer_cast<ContField>(this->m_expansionList);

        for (const auto &component : components)
        {
            if (expContField &&
                expContField->GetLocalToGlobalMap()->GetVariable() == component)
            {
                assemblyMap.push_back(expContField->GetLocalToGlobalMap());
                continue;
            }

            auto contfield = MemoryManager<ContField>::AllocateSharedPtr(
                session, graph, component, true, false,
                Collections::eNoCollection);
            assemblyMap.push_back(contfield->GetLocalToGlobalMap());
        }
    }

    return m_assemblyMaps.emplace(hash, std::move(assemblyMap)).first->second;
}

/**
 * @brief Position of the local coefficient @p lid, numbered as in a field
 * with every block at interleave width 1, in the same field with each block
 * `blk` interleaved at `fieldInterleaveWidths[blk]`.
 */
template <FieldState TState>
static unsigned InterleavedIndex(
    const unsigned lid,
    const std::vector<LibUtilities::BlockAttributes<TState>> &blockAttr,
    const unsigned numComp, const std::vector<unsigned> &fieldInterleaveWidths)
{
    unsigned blk       = 0;
    unsigned blkOffset = 0;
    while (lid >= blkOffset + blockAttr[blk].CompSize() * numComp)
    {
        blkOffset += blockAttr[blk].CompSize() * numComp;
        ++blk;
    }

    const unsigned width    = fieldInterleaveWidths[blk];
    const unsigned ndata    = blockAttr[blk].GetNumData();
    const unsigned compSize = blockAttr[blk].CompSize();
    const unsigned blkLid   = lid - blkOffset;
    const unsigned nc       = blkLid / compSize;
    const unsigned elmt     = (blkLid % compSize) / ndata;
    const unsigned ilane    = elmt % width;
    const unsigned iwarp    = elmt / width;

    return blkOffset + nc * compSize + iwarp * ndata * width +
           (blkLid % ndata) * width + ilane;
}

/**
 * @brief Interior gather-scatter table, built once per set of parameters.
 */
template <typename TPadding>
const std::vector<unsigned> &LocalToGlobalDataCreator::GetGSInfo(
    bool zeroDir, const std::vector<std::string> &components, unsigned width,
    const std::vector<unsigned> &fieldInterleaveWidths)
{
    size_t hash = 0;
    hash_combine(hash, std::type_index(typeid(TPadding)), zeroDir, width);
    for (const auto &component : components)
    {
        hash_combine(hash, component);
    }
    for (const auto &fieldInterleaveWidth : fieldInterleaveWidths)
    {
        hash_combine(hash, fieldInterleaveWidth);
    }

    auto it = m_gsInfo.find(hash);
    if (it != m_gsInfo.end())
    {
        return it->second;
    }

    // Get Local To Global Map.
    auto &loc2glo = GetAssemblyMap(components);
    auto numComp  = loc2glo.size();

    Array<OneD, unsigned> ndir(numComp);
    for (unsigned i = 0; i < numComp; ++i)
    {
        ndir[i] = loc2glo[i]->GetNumGlobalDirBndCoeffs();
    }

    auto l2gmap0 = loc2glo[0]->GetLocalToGlobalMap();

    // set for fast lookup of parallel gids
    std::unordered_set<size_t> parallel_gid(loc2glo[0]->GetSREntries().begin(),
                                            loc2glo[0]->GetSREntries().end());

    // here we are using double as the type since this is what the datatype
    // of the assembled data is assumed to be. Not sure what we shoudl do it
    // is float however but legacy code is not set up for this either
    auto blockAttr =
        GetBlockAttributes<TPadding, FieldState::Coeff>(this->m_expansionList);

    auto nblks = blockAttr.size();
    std::map<unsigned, std::vector<std::pair<unsigned, unsigned>>> GloToLoc;

    // Set up a map of the local ids and blk ids that are
    // associated with the global id of this expansion excluding point on
    // parallel boundaries
    unsigned coeff_offset = 0;
    for (unsigned blk = 0; blk < nblks; ++blk)
    {
        auto blksize =
            blockAttr[blk].GetNumData() * blockAttr[blk].GetNumElements();

        // Gather local id in block that share the same global id.
        for (unsigned lid = 0; lid < blksize; ++lid)
        {
            auto gid = l2gmap0[lid + coeff_offset];
            if (parallel_gid.count(gid) == 0) // not a parallel gid
            {
                GloToLoc[gid].push_back(
                    std::pair<unsigned, unsigned>(blk, lid));
            }
        }
        coeff_offset += blksize;
    }

    // At this point we have a list of global ids not on parallel boundaries
    // and the related blocks and ids within the blocks that are connected
    // to this global id. Next assemble a list of gids and components that
    // should be ordered in the final output.
    std::map<unsigned, std::pair<unsigned, unsigned>> SortVals;
    std::vector<unsigned> BlkOffset(nblks);
    std::set<unsigned> gidDone;

    unsigned nvalstot = 0;
    unsigned nidx     = 0;
    coeff_offset      = 0;

    for (unsigned blk = 0; blk < nblks; ++blk)
    {
        unsigned blksize =
            blockAttr[blk].GetNumElements() * blockAttr[blk].GetNumData();
        for (int nc = 0; nc < numComp; ++nc)
        {
            for (int lid = 0; lid < blksize; ++lid)
            {
                unsigned gid = l2gmap0[lid + coeff_offset];

                if ((gidDone.count(gid * numComp + nc) == 0) &&
                    (GloToLoc.count(gid)))
                {
                    unsigned val = GloToLoc[gid].size();
                    if (zeroDir)
                    {
                        // Add point if valance is more than one
                        // or if global dof < ndir on this component
                        if ((val > 1) || (loc2glo[nc]->GetLocalToGlobalMap(
                                              lid + coeff_offset) < ndir[nc]))
                        {
                            nidx += val;
                            SortVals[nvalstot++] =
                                std::pair<unsigned, unsigned>(gid, nc);
                        }
                    }
                    else
                    {
                        // Only add  points with valence more than 1
                        if (val > 1)
                        {
                            nidx += val;
                            SortVals[nvalstot++] =
                                std::pair<unsigned, unsigned>(gid, nc);
                        }
                    }
                    gidDone.insert(gid * numComp + nc);
                }
            }
        }
        // set up block offset to allow for local id determineation
        if (blk)
        {
            BlkOffset[blk] =
                BlkOffset[blk - 1] + blockAttr[blk - 1].CompSize() * numComp;
        }
        else
        {
            BlkOffset[blk] = 0;
        }
        coeff_offset += blksize;
    }

    // Order the values by the position of their first local copy in the
    // field, at its interleave widths, so that consecutive values address
    // neighbouring coefficients: consecutive coefficients of an element at
    // width 1, the same coefficient of consecutive elements when interleaved.
    // If width > 1 order by descending valence first, so that the values of
    // a width block share their valence wherever possible and
    // SortVals[i*width] carries the largest valence of its block, which sizes
    // the index and sign storage of the block.
    ASSERTL1(fieldInterleaveWidths.size() == nblks,
             "One interleave width per block is required");

    struct SortKey
    {
        unsigned valence;
        unsigned position;
        std::pair<unsigned, unsigned> val;
    };
    std::vector<SortKey> keys;
    keys.reserve(nvalstot);
    for (auto &sort : SortVals)
    {
        auto gid = sort.second.first;
        auto nc  = sort.second.second;

        unsigned position = std::numeric_limits<unsigned>::max();
        for (auto &[blk, lid] : GloToLoc[gid])
        {
            position = std::min(
                position,
                InterleavedIndex(lid + BlkOffset[blk] +
                                     blockAttr[blk].CompSize() * nc,
                                 blockAttr, numComp, fieldInterleaveWidths));
        }
        keys.push_back(SortKey{static_cast<unsigned>(GloToLoc[gid].size()),
                               position, sort.second});
    }

    std::sort(keys.begin(), keys.end(),
              [width](const SortKey &a, const SortKey &b) {
                  if (width > 1 && a.valence != b.valence)
                  {
                      return a.valence > b.valence;
                  }
                  return a.position < b.position;
              });

    for (unsigned i = 0; i < nvalstot; ++i)
    {
        SortVals[i] = keys[i].val;
    }

    // Decalare memory for all local to global informaiton.
    std::vector<unsigned> gsInfo(nvalstot + 2 + nidx);
    auto ptr = gsInfo.data();

    // Fill the pointer
    //  ptr[0] = nvals = number global points i
    //  ptr[1] = starting offset of local index to be assembled
    //  ptr[2] = starting offset of local index to be assembled
    //  ptr[3] = ....
    //  ptr[nvals+1] = total number of local indices to be assembled
    //  ptr[nvals+2] = local index of point to be assembled
    //  ptr[nvals+3] = local index of point to be assembled
    //  ptr[nvals+...] =  ...
    //  ptr[nvals+1+ ptr[nvals+1]] = last  index of point to be assembled
    auto ptr1     = ptr + nvalstot + 2;
    ptr[0]        = nvalstot;
    ptr[1]        = 0;
    unsigned cnt  = 2;
    unsigned cnt1 = 0;

    for (auto &sort : SortVals)
    {
        auto gid = sort.second.first;
        auto nc  = sort.second.second;
        ptr[cnt] = ptr[cnt - 1]; // start new offset

        for (auto &iter : GloToLoc[gid])
        {
            unsigned blk = iter.first;
            unsigned lid = iter.second;

            // offset local id by Block offset and num component * each
            // block size;
            lid += BlkOffset[blk] + blockAttr[blk].CompSize() * nc;

            ptr1[cnt1++] = lid; // set index
            ptr[cnt] += 1;      // add one to offset
        }
        cnt++;
    }

    return m_gsInfo.emplace(hash, std::move(gsInfo)).first->second;
}

template <typename MemSpace, typename TPadding>
LibUtilities::MemoryRegion<
    typename DeviceLocalToGlobalNumGidsKey<TPadding>::value_type>
LocalToGlobalDataCreator::Create(
    const DeviceLocalToGlobalNumGidsKey<TPadding> &LocToGloKey)
{
    using value_type =
        typename DeviceLocalToGlobalNumGidsKey<TPadding>::value_type;

    const auto &gsinfo = GetGSInfo<TPadding>(
        LocToGloKey.m_zeroDir, LocToGloKey.m_components, LocToGloKey.m_width,
        LocToGloKey.m_fieldInterleaveWidths);

    auto LocToGlo = LibUtilities::MemoryRegion<value_type>(1);
    auto ptr = LocToGlo.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
    ptr[0]   = gsinfo[0];

    return LocToGlo;
}

template <typename MemSpace, typename TPadding>
LibUtilities::MemoryRegion<
    typename DeviceLocalToGlobalNumAssembleKey<TPadding>::value_type>
LocalToGlobalDataCreator::Create(
    const DeviceLocalToGlobalNumAssembleKey<TPadding> &LocToGloKey)
{
    using value_type =
        typename DeviceLocalToGlobalNumAssembleKey<TPadding>::value_type;

    auto zeroDir = LocToGloKey.m_zeroDir;
    auto width   = LocToGloKey.m_width;

    const unsigned *gsinfo =
        GetGSInfo<TPadding>(zeroDir, LocToGloKey.m_components, width,
                            LocToGloKey.m_fieldInterleaveWidths)
            .data();

    auto nvals = gsinfo[0];
    unsigned nvalswidth =
        (nvals + (width - 1)) / width * width; // width aligned length

    // Decalare memory
    auto LocToGlo = LibUtilities::MemoryRegion<value_type>(nvalswidth);
    auto ptr = LocToGlo.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();

    for (unsigned i = 0; i < nvals; ++i)
    {
        ptr[i] = gsinfo[2 + i] - gsinfo[1 + i];
    }
    // pack out array to be multiple of widths
    for (unsigned i = nvals; i < nvalswidth; ++i)
    {
        ptr[i] = 0;
    }

    return LocToGlo;
}

template <typename MemSpace, typename TPadding>
LibUtilities::MemoryRegion<
    typename DeviceLocalToGlobalIndexKey<TPadding>::value_type>
LocalToGlobalDataCreator::Create(
    const DeviceLocalToGlobalIndexKey<TPadding> &LocToGloKey)
{
    using value_type =
        typename DeviceLocalToGlobalIndexKey<TPadding>::value_type;

    auto zeroDir = LocToGloKey.m_zeroDir;
    auto width   = LocToGloKey.m_width;

    const unsigned *gsinfo =
        GetGSInfo<TPadding>(zeroDir, LocToGloKey.m_components, width,
                            LocToGloKey.m_fieldInterleaveWidths)
            .data();

    auto nvals = gsinfo[0];
    auto nidx  = 0;

    auto numComp = GetAssemblyMap(LocToGloKey.m_components).size();
    auto blockAttr =
        GetBlockAttributes<TPadding, FieldState::Coeff>(this->m_expansionList);
    ASSERTL1(LocToGloKey.m_fieldInterleaveWidths.size() == blockAttr.size(),
             "One interleave width per block is required");

    // calculate number of indices as sum of i*width point times width
    for (unsigned i = 0; i < nvals; i = i + width)
    {
        nidx += (gsinfo[i + 2] - gsinfo[i + 1]) * width;
    }

    // Decalare memory
    auto LocToGlo = LibUtilities::MemoryRegion<value_type>(nidx);
    auto ptr = LocToGlo.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();

    unsigned offset = nvals + 2;
    unsigned cnt    = 0;
    // pack points in width consecutive order
    for (unsigned i = 0; i < nvals; i += width)
    {
        auto wres    = (i + width < nvals) ? width : nvals - i;
        auto nassmb0 = gsinfo[i + 2] - gsinfo[i + 1];

        unsigned j, k, index = 0;
        for (j = 0; j < wres; ++j)
        {
            // location of starting index for this point
            auto loc = gsinfo[i + j + 1];
            // number of points to be assembled
            auto nassmb = gsinfo[i + j + 2] - loc;
            // redistribute indices in width major format, addressing the
            // field at its interleave widths
            for (k = 0; k < nassmb; ++k)
            {
                index = InterleavedIndex(gsinfo[offset + loc + k], blockAttr,
                                         numComp,
                                         LocToGloKey.m_fieldInterleaveWidths);
                ptr[cnt + k * width + j] = index;
            }
            // fill out any unused value with the last index.
            for (; k < nassmb0; ++k)
            {
                ptr[cnt + k * width + j] = index;
            }
        }
        // add in additional padded values
        for (; j < width; ++j)
        {
            for (k = 0; k < nassmb0; ++k)
            {
                ptr[cnt + k * width + j] = index;
            }
        }
        cnt += width * nassmb0;
    }
    return LocToGlo;
}

template <typename MemSpace, typename TPadding>
LibUtilities::MemoryRegion<
    typename DeviceLocalToGlobalIndexOffsetKey<TPadding>::value_type>
LocalToGlobalDataCreator::Create(
    const DeviceLocalToGlobalIndexOffsetKey<TPadding> &LocToGloKey)
{
    using value_type =
        typename DeviceLocalToGlobalIndexOffsetKey<TPadding>::value_type;

    auto zeroDir = LocToGloKey.m_zeroDir;
    auto width   = LocToGloKey.m_width;

    const unsigned *gsinfo =
        GetGSInfo<TPadding>(zeroDir, LocToGloKey.m_components, width,
                            LocToGloKey.m_fieldInterleaveWidths)
            .data();

    auto nvals = gsinfo[0];
    unsigned nvalswidth =
        (nvals + (width - 1)) / width * width; // width aligned length

    // Decalare memory
    auto LocToGlo = LibUtilities::MemoryRegion<value_type>(nvalswidth);
    auto ptr = LocToGlo.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();

    // Determine the offset for index and sign data
    // possibly could be in LocalToGlobalDatawarehouse if used elsewhere
    unsigned cnt = 0;
    for (unsigned i = 0; i < nvals; i += width)
    {
        for (unsigned j = 0; j < width; ++j)
        {
            // put in local offset of jth point in width
            ptr[i + j] = cnt + j;
        }
        // skip forward to next block of indices
        cnt += (gsinfo[i + 2] - gsinfo[i + 1]) * width;
    }

    return LocToGlo;
}

template <typename TPadding>
void LocalToGlobalDataCreator::FillSignArray(
    std::vector<unsigned> &index,
    const std::vector<AssemblyMapCGSharedPtr> &loc2glo, bool zeroDir,
    bool signChange, int8_t *out)
{
    auto numComp = loc2glo.size();

    Array<OneD, unsigned> ndir(numComp);
    for (unsigned i = 0; i < numComp; ++i)
    {
        ndir[i] = loc2glo[i]->GetNumGlobalDirBndCoeffs();
    }

    auto sign =
        loc2glo[0]->GetSignChange()
            ? loc2glo[0]->GetLocalToGlobalSign()
            : Array<OneD, double>(this->m_expansionList->GetNcoeffs(), 1.0);

    auto blockAttr =
        GetBlockAttributes<TPadding, FieldState::Coeff>(this->m_expansionList);

    auto nblks = blockAttr.size();
    std::vector<unsigned> blkoffset(nblks + 1);
    std::vector<unsigned> coeffoffset(nblks);
    blkoffset[0] = 0;

    unsigned offset = 0;
    for (unsigned blk = 0; blk < nblks; ++blk)
    {
        blkoffset[blk + 1] =
            blkoffset[blk] + blockAttr[blk].CompSize() * numComp;
        coeffoffset[blk] = offset;
        offset += blockAttr[blk].GetNumElements() * blockAttr[blk].GetNumData();
    }

    // Fill the pointer with sign of local index.
    unsigned cnt = 0;
    if (zeroDir)
    {
        for (auto idx : index)
        {
            unsigned nc = 0;
            for (int blk = 0; blk < nblks; ++blk)
            {
                if ((idx >= blkoffset[blk]) && (idx < blkoffset[blk + 1]))
                {
                    // offset index to this block
                    idx -= blkoffset[blk];
                    // offset index for number of components;
                    nc  = idx / blockAttr[blk].CompSize();
                    idx = idx % blockAttr[blk].CompSize();
                    // add back in coeff offset to be able to access legacy
                    // index
                    idx += coeffoffset[blk];
                    break;
                }
            }

            if (signChange)
            {
                out[cnt++] = (loc2glo[nc]->GetLocalToGlobalMap(idx) < ndir[nc])
                                 ? 0
                                 : sign[idx];
            }
            else // not sure this case will be used.
            {
                out[cnt++] = (loc2glo[nc]->GetLocalToGlobalMap(idx) < ndir[nc])
                                 ? 0
                                 : std::abs(sign[idx]);
            }
        }
    }
    else
    {
        for (auto idx : index)
        {
            // Evaluate legacy index of point
            for (int blk = 0; blk < nblks; ++blk)
            {
                if ((idx >= blkoffset[blk]) && (idx < blkoffset[blk + 1]))
                {
                    // offset index to this block
                    idx -= blkoffset[blk];
                    // offset index for number of components;
                    idx = idx % blockAttr[blk].CompSize();
                    // add back in coeff offset to be able to access legacy
                    // index
                    idx += coeffoffset[blk];
                    break;
                }
            }

            if (signChange)
            {

                out[cnt++] = sign[idx];
            }
            else
            {
                out[cnt++] = std::abs(sign[idx]);
            }
        }
    }
}

template <typename MemSpace, typename TPadding>
LibUtilities::MemoryRegion<
    typename DeviceLocalToGlobalSignKey<TPadding>::value_type>
LocalToGlobalDataCreator::Create(
    const DeviceLocalToGlobalSignKey<TPadding> &LocToGloKey)
{
    using value_type =
        typename DeviceLocalToGlobalSignKey<TPadding>::value_type;

    auto zeroDir    = LocToGloKey.m_zeroDir;
    auto signChange = LocToGloKey.m_signChange;
    auto width      = LocToGloKey.m_width;
    auto &loc2glo   = GetAssemblyMap(LocToGloKey.m_components);

    const unsigned *gsinfo =
        GetGSInfo<TPadding>(zeroDir, LocToGloKey.m_components, width,
                            LocToGloKey.m_fieldInterleaveWidths)
            .data();

    auto nvals = gsinfo[0];
    auto nidx  = gsinfo[nvals + 1];

    // get a vector of the index values
    std::vector<unsigned> index;
    for (unsigned i = 0; i < nidx; ++i)
    {
        index.push_back(gsinfo[nvals + 2 + i]);
    }

    // evaluate the sign of each point
    std::vector<value_type> sign(nidx);
    FillSignArray<TPadding>(index, loc2glo, zeroDir, signChange, sign.data());

    // calculate number of sign values as sum of i*width point times width
    nidx = 0;
    for (unsigned i = 0; i < nvals; i += width)
    {
        nidx += (gsinfo[i + 2] - gsinfo[i + 1]) * width;
    }

    // Decalare memory for all local to global information.
    auto LocToGloSign = LibUtilities::MemoryRegion<value_type>(nidx);
    auto ptr =
        LocToGloSign.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();

    unsigned cnt = 0;
    // pack points in width consecutive order
    for (unsigned i = 0; i < nvals; i = i + width)
    {
        auto wres    = (i + width < nvals) ? width : nvals - i;
        auto nassmb0 = (gsinfo[i + 2] - gsinfo[i + 1]);

        unsigned j, k;
        for (j = 0; j < wres; ++j)
        {
            // location of starting index for this point
            auto loc = gsinfo[i + j + 1];
            // number of sign values to be assembled
            auto nassmb = gsinfo[i + j + 2] - loc;

            for (k = 0; k < nassmb; ++k)
            {
                ptr[cnt + k * width + j] = sign[loc + k];
            }
            // zero out any unused values
            for (; k < nassmb0; ++k)
            {
                ptr[cnt + k * width + j] = 0;
            }
        }
        // add in additional padded values
        for (; j < width; ++j)
        {
            for (k = 0; k < nassmb0; ++k)
            {
                ptr[cnt + k * width + j] = 0;
            }
        }
        cnt += width * nassmb0;
    }

    return LocToGloSign;
}

/**
 * @brief Parallel-boundary gather-scatter table, built once per set of
 * components.
 */
template <typename TPadding>
const std::vector<unsigned> &LocalToGlobalDataCreator::GetGSBndInfo(
    const std::vector<std::string> &components)
{
    size_t hash = 0;
    hash_combine(hash, std::type_index(typeid(TPadding)));
    for (const auto &component : components)
    {
        hash_combine(hash, component);
    }

    auto it = m_gsBndInfo.find(hash);
    if (it != m_gsBndInfo.end())
    {
        return it->second;
    }

    // Get Local To Global Map.
    auto &loc2gloAll = GetAssemblyMap(components);
    auto loc2glo     = loc2gloAll[0];
    auto numComp     = loc2gloAll.size();

    unsigned myrank = loc2glo->GetComm()->GetRank();

    // local to global mapping from legacy code
    auto loc2glomap = loc2glo->GetLocalToGlobalMap();

    // set for fast lookup of parallel gids
    std::unordered_set<size_t> parallel_gid(loc2glo->GetSREntries().begin(),
                                            loc2glo->GetSREntries().end());

    auto blockAttr =
        GetBlockAttributes<TPadding, FieldState::Coeff>(this->m_expansionList);

    auto nblks = blockAttr.size();
    std::map<unsigned, std::vector<std::pair<unsigned, unsigned>>> GloToLoc;

    unsigned coeff_offset = 0;
    for (unsigned blk = 0; blk < nblks; ++blk)
    {
        auto blksize =
            blockAttr[blk].GetNumData() * blockAttr[blk].GetNumElements();

        // Gather local id in block that share the same global id.
        for (unsigned lid = 0; lid < blksize; ++lid)
        {
            auto gid = loc2glomap[lid + coeff_offset];
            if (parallel_gid.count(gid)) // Is a parallel gid
            {
                GloToLoc[gid].push_back(
                    std::pair<unsigned, unsigned>(blk, lid));
            }
        }
        coeff_offset += blksize;
    }

    // At this point we have a list of global ids on parallel boundary
    // and the related blocks and ids within the blocks that are
    // connected to this global id.

    // Get vector of send receive entries of gids in order that are required
    // for communication in AssemblyComm
    auto SREntries = loc2glo->GetSREntries();
    // Get from rank index of each sent/receive entry to ensure rank
    // ordering of assemble points for inter node consistency
    auto fromRank = loc2glo->GetFromRank();

    // set up block offset to allow for local id determination
    std::vector<unsigned> BlkOffset(nblks);
    std::vector<unsigned> BlkSize(nblks);
    for (unsigned blk = 0; blk < nblks; ++blk)
    {
        BlkSize[blk] = blockAttr[blk].CompSize();
        if (blk)
        {
            BlkOffset[blk] = BlkOffset[blk - 1] + BlkSize[blk - 1] * numComp;
        }
        else
        {
            BlkOffset[blk] = 0;
        }
    }

    // Set up sorted list that can be used to fill in return data.
    std::map<unsigned, std::pair<std::vector<unsigned>, std::vector<unsigned>>>
        SortVals;
    std::map<unsigned, unsigned> GidExists;
    std::map<unsigned, std::vector<unsigned>> rankOrder;
    unsigned nvalstot = 0;
    unsigned nidx     = 0;
    // esure each compoent is done in sequence with nc in outer loop
    for (unsigned nc = 0; nc < numComp; ++nc)
    {
        for (unsigned i = 0; i < SREntries.size(); ++i)
        {
            unsigned gid = SREntries[i];

            if (GidExists.count(gid * numComp + nc))
            {
                SortVals[GidExists[gid * numComp + nc]].second.push_back(
                    i * numComp + nc);
                rankOrder[GidExists[gid * numComp + nc]].push_back(fromRank[i]);
                nidx++;
            }
            else // setup new sort value with local ids
            {
                std::pair<std::vector<unsigned>, std::vector<unsigned>> data;
                // offset for number of points and assemble order point
                nidx += 3;

                // gather the local ids
                for (auto [blk, lids] : GloToLoc[gid])
                {
                    data.first.push_back(lids + BlkSize[blk] * nc +
                                         BlkOffset[blk]);
                    nidx++;
                }
                data.second.push_back(i * numComp + nc);
                nidx++;

                SortVals[nvalstot] = data;

                GidExists[gid * numComp + nc] = nvalstot;
                rankOrder[nvalstot].push_back(fromRank[i]);
                nvalstot++;
            }
        }
    }

    // now have a list of sorted values containing a vector of local points
    // and a vector of global boundary entries. Next sort global boundary
    // entries into rank order so that boundary assembly is acheived in rank
    // order for consistency. Also identify order of this rank
    std::map<unsigned, unsigned> LocRankOrder;
    for (auto &vals : SortVals)
    {
        auto ind  = vals.first;
        auto npts = vals.second.second.size();
        ASSERTL1(npts == rankOrder[ind].size(),
                 "mismatch between bnd index and rank order");

        // determine ordering of ranks
        std::vector<unsigned> order(npts);
        for (int i = 0; i < npts; ++i)
        {
            unsigned rk  = rankOrder[ind][i];
            unsigned ord = 0;
            for (int j = 0; j < npts; ++j)
            {
                if (rk > rankOrder[ind][j])
                {
                    ord++;
                }
            }
            order[i] = ord;
        }
        std::vector<unsigned> bvals = vals.second.second;
        // reorder bvals
        for (int i = 0; i < npts; ++i)
        {
            vals.second.second[order[i]] = bvals[i];
        }

        // finally determien order of this rank
        unsigned ord = 0;
        for (int j = 0; j < npts; ++j)
        {
            if (myrank > rankOrder[ind][j])
            {
                ord++;
            }
        }
        LocRankOrder[vals.first] = ord;
    }

    // Decalare memory for all local to global informaiton.
    std::vector<unsigned> gsBndInfo(nvalstot + 2 + nidx);
    auto ptr = gsBndInfo.data();

    // Fill the pointer
    //  ptr[0] = nvals = number global points i
    //  ptr[1] = starting offset of local index to be assembled
    //  ptr[2] = starting offset of local index to be assembled
    //  ptr[3] = ....
    //  ptr[nvals+1] = total number of indices that follow
    //
    //  ptr[nvals+1 +0] = number of local indices to be assembled (nloc)
    //  ptr[nvals+1 +1] = number of bnd index to write (nbnd)
    //  ptr[nvals+1 +2] = local index of point to be assembled
    //  ptr[nvals+1 +x] = ..
    //  ptr[nvals+1 +nloc  ] = index of boundary point to write into
    //  ptr[nvals+1 +nloc+1] = index of boundary point to write into
    //  ptr[nvals+1 +nloc+x] = ..
    //  ptr[nvals+1 +nloc+nbnd] = rank order in which to assmeble
    //
    //  repeat above pattern for nvals entries
    auto ptr1     = ptr + nvalstot + 2;
    ptr[0]        = nvalstot;
    ptr[1]        = 0;
    unsigned cnt  = 2;
    unsigned cnt1 = 0;

    for (auto &vals : SortVals)
    {
        ptr[cnt] = ptr[cnt - 1]; // start new offset

        ptr1[cnt1++] = vals.second.first.size();
        ptr1[cnt1++] = vals.second.second.size();
        for (auto &iter : vals.second.first)
        {
            ptr1[cnt1++] = iter;
        }
        for (auto &iter : vals.second.second)
        {
            ptr1[cnt1++] = iter;
        }

        // rank offset of lcoal point
        ptr1[cnt1++] = LocRankOrder[vals.first];

        ptr[cnt] += 3 + vals.second.first.size() + vals.second.second.size();
        cnt++;
    }

    return m_gsBndInfo.emplace(hash, std::move(gsBndInfo)).first->second;
}

template <typename MemSpace, typename TPadding>
LibUtilities::MemoryRegion<
    typename DeviceBndLocalToGlobalNumGidsKey<TPadding>::value_type>
LocalToGlobalDataCreator::Create(
    const DeviceBndLocalToGlobalNumGidsKey<TPadding> &LocToGloKey)
{
    using value_type =
        typename DeviceBndLocalToGlobalNumGidsKey<TPadding>::value_type;

    const auto &gsinfo = GetGSBndInfo<TPadding>(LocToGloKey.m_components);

    auto LocToGlo = LibUtilities::MemoryRegion<value_type>(1);
    auto ptr = LocToGlo.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();
    ptr[0]   = gsinfo[0];

    return LocToGlo;
}

template <typename MemSpace, typename TPadding>
LibUtilities::MemoryRegion<
    typename DeviceBndLocalToGlobalNumAssembleKey<TPadding>::value_type>
LocalToGlobalDataCreator::Create(
    const DeviceBndLocalToGlobalNumAssembleKey<TPadding> &LocToGloKey)
{
    using value_type =
        typename DeviceBndLocalToGlobalNumAssembleKey<TPadding>::value_type;

    const unsigned *gsinfo =
        GetGSBndInfo<TPadding>(LocToGloKey.m_components).data();

    auto nvals = gsinfo[0];

    // Decalare memory
    auto LocToGlo = LibUtilities::MemoryRegion<value_type>(nvals);
    auto ptr = LocToGlo.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();

    const unsigned *ind = gsinfo + nvals + 2;
    for (unsigned i = 0; i < nvals; ++i)
    {
        auto offset = gsinfo[1 + i];
        ptr[i]      = ind[offset];
    }

    return LocToGlo;
}

template <typename MemSpace, typename TPadding>
LibUtilities::MemoryRegion<
    typename DeviceBndLocalToGlobalNumBndValsKey<TPadding>::value_type>
LocalToGlobalDataCreator::Create(
    const DeviceBndLocalToGlobalNumBndValsKey<TPadding> &LocToGloKey)
{
    using value_type =
        typename DeviceBndLocalToGlobalNumBndValsKey<TPadding>::value_type;

    const unsigned *gsinfo =
        GetGSBndInfo<TPadding>(LocToGloKey.m_components).data();

    auto nvals = gsinfo[0];

    // Decalare memory
    auto LocToGlo = LibUtilities::MemoryRegion<value_type>(nvals);
    auto ptr = LocToGlo.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();

    const unsigned *ind = gsinfo + nvals + 2;
    for (unsigned i = 0; i < nvals; ++i)
    {
        auto offset = gsinfo[1 + i];
        ptr[i]      = ind[offset + 1];
    }

    return LocToGlo;
}

template <typename MemSpace, typename TPadding>
LibUtilities::MemoryRegion<
    typename DeviceBndLocalToGlobalIndexKey<TPadding>::value_type>
LocalToGlobalDataCreator::Create(
    const DeviceBndLocalToGlobalIndexKey<TPadding> &LocToGloKey)
{
    using value_type =
        typename DeviceBndLocalToGlobalIndexKey<TPadding>::value_type;

    const unsigned *gsinfo =
        GetGSBndInfo<TPadding>(LocToGloKey.m_components).data();

    auto nvals           = gsinfo[0];
    const unsigned *ind  = gsinfo + nvals + 2;
    unsigned int nvaltot = 0;
    for (unsigned i = 0; i < nvals; ++i)
    {
        auto offset = gsinfo[1 + i];
        // count nassemble and nbndvals
        nvaltot += ind[offset] + ind[offset + 1];
    }

    // Decalare memory
    auto LocToGlo = LibUtilities::MemoryRegion<value_type>(nvaltot);
    auto ptr = LocToGlo.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();

    auto numComp = GetAssemblyMap(LocToGloKey.m_components).size();
    auto blockAttr =
        GetBlockAttributes<TPadding, FieldState::Coeff>(this->m_expansionList);
    ASSERTL1(LocToGloKey.m_fieldInterleaveWidths.size() == blockAttr.size(),
             "One interleave width per block is required");

    unsigned cnt = 0;
    for (unsigned i = 0; i < nvals; ++i)
    {
        unsigned offset = gsinfo[1 + i];
        unsigned nloc   = ind[offset];
        unsigned nbnd   = ind[offset + 1];

        // local indices address the field at its interleave widths
        for (unsigned j = 0; j < nloc; ++j)
        {
            ptr[cnt++] =
                InterleavedIndex(ind[offset + 2 + j], blockAttr, numComp,
                                 LocToGloKey.m_fieldInterleaveWidths);
        }

        // buffer slots
        for (unsigned j = nloc; j < nloc + nbnd; ++j)
        {
            ptr[cnt++] = ind[offset + 2 + j];
        }
    }

    return LocToGlo;
}

template <typename MemSpace, typename TPadding>
LibUtilities::MemoryRegion<
    typename DeviceBndLocalToGlobalOffsetKey<TPadding>::value_type>
LocalToGlobalDataCreator::Create(
    const DeviceBndLocalToGlobalOffsetKey<TPadding> &LocToGloKey)
{
    using value_type =
        typename DeviceBndLocalToGlobalOffsetKey<TPadding>::value_type;

    const unsigned *gsinfo =
        GetGSBndInfo<TPadding>(LocToGloKey.m_components).data();

    auto nvals          = gsinfo[0];
    const unsigned *ind = gsinfo + nvals + 2;

    // Decalare memory
    auto LocToGlo = LibUtilities::MemoryRegion<value_type>(nvals);
    auto ptr = LocToGlo.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();

    ptr[0] = 0;
    for (unsigned i = 0; i < nvals - 1; ++i)
    {
        unsigned offset = gsinfo[1 + i];
        // nassemble and nbndvals
        unsigned nind = ind[offset] + ind[offset + 1];

        ptr[i + 1] = ptr[i] + nind;
    }

    return LocToGlo;
}

template <typename MemSpace, typename TPadding>
LibUtilities::MemoryRegion<
    typename DeviceBndLocalToGlobalAssembleOrderKey<TPadding>::value_type>
LocalToGlobalDataCreator::Create(
    const DeviceBndLocalToGlobalAssembleOrderKey<TPadding> &LocToGloKey)
{
    using value_type =
        typename DeviceBndLocalToGlobalAssembleOrderKey<TPadding>::value_type;

    const unsigned *gsinfo =
        GetGSBndInfo<TPadding>(LocToGloKey.m_components).data();

    auto nvals          = gsinfo[0];
    const unsigned *ind = gsinfo + nvals + 2;

    // Decalare memory
    auto LocToGlo = LibUtilities::MemoryRegion<value_type>(nvals);
    auto ptr = LocToGlo.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();

    for (unsigned i = 0; i < nvals; ++i)
    {
        unsigned offset = gsinfo[1 + i];
        // nassemble and nbndvals
        unsigned nind = ind[offset] + ind[offset + 1];

        ptr[i] = ind[offset + nind + 2];
    }

    return LocToGlo;
}

template <typename MemSpace, typename TPadding>
LibUtilities::MemoryRegion<
    typename DeviceBndLocalToGlobalSignKey<TPadding>::value_type>
LocalToGlobalDataCreator::Create(
    const DeviceBndLocalToGlobalSignKey<TPadding> &LocToGloKey)
{
    using value_type =
        typename DeviceBndLocalToGlobalSignKey<TPadding>::value_type;

    auto zeroDir    = LocToGloKey.m_zeroDir;
    auto signChange = LocToGloKey.m_signChange;

    auto &loc2glo = GetAssemblyMap(LocToGloKey.m_components);
    const unsigned *gsinfo =
        GetGSBndInfo<TPadding>(LocToGloKey.m_components).data();

    // extract lids from gsinfo
    std::vector<unsigned> lids;
    unsigned nvals = gsinfo[0];
    auto idx_ptr   = gsinfo + nvals + 2;
    unsigned ntot  = 0;
    for (unsigned i = 1; i <= nvals; ++i)
    {
        auto ptr  = idx_ptr + gsinfo[i];
        auto nidx = ptr[0];
        for (unsigned j = 0; j < nidx; ++j)
        {
            lids.push_back(ptr[j + 2]);
        }
        ntot += nidx + ptr[1];
    }

    std::vector<value_type> sign(lids.size());

    FillSignArray<TPadding>(lids, loc2glo, zeroDir, signChange, sign.data());

    auto LocToGloSign = LibUtilities::MemoryRegion<value_type>(ntot);
    auto signptr =
        LocToGloSign.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();

    unsigned cnt  = 0;
    unsigned cnt1 = 0;
    // write out sign array with boundary point packing.
    for (unsigned i = 1; i <= nvals; ++i)
    {
        auto ptr  = idx_ptr + gsinfo[i];
        auto nidx = ptr[0];
        for (unsigned j = 0; j < nidx; ++j)
        {
            signptr[cnt++] = sign[cnt1++];
        }
        auto nbnd = ptr[1];
        for (unsigned j = 0; j < nbnd; ++j)
        {
            signptr[cnt++] = 0;
        }
    }
    return LocToGloSign;
}

template <typename MemSpace, typename TPadding>
LibUtilities::MemoryRegion<typename LocalToGlobalMaskKey<TPadding>::value_type>
LocalToGlobalDataCreator::Create(
    const LocalToGlobalMaskKey<TPadding> &LocToGloKey)
{
    using value_type = typename LocalToGlobalMaskKey<TPadding>::value_type;

    // Get Local To Global Map.
    auto &loc2gloAll = GetAssemblyMap(LocToGloKey.m_components);
    auto loc2glo     = loc2gloAll[0];
    auto l2gmap0     = loc2glo->GetLocalToGlobalMap();
    auto blockAttr =
        GetBlockAttributes<TPadding, FieldState::Coeff>(this->m_expansionList);
    unsigned ntot = 0;
    unsigned blk  = 0;
    for (blk = 0; blk < blockAttr.size(); ++blk)
    {
        ntot += blockAttr[blk].CompSize();
    }

    // Decalare memory for all local to global informaiton.
    auto LocToGlo = LibUtilities::MemoryRegion<value_type>(ntot);
    auto ptr = LocToGlo.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();

    std::set<unsigned> done;
    unsigned offset = 0;
    unsigned cnt    = 0;
    for (blk = 0; blk < blockAttr.size(); ++blk)
    {
        auto &block           = blockAttr[blk];
        auto num_elements     = block.GetNumElements();
        auto num_elements_pad = block.GetNumElementsWithPadding();
        auto num_data         = block.GetNumData();

        // Loop over chunks.
        for (unsigned el = 0; el < num_elements; ++el)
        {
            for (unsigned j = 0; j < num_data; ++j)
            {
                auto lid    = offset + el * num_data + j;
                auto gid    = l2gmap0[lid];
                auto unique = loc2glo->GetGlobalToUniversalMapUnique(gid);

                if (unique && done.count(gid) == 0)
                {
                    ptr[cnt++] = 1;
                    done.insert(gid);
                }
                else
                {
                    ptr[cnt++] = 0;
                }
            }
        }

        for (size_t el = num_elements; el < num_elements_pad; ++el)
        {
            for (unsigned j = 0; j < num_data; ++j)
            {
                ptr[cnt++] = 0;
            }
        }

        offset += num_data * num_elements;
    }
    return LocToGlo;
}

} // namespace Nektar::MultiRegions
