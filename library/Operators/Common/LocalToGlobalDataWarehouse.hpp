///////////////////////////////////////////////////////////////////////////////
//
// File: LocalToGlobalDataWarehouse.hpp
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

#include <MultiRegions/ContField.h>

#include "Operators/Common/NekDataWarehouse.hpp"
#include "Operators/Field/Field.hpp"

#if defined(_MSC_VER)
#undef max
#undef min
#endif

namespace Nektar::Operators
{

class LocalToGlobalDataCreator;

template <typename TData> class LocalToGlobalKey : public BaseKey
{
    friend class LocalToGlobalDataCreator;

public:
    using creator = LocalToGlobalDataCreator;
    typedef unsigned value_type;

    ~LocalToGlobalKey() override = default;

    LocalToGlobalKey(unsigned ndir, unsigned numComp, bool zeroDir)
        : m_ndir(ndir), m_numComp(numComp), m_zeroDir(zeroDir)
    {
        hash_combine(m_hash, m_ndir, m_numComp, m_zeroDir,
                     typeid(value_type).name(), "LocalToGlobalKey");
    }

private:
    unsigned m_ndir;
    unsigned m_numComp;
    bool m_zeroDir;
};

class LocalToGlobalDataCreator : public DataCreatorClass
{
public:
    ~LocalToGlobalDataCreator() override = default;
    LocalToGlobalDataCreator(
        const MultiRegions::ExpListSharedPtr &expansionList)
        : m_expansionList(expansionList)
    {
    }

    // can use any template declaration as only trying to get hold of value_type
    // which is fixed
    using value_type = LocalToGlobalKey<unsigned>::value_type;

    template <typename MemSpace, typename TData>
    MemoryRegion<value_type> Create(const LocalToGlobalKey<TData> &LocToGloKey,
                                    const size_t alignment)
    {
        // Get Local To Global Map.
        auto ndir    = LocToGloKey.m_ndir;
        auto zeroDir = LocToGloKey.m_zeroDir;
        auto numComp = LocToGloKey.m_numComp;

        auto contfield =
            std::dynamic_pointer_cast<MultiRegions::ContField>(m_expansionList);
        auto loc2glomap =
            contfield->GetLocalToGlobalMap()->GetLocalToGlobalMap();

        // here we are using double as the type since this is what the datatype
        // of the assembled data is assumed to be. Not sure what we shoudl do it
        // is float however but legacy code is not set up for this either
        auto blocks =
            GetBlockAttributes<TData>(FieldState::Coeff, m_expansionList);

        auto nblks = blocks.size();
        std::map<unsigned, std::vector<std::pair<unsigned, unsigned>>> GloToLoc;

        unsigned coeff_offset = 0;
        for (unsigned blk = 0; blk < nblks; ++blk)
        {
            auto blksize =
                blocks[blk].GetNumData() * blocks[blk].GetNumElements();

            // Gather local id in block that share the same global id.
            for (unsigned lid = 0; lid < blksize; ++lid)
            {
                auto gid = loc2glomap[lid + coeff_offset];
                GloToLoc[gid].push_back(
                    std::pair<unsigned, unsigned>(blk, lid));
            }
            coeff_offset += blksize;
        }

        // At this point we have a list of global ids and the related
        // blocks and ids within the blocks that are connected to this
        // global id. Next assemble a list of gids and components that should be
        // ordered in the final output.
        std::map<unsigned, std::pair<unsigned, unsigned>> SortVals;
        std::vector<unsigned> BlkOffset(nblks);
        std::set<unsigned> gidDone;

        unsigned nvalstot = 0;
        unsigned nidx     = 0;
        coeff_offset      = 0;

        for (unsigned blk = 0; blk < nblks; ++blk)
        {
            unsigned blksize =
                blocks[blk].GetNumElements() * blocks[blk].GetNumData();
            for (int nc = 0; nc < numComp; ++nc)
            {
                for (int lid = 0; lid < blksize; ++lid)
                {
                    unsigned gid = loc2glomap[lid + coeff_offset];

                    if (gidDone.count(gid * numComp + nc) == 0)
                    {
                        // Only add  points with valence more than 1 or if
                        // global dof < ndir then add to list so it can be
                        // zeroed if so desired
                        unsigned val = GloToLoc[gid].size();
                        if (zeroDir)
                        {
                            if ((val > 1) || (gid < ndir))
                            {
                                nidx += val;
                                SortVals[nvalstot++] =
                                    std::pair<unsigned, unsigned>(gid, nc);
                            }
                        }
                        else
                        {
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
                    BlkOffset[blk - 1] + blocks[blk - 1].size() * numComp;
            }
            else
            {
                BlkOffset[blk] = 0;
            }
            coeff_offset += blksize;
        }

        // Decalare memory for all local to global informaiton.
        auto LocToGlo =
            MemoryRegion<value_type>(nvalstot + 2 + nidx, alignment);
        auto ptr =
            LocToGlo.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();

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
                lid += BlkOffset[blk] + blocks[blk].size() * nc;

                ptr1[cnt1++] = lid; // set index
                ptr[cnt] += 1;      // add one to offset
            }
            cnt++;
        }

        return LocToGlo;
    }

    inline static const std::string m_name = "LocalToGlobalDataCreator";

private:
    MultiRegions::ExpListSharedPtr m_expansionList;
};

class LocalToGlobalSignDataCreator;

template <typename TData> class LocalToGlobalSignKey : public BaseKey
{
    friend class LocalToGlobalSignDataCreator;

public:
    using creator = LocalToGlobalSignDataCreator;
    // typedef int8_t value_type;
    typedef int value_type;

    ~LocalToGlobalSignKey() override = default;

    LocalToGlobalSignKey(unsigned nDir, unsigned numComp, bool zeroDir,
                         bool signChange)
        : m_nDir(nDir), m_numComp(numComp), m_zeroDir(zeroDir),
          m_signChange(signChange)
    {
        hash_combine(m_hash, m_nDir, m_numComp, m_zeroDir, m_signChange,
                     typeid(value_type).name(), "LocalToGlobalSignKey");
    }

private:
    unsigned m_nDir;
    unsigned m_numComp;
    bool m_zeroDir;
    bool m_signChange;
};

class LocalToGlobalSignDataCreator : public DataCreatorClass
{
public:
    ~LocalToGlobalSignDataCreator() override = default;
    LocalToGlobalSignDataCreator(
        const MultiRegions::ExpListSharedPtr &expansionList)
        : m_expansionList(expansionList)
    {
    }

    // can use any template declaration as only trying to get hold of value_type
    // which is fixed
    using value_type = LocalToGlobalSignKey<unsigned>::value_type;

    template <typename MemSpace, typename TData>
    MemoryRegion<value_type> Create(
        const LocalToGlobalSignKey<TData> &LocToGloKey, const size_t alignment)
    {
        // Get Local To Global Map.
        auto numDir     = LocToGloKey.m_nDir;
        auto numComp    = LocToGloKey.m_numComp;
        auto zeroDir    = LocToGloKey.m_zeroDir;
        auto signChange = LocToGloKey.m_signChange;

        auto contfield =
            std::dynamic_pointer_cast<MultiRegions::ContField>(m_expansionList);

        auto loc2glo = contfield->GetLocalToGlobalMap();
        auto l2gmap  = loc2glo->GetLocalToGlobalMap();
        auto sign    = loc2glo->GetSignChange()
                           ? loc2glo->GetLocalToGlobalSign()
                           : Array<OneD, double>(contfield->GetTotPoints(), 1.0);

        auto dataWarehouse = this->m_expansionList->GetDataWarehouseSharedPtr();

        const auto *gsinfo =
            dataWarehouse->template GetData<NektarSpaces::Serial>(
                LocalToGlobalKey<TData>(numDir, numComp, zeroDir));

        auto nvals = gsinfo[0];
        auto nidx  = gsinfo[nvals + 1];

        // Decalare memory for all local to global information.
        auto LocToGloSign = MemoryRegion<value_type>(nidx, alignment);
        auto ptr =
            LocToGloSign.template GetPtr<NektarSpaces::HostSpace, WriteOnly>();

        // generate a list of the block and coeff offsets to deduce the local
        // coefficient entry of each index to use the sign array from legacy
        // code.
        auto blocks =
            GetBlockAttributes<TData>(FieldState::Coeff, m_expansionList);

        auto nblks = blocks.size();
        std::vector<unsigned> blkoffset(nblks + 1);
        std::vector<unsigned> coeffoffset(nblks);
        blkoffset[0] = 0;

        unsigned offset = 0;
        for (unsigned blk = 0; blk < nblks; ++blk)
        {
            blkoffset[blk + 1] = blkoffset[blk] + blocks[blk].size() * numComp;
            coeffoffset[blk]   = offset;
            offset += blocks[blk].GetNumElements() * blocks[blk].GetNumData();
        }

        // Fill the pointer with sign of local index.
        unsigned cnt = 0;
        offset       = nvals + 2;

        if (zeroDir)
        {
            for (unsigned i = 0; i < nidx; ++i)
            {
                // Evaluate legacy index of point
                unsigned idx = gsinfo[offset + i];
                for (int blk = 0; blk < nblks; ++blk)
                {
                    if ((idx >= blkoffset[blk]) && (idx < blkoffset[blk + 1]))
                    {
                        // offset index to this block
                        idx -= blkoffset[blk];
                        // offset index for number of components;
                        idx = idx % blocks[blk].size();
                        // add back in coeff offset to be able to access legacy
                        // index
                        idx += coeffoffset[blk];
                        break;
                    }
                }

                if (signChange)
                {
                    ptr[cnt++] = (l2gmap[idx] < numDir) ? 0 : sign[idx];
                }
                else // not sure this case will be used.
                {
                    ptr[cnt++] =
                        (l2gmap[idx] < numDir) ? 0 : std::abs(sign[idx]);
                }
            }
        }
        else
        {
            for (unsigned i = 0; i < nidx; ++i)
            {
                // Evaluate legacy index of point
                unsigned idx = gsinfo[offset + i];
                for (int blk = 0; blk < nblks; ++blk)
                {
                    if ((idx >= blkoffset[blk]) && (idx < blkoffset[blk + 1]))
                    {
                        // offset index to this block
                        idx -= blkoffset[blk];
                        // offset index for number of components;
                        idx = idx % blocks[blk].size();
                        // add back in coeff offset to be able to access legacy
                        // index
                        idx += coeffoffset[blk];
                        break;
                    }
                }

                if (signChange)
                {
                    ptr[cnt++] = sign[idx];
                }
                else
                {
                    ptr[cnt++] = std::abs(sign[idx]);
                }
            }
        }

        return LocToGloSign;
    }

    inline static const std::string m_name = "LocalToGlobalSignDataCreator";

private:
    MultiRegions::ExpListSharedPtr m_expansionList;
};

} // namespace Nektar::Operators
