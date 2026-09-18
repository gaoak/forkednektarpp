////////////////////////////////////////////////////////////////////////////////
//
//  File: Mesh.h
//
//  For more information, please see: http://www.nektar.info/
//
//  The MIT License
//
//  Copyright (c) 2006 Division of Applied Mathematics, Brown University (USA),
//  Department of Aeronautics, Imperial College London (UK), and Scientific
//  Computing and Imaging Institute, University of Utah (USA).
//
//  Permission is hereby granted, free of charge, to any person obtaining a
//  copy of this software and associated documentation files (the "Software"),
//  to deal in the Software without restriction, including without limitation
//  the rights to use, copy, modify, merge, publish, distribute, sublicense,
//  and/or sell copies of the Software, and to permit persons to whom the
//  Software is furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included
//  in all copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
//  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//  DEALINGS IN THE SOFTWARE.
//
//  Description: Mesh object.
//
////////////////////////////////////////////////////////////////////////////////

#ifndef NEKMESH_MESHELEMENTS_MESH
#define NEKMESH_MESHELEMENTS_MESH

#include <algorithm>
#include <set>

#include <LibUtilities/BasicUtils/FieldIO.h>

#include <LibUtilities/BasicUtils/Log.hpp>
#include <NekMesh/MeshElements/Element.h>
#include <NekMesh/Module/ModuleContext.h>
#include <NekMesh/NekMeshDeclspec.h>
#include <SpatialDomains/MeshGraph.h>

namespace Nektar::NekMesh
{

/**
 * Enumeration of condition types (Dirichlet, Neumann, etc).
 */
enum ConditionType
{
    eDirichlet,
    eNeumann,
    eRobin,
    ePeriodic,
    eHOPCondition,
    SIZE_ConditionType
};

/**
 * @brief Defines a boundary condition.
 *
 * A boundary condition is defined by its type (e.g. Dirichlet), the
 * field it applies to, the value imposed on this field and the
 * composite which the boundary condition is applied to.
 */
struct Condition
{
    Condition() : type(), field(), value(), m_composite()
    {
    }
    std::vector<ConditionType> type;
    std::vector<std::string> field;
    std::vector<std::string> value;
    std::vector<int> m_composite;
};

// Custom hash function used so Geometry* and SegGeom* pointing to the same
// SegGeom act as the same key
struct GeometryPtrHash
{
    std::size_t operator()(const SpatialDomains::Geometry *geom) const
    {
        return std::hash<const void *>()(static_cast<const void *>(geom));
    }
};
/**
 * @brief Orders geometry pointers by global ID rather than by address.
 *
 * For containers whose iteration order decides a result and whose entities are
 * not renumbered while they are alive. A global ID is not fixed for the life of
 * a geometry, and mutating one that is a live key breaks the ordering invariant
 * and with it every lookup, so anything longer-lived than a single meshing pass
 * wants TaggedEntities' insertion order instead.
 */
struct GeometryPtrIdLess
{
    bool operator()(const SpatialDomains::Geometry *a,
                    const SpatialDomains::Geometry *b) const
    {
        return a->GetGlobalID() != b->GetGlobalID()
                   ? a->GetGlobalID() < b->GetGlobalID()
                   : a < b;
    }
};

/**
 * @brief The entities of one dimension that carry a composite tag.
 *
 * Two jobs that pull in opposite directions: modules walk these in order, and
 * they look up a single entity's tag. The order has to come out the same on
 * every platform, so it is the order the entities were tagged in, held in a
 * flat vector. The tag lookup rides alongside in a hash keyed on the pointer,
 * where the order never escapes.
 *
 * Sorted by neither key, deliberately. A global ID is renumbered while entries
 * are live (e.g. see ProcessBL), so a container ordered on one loses track of
 * its own keys; an address orders differently on every platform.
 */
class TaggedEntities
{
public:
    /// Members named first and second so that this stands in for the map it
    /// replaces without touching the code that reads it.
    struct Entry
    {
        SpatialDomains::Geometry *first;
        int second;
    };

    typedef SpatialDomains::Geometry *key_type;
    typedef int mapped_type;
    typedef Entry value_type;

    /**
     * @brief Steps over removed entries.
     *
     * Removing leaves a hole rather than closing it up, so that erasing in a
     * loop -- which most callers do -- does not shift everything after it each
     * time. The holes are collected once enough of them accumulate.
     */
    template <typename EntryType, typename VecIter> class Iterator
    {
    public:
        typedef std::forward_iterator_tag iterator_category;
        typedef EntryType value_type;
        typedef std::ptrdiff_t difference_type;
        typedef EntryType *pointer;
        typedef EntryType &reference;

        Iterator() = default;
        Iterator(VecIter pos, VecIter last) : m_pos(pos), m_end(last)
        {
            SkipHoles();
        }

        EntryType &operator*() const
        {
            return *m_pos;
        }
        EntryType *operator->() const
        {
            return &*m_pos;
        }

        Iterator &operator++()
        {
            ++m_pos;
            SkipHoles();
            return *this;
        }

        Iterator operator++(int)
        {
            Iterator tmp = *this;
            ++(*this);
            return tmp;
        }

        bool operator==(const Iterator &other) const
        {
            return m_pos == other.m_pos;
        }
        bool operator!=(const Iterator &other) const
        {
            return m_pos != other.m_pos;
        }

    private:
        void SkipHoles()
        {
            while (m_pos != m_end && m_pos->first == nullptr)
            {
                ++m_pos;
            }
        }

        VecIter m_pos{}, m_end{};
    };

    typedef Iterator<Entry, std::vector<Entry>::iterator> iterator;
    typedef Iterator<const Entry, std::vector<Entry>::const_iterator>
        const_iterator;

    iterator begin()
    {
        return iterator(m_entries.begin(), m_entries.end());
    }
    iterator end()
    {
        return iterator(m_entries.end(), m_entries.end());
    }
    const_iterator begin() const
    {
        return const_iterator(m_entries.begin(), m_entries.end());
    }
    const_iterator end() const
    {
        return const_iterator(m_entries.end(), m_entries.end());
    }

    /// Number of tagged entities, not counting the holes left by removals.
    size_t size() const
    {
        return m_index.size();
    }
    bool empty() const
    {
        return m_index.empty();
    }

    void clear()
    {
        m_entries.clear();
        m_index.clear();
    }

    /**
     * @brief The tag of @p geom, tagging it 0 if it is not tagged yet.
     *
     * The reference points into the flat store, so a later insertion or
     * removal can invalidate it where an unordered_map's would have stayed
     * good. Read or assign through it and let it go.
     */
    int &operator[](SpatialDomains::Geometry *geom)
    {
        auto [it, isNew] = m_index.try_emplace(geom, m_entries.size());
        if (isNew)
        {
            m_entries.push_back({geom, 0});
        }
        return m_entries[it->second].second;
    }

    /// Tag @p geom, for generic code that expects the map interface. Leaves
    /// an entity that is already tagged alone, as inserting into the map this
    /// replaces did.
    std::pair<iterator, bool> emplace(SpatialDomains::Geometry *geom, int tag)
    {
        auto [it, isNew] = m_index.try_emplace(geom, m_entries.size());
        if (isNew)
        {
            m_entries.push_back({geom, tag});
        }
        return {iterator(m_entries.begin() + it->second, m_entries.end()),
                isNew};
    }

    iterator find(SpatialDomains::Geometry *geom)
    {
        auto it = m_index.find(geom);
        return it == m_index.end()
                   ? end()
                   : iterator(m_entries.begin() + it->second, m_entries.end());
    }

    const_iterator find(SpatialDomains::Geometry *geom) const
    {
        auto it = m_index.find(geom);
        return it == m_index.end()
                   ? end()
                   : const_iterator(m_entries.begin() + it->second,
                                    m_entries.end());
    }

    size_t count(SpatialDomains::Geometry *geom) const
    {
        return m_index.count(geom);
    }

    /// Untag @p geom, returning how many entries that removed, as erasing
    /// from the map this replaces did.
    size_t erase(SpatialDomains::Geometry *geom)
    {
        auto it = m_index.find(geom);
        if (it == m_index.end())
        {
            return 0;
        }

        m_entries[it->second].first = nullptr;
        m_index.erase(it);
        Compact();
        return 1;
    }

    /// Untag whatever @p pos refers to.
    void erase(const_iterator pos)
    {
        erase(pos->first);
    }

    void erase(iterator pos)
    {
        erase(pos->first);
    }

    /**
     * @brief Drop every entity @p pred holds for.
     *
     * One pass, leaving the order of what remains alone. Prefer this to
     * erasing in a loop when the condition is known up front.
     */
    template <typename Pred> void RemoveIf(Pred pred)
    {
        for (auto &entry : m_entries)
        {
            if (entry.first != nullptr && pred(entry.first, entry.second))
            {
                m_index.erase(entry.first);
                entry.first = nullptr;
            }
        }
        Compact(true);
    }

private:
    /// Close up the holes once they outnumber the entries worth keeping, so
    /// that repeated removal stays linear overall rather than quadratic.
    void Compact(bool force = false)
    {
        if (!force && m_entries.size() < 2 * m_index.size() + 32)
        {
            return;
        }

        m_entries.erase(
            std::remove_if(m_entries.begin(), m_entries.end(),
                           [](const Entry &e) { return e.first == nullptr; }),
            m_entries.end());

        for (size_t i = 0; i < m_entries.size(); ++i)
        {
            m_index[m_entries[i].first] = i;
        }
    }

    /// Tagged entities in the order they were tagged, holes included.
    std::vector<Entry> m_entries;
    /// Position in m_entries of each tagged entity.
    std::unordered_map<SpatialDomains::Geometry *, size_t, GeometryPtrHash>
        m_index;
};

typedef TaggedEntities GeomTagMap;

typedef std::shared_ptr<Condition> ConditionSharedPtr;
typedef std::map<int, ConditionSharedPtr> ConditionMap;

NEKMESH_EXPORT bool operator==(ConditionSharedPtr const &c1,
                               ConditionSharedPtr const &c2);

class Mesh
{
public:
    NEKMESH_EXPORT Mesh()
    {
        m_meshGraph = std::make_shared<SpatialDomains::MeshGraph>();
    }
    /// Metadata map for storing any mesh generation parameters
    LibUtilities::FieldMetaDataMap m_metadata;
    /// MPI communicator in case we end up using MPI multiple times from
    /// Nektar++ SessionReader object.
    LibUtilities::CommSharedPtr m_comm;
    /// Mesh graph shared ptr storing the geometry info
    SpatialDomains::MeshGraphSharedPtr m_meshGraph;
    /// Array by dimension of maps of all elements and boundary surfaces to a
    /// vector of their tags.
    std::array<GeomTagMap, 4> m_elementTags;
    /// Map of element edges.
    EdgeMap m_edgeSet;
    /// Map of element faces.
    FaceMap m_faceSet;

    /// Returns the total number of elements in the mesh with
    /// dimension expDim.
    NEKMESH_EXPORT unsigned int GetNumElements();
    /// Returns the total number of elements in the mesh with
    /// dimension < expDim.
    NEKMESH_EXPORT unsigned int GetNumBndryElements();
    /// Returns the total number of entities in the mesh.
    NEKMESH_EXPORT unsigned int GetNumTaggedEntities();

    NEKMESH_EXPORT std::array<GeomTagMap, 4> &GetAllElementTags()
    {
        return m_elementTags;
    }

    NEKMESH_EXPORT GeomTagMap &GetAllElementTags(int i)
    {
        return m_elementTags[i];
    }

    NEKMESH_EXPORT bool IsBoundary(SpatialDomains::Geometry *geomPtr, int dim)
    {
        return m_elementTags[dim].find(geomPtr) != m_elementTags[dim].end();
    }

    NEKMESH_EXPORT void MakeOrder(int order, LibUtilities::PointsType distType,
                                  Logger &log);

    NEKMESH_EXPORT void PrintStats(Logger &out);

    /**
     * @brief Information handed from one module in the pipeline to another.
     *
     * For payloads that are not mesh data and are only of interest to a
     * handful of modules; see ModuleContext.
     */
    NEKMESH_EXPORT ModuleContext &GetContext()
    {
        return m_context;
    }

private:
    /// Store for inter-module payloads.
    ModuleContext m_context;
};

/// Shared pointer to a mesh.
typedef std::shared_ptr<Mesh> MeshSharedPtr;

} // namespace Nektar::NekMesh

#endif
