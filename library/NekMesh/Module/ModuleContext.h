////////////////////////////////////////////////////////////////////////////////
//
//  File: ModuleContext.h
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
//  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
//  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
//  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
//  CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
//  TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
//  SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//
//  Description: Type-keyed store for data passed between NekMesh modules.
//
////////////////////////////////////////////////////////////////////////////////

#ifndef NEKMESH_MODULE_MODULECONTEXT
#define NEKMESH_MODULE_MODULECONTEXT

#include <cstdint>
#include <memory>
#include <type_traits>
#include <typeindex>
#include <unordered_map>
#include <utility>

namespace Nektar::NekMesh
{

/**
 * @brief Whether a payload must be discarded when mesh entities are destroyed.
 *
 * Payloads keyed on entity pointers -- the usual case, and so the default --
 * are left holding dangling keys when geometry is freed, and must go with it.
 * Specialise this to std::false_type for a payload that references no mesh
 * entity at all: the octree, for instance, is built from the CAD and is
 * unaffected by anything that happens to the mesh.
 */
template <typename T> struct PayloadFollowsEntities : std::true_type
{
};

/**
 * @brief Store for information handed from one module to another.
 *
 * Modules in a NekMesh pipeline are constructed, configured and then run in
 * sequence, and the only object they share is the Mesh. Anything a module
 * wants to hand to a later module has therefore historically been added as a
 * Mesh member, even when only two modules in the whole library ever touch it:
 * Mesh::m_octree is written by @c loadoctree and read by the surface and
 * volume meshers, and nothing else refers to it.
 *
 * A ModuleContext holds those payloads instead, keyed on their own type so
 * that a payload stays a real declared struct rather than a string lookup.
 * Producers call Set() or Get(); consumers call TryGet() and get a null
 * pointer if no module in this pipeline produced one.
 *
 * Payloads are *not* mesh data. Anything the output modules write, or that
 * more than a handful of modules read, belongs on Mesh or MeshGraph. In
 * particular a payload that must react to individual entities being destroyed
 * needs lifetime hooks where the entities are owned -- that is what
 * SpatialDomains::CADAssociation does -- whereas a payload for which "the
 * mesh changed, throw it away" is a safe response belongs here.
 *
 * That coarse response is what Invalidate() implements. It is called when
 * mesh entities are destroyed, so a payload keyed on entity pointers is
 * dropped rather than left dangling, and its consumer falls back to whatever
 * it would have done had no producer run. Note that renumbering is
 * deliberately *not* an invalidation: keying on pointers rather than IDs is
 * precisely what makes a payload survive it.
 */
class ModuleContext
{
public:
    ModuleContext()  = default;
    ~ModuleContext() = default;

    /**
     * @brief Construct payload @tparam T, replacing any existing one.
     *
     * For a producer that owns the payload outright and should overwrite
     * whatever an earlier module left behind.
     */
    template <typename T, typename... Args> T &Set(Args &&...args)
    {
        auto ptr = std::make_shared<T>(std::forward<Args>(args)...);
        m_data[std::type_index(typeid(T))] = {ptr, m_generation};
        return *ptr;
    }

    /**
     * @brief Fetch payload @tparam T, constructing it if absent or stale.
     *
     * For a producer that accumulates into the payload, and for a consumer
     * that treats an absent payload and an empty one alike.
     */
    template <typename T, typename... Args> T &Get(Args &&...args)
    {
        T *existing = TryGet<T>();
        return existing != nullptr ? *existing
                                   : Set<T>(std::forward<Args>(args)...);
    }

    /**
     * @brief Fetch payload @tparam T, or nullptr if no module produced one.
     *
     * For a consumer that must distinguish "nothing was supplied" from
     * "something was supplied and is empty" -- the two cases usually call for
     * different behaviour, and a size() == 0 test cannot tell them apart.
     */
    template <typename T> T *TryGet()
    {
        auto it = m_data.find(std::type_index(typeid(T)));

        if (it == m_data.end())
        {
            return nullptr;
        }

        // Produced before the mesh last lost entities, so its keys may no
        // longer be valid; drop it and report it as absent. A payload that
        // references no entity is exempt -- see PayloadFollowsEntities.
        if constexpr (PayloadFollowsEntities<T>::value)
        {
            if (it->second.generation != m_generation)
            {
                m_data.erase(it);
                return nullptr;
            }
        }

        return static_cast<T *>(it->second.data.get());
    }

    template <typename T> bool Has()
    {
        return TryGet<T>() != nullptr;
    }

    /// Discard payload @tparam T, if present.
    template <typename T> void Remove()
    {
        m_data.erase(std::type_index(typeid(T)));
    }

    /**
     * @brief Mark every payload produced so far as stale.
     *
     * Called when mesh entities are destroyed, e.g. by
     * Module::RemoveOrphanedEntities(). Payloads are dropped lazily on the
     * next TryGet() so that this stays cheap to call.
     */
    void Invalidate()
    {
        ++m_generation;
    }

private:
    struct Entry
    {
        /// Type-erased payload. shared_ptr<void> is used rather than
        /// unique_ptr<void> or std::any because it retains the deleter for
        /// the concrete type without requiring the payload to be copyable or
        /// to derive from a common base.
        std::shared_ptr<void> data;
        /// Value of #m_generation when this payload was produced.
        std::uint64_t generation;
    };

    std::unordered_map<std::type_index, Entry> m_data;
    std::uint64_t m_generation = 0;
};

} // namespace Nektar::NekMesh

#endif
