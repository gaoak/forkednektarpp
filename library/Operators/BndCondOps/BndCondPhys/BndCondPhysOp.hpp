///////////////////////////////////////////////////////////////////////////////
//
// File: BndCondPhysOp.hpp
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

#include <SpatialDomains/Conditions.h>

#include "Operators/Common/Operator.hpp"

namespace Nektar::Operators
{

// Boundary condition phys-value operator base class.
// Holds the evaluated values of every non-periodic boundary region - Dirichlet,
// Neumann or Robin - one storage block per region. Periodic boundaries are not
// held here: the interior trace strategy pairs them with their partner.
template <typename TData> class BndCondPhysOp : public Operator<TData>
{

public:
    static std::shared_ptr<BndCondPhysOp<TData>> Create(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components,
        const std::string &execStr = "")
    {
        return Operator<TData>::template Create<BndCondPhysOp>(
            expansionList, components, execStr);
    }

    static inline const std::string name = "BndCondPhys";

    /// Number of blocks the boundary data is stored in.
    size_t GetNumBlocks() const
    {
        return v_GetNumBlocks();
    }

    /// Components held per block, i.e. the range valid for GetBndCondType().
    size_t GetNumComponents() const
    {
        return this->m_components.size();
    }

    /**
     * @brief Bring the stored values up to date for time @p time.
     *
     * The constructor evaluates every region once, so this is only needed
     * where a condition actually depends on time; it returns immediately
     * otherwise. Callers on the DG path should invoke it once per step, as
     * LinearSystemOp does with the corresponding CG operators - without it a
     * time-dependent condition stays pinned at the value it had when the
     * operator was built.
     *
     * Consumers need do nothing to see the new values: they reach the data
     * through GetBlockPtr(), which is refetched per apply.
     */
    void UpdateBndPhys(const TData &time = 0.0)
    {
        if (m_bndPhysInitialised && !m_hasTimeDependentBndPhys)
        {
            return;
        }
        v_UpdateBndPhys(time);
        m_bndPhysInitialised = true;
    }

    /**
     * @brief Where the data for global trace @p tid lives.
     *
     * The mapping is fixed once the operator is constructed, so a caller that
     * gathers many traces can resolve it up front and keep only the integers,
     * rather than paying a virtual call and two map lookups per trace. The
     * base pointer is deliberately not returned: it must come from
     * GetBlockPtr() at the point of use, so that the memory manager still sees
     * every access.
     */
    void GetTraceLocation(const size_t tid, unsigned &blk, size_t &offset,
                          size_t &compOffset)
    {
        v_GetTraceLocation(tid, blk, offset, compOffset);
    }

    /**
     * @brief Does this operator hold boundary data for global trace @p tid?
     *
     * GetTraceLocation() assumes it does and asserts otherwise, which only
     * reports in a debug build. This is the question asked before that, by a
     * caller holding a trace it cannot pair locally: a trace on no boundary
     * region of this rank is one whose neighbour lives on another, and it is
     * the answer here rather than an out-of-range read that must decide
     * which.
     */
    bool HasTrace(const size_t tid) const
    {
        return v_HasTrace(tid);
    }

    /// Base pointer for block @p blk, fetched through the memory manager.
    const TData *GetBlockPtr(const unsigned blk)
    {
        return v_GetBlockPtr(blk);
    }

    /**
     * @brief Writable base pointer for block @p blk.
     *
     * For an operator that *computes* boundary values rather than evaluating
     * them from the session - a wall state derived from the interior trace,
     * say. Writing here rather than intercepting the gather keeps the flux
     * path unchanged: consumers refetch through GetBlockPtr() on every apply
     * and so pick the new values up without knowing where they came from.
     *
     * The layout is the region's own component-major phys layout, the same one
     * GetTraceLocation() resolves into.
     */
    TData *UpdateBlockPtr(const unsigned blk)
    {
        return v_UpdateBlockPtr(blk);
    }

    /// Phys points in block @p blk, i.e. the stride between its components and
    /// equally the number of points to walk when transforming the whole block.
    size_t GetBlockCompStride(const unsigned blk) const
    {
        return v_GetBlockCompStride(blk);
    }

    /**
     * @brief Outward normals of block @p blk, in the same layout as its
     * values: component major, #GetBlockCompStride() apart, one component per
     * coordinate direction.
     *
     * Cached in the data warehouse, so repeated calls are free and every
     * boundary condition that needs a normal direction - a symmetry plane, a
     * characteristic inflow, an outflow deciding whether it is subsonic -
     * draws on one copy.
     */
    const TData *GetBlockNormals(const unsigned blk)
    {
        return v_GetBlockNormals(blk);
    }

    /**
     * @brief Condition type component @p comp of block @p blk carries.
     *
     * Values of every type are stored identically; what differs is how they
     * must be interpreted, so a consumer that treats a Neumann or Robin
     * boundary differently from a Dirichlet one asks here. Never
     * SpatialDomains::ePeriodic - those regions hold no values and are not
     * given a block.
     */
    SpatialDomains::BoundaryConditionType GetBndCondType(
        const unsigned blk, const unsigned comp) const
    {
        return v_GetBndCondType(blk, comp);
    }

    /**
     * @brief USERDEFINEDTYPE of component @p comp of block @p blk.
     *
     * Empty when the session gives none. The tags are solver specific, so this
     * hands back the session's own string rather than a code: a compressible
     * flow solver knows what "WallAdiabatic" means and this operator does not.
     * A condition so tagged is still an ordinary condition of its
     * GetBndCondType() as far as the storage here goes.
     *
     * Host side only - the string is not something a kernel can read. A caller
     * that needs the distinction on device should resolve these once into its
     * own per-block table of codes, as it must do for its own enum anyway.
     */
    const std::string &GetBndUserDefined(const unsigned blk,
                                         const unsigned comp) const
    {
        return v_GetBndUserDefined(blk, comp);
    }

    /**
     * @brief A named parameter carried on the boundary tag.
     *
     * A USERDEFINEDTYPE may carry parameters after the condition itself,
     *
     *     USERDEFINEDTYPE="WallViscous:twall=300.15"
     *     USERDEFINEDTYPE="WallViscous:twall=TInf"
     *
     * so that a quantity belonging to one region is written on that region
     * rather than as a session-wide parameter. Two regions may then hold
     * different values, which a session parameter cannot express.
     *
     * The value is evaluated as an expression against the session, as the
     * `Rotated:` periodic tag does, so it may name session parameters and use
     * arithmetic rather than having to be a literal.
     *
     * GetBndUserDefined() returns the condition with any such parameters
     * stripped, so a condition that takes none - and the check that every tag
     * is claimed by some operator - are unaffected by their presence.
     *
     * @param blk   - Storage block.
     * @param comp  - Component; a region carries one tag for all of them.
     * @param name  - Parameter name, matched without regard to case.
     * @param value - Set only when the parameter is present, so a caller may
     *                seed it with a default and ignore the return.
     *
     * @return Whether the parameter was present.
     */
    bool GetBndUserDefinedParam(const unsigned blk, const unsigned comp,
                                const std::string &name, TData &value) const
    {
        return v_GetBndUserDefinedParam(blk, comp, name, value);
    }

protected:
    /// Set once the constructor's initial evaluation has happened, so that
    /// UpdateBndPhys() can skip regions that will never change.
    bool m_bndPhysInitialised      = false;
    bool m_hasTimeDependentBndPhys = false;

    BndCondPhysOp(const MultiRegions::ExpListSharedPtr &expansionList,
                  const std::vector<std::string> &components)
        : Operator<TData>(expansionList, components)
    {
    }

    ~BndCondPhysOp() override = default;

    virtual size_t v_GetNumBlocks() const = 0;

    virtual void v_GetTraceLocation(const size_t tid, unsigned &blk,
                                    size_t &offset, size_t &compOffset) = 0;

    virtual bool v_HasTrace(const size_t tid) const = 0;

    virtual const TData *v_GetBlockPtr(const unsigned blk) = 0;

    virtual TData *v_UpdateBlockPtr(const unsigned blk) = 0;

    virtual size_t v_GetBlockCompStride(const unsigned blk) const = 0;

    virtual const TData *v_GetBlockNormals(const unsigned blk) = 0;

    virtual SpatialDomains::BoundaryConditionType v_GetBndCondType(
        const unsigned blk, const unsigned comp) const = 0;

    virtual bool v_GetBndUserDefinedParam(const unsigned blk,
                                          const unsigned comp,
                                          const std::string &name,
                                          TData &value) const = 0;

    virtual const std::string &v_GetBndUserDefined(
        const unsigned blk, const unsigned comp) const = 0;

    virtual void v_UpdateBndPhys(const TData &time) = 0;
};

} // namespace Nektar::Operators
