///////////////////////////////////////////////////////////////////////////////
//
// File: BndCondPhysOpImpl.hpp
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

#include <map>

#include <boost/algorithm/string.hpp>

#include <LibUtilities/Interpreter/Interpreter.h>

#include <MultiRegions/DisContField.h>

#include "MultiRegions/DataWarehouse/TraceDataWarehouse.hpp"
#include "Operators/BndCondOps/BndCondPhys/BndCondPhysOp.hpp"

#include "Operators/ElmtOps/Expression/ExpressionOp.hpp"

using namespace Nektar;

namespace Nektar::Operators::detail
{

template <typename ExecSpace, typename TData>
class BndCondPhysOpImpl : public BndCondPhysOp<TData>
{
    using MemSpace = typename ExecSpace::memory_space;

public:
    BndCondPhysOpImpl(const MultiRegions::ExpListSharedPtr &expansionList,
                      const std::vector<std::string> &components)
        : BndCondPhysOp<TData>(expansionList, components)
    {
        auto session       = this->m_expansionList->GetSession();
        m_session          = session;
        auto graph         = this->m_expansionList->GetGraph();
        unsigned int nhomo = 1;
        session->LoadParameter("HomModesZ", nhomo, 1);
        m_numComp                = components.size();
        const unsigned int nComp = m_numComp;

        // Create boundary conditions
        const SpatialDomains::BoundaryConditions bcs(session, graph);
        const SpatialDomains::BoundaryRegionCollection &bregions =
            bcs.GetBoundaryRegions();
        const SpatialDomains::BoundaryConditionCollection &bconditions =
            bcs.GetBoundaryConditions();

        SpatialDomains::BoundaryConditionShPtr bc;
        MultiRegions::ExpListSharedPtr bcExpList;

        // Position of the region in GetBndCondExpansions(), which is not the
        // region id - ids need be neither contiguous nor zero based - and not
        // the storage block index either, because periodic regions are skipped
        // below while the expansion list still holds them. Anything indexing
        // the expansion list, such as the boundary normals, needs this.
        unsigned int regionOrdinal = 0;

        // Loop all boundary regions to create Fields and Operators
        for (auto &it : bregions)
        {
            const unsigned int thisOrdinal = regionOrdinal++;

            auto regionId       = it.first;
            auto collectionIter = bconditions.find(regionId);

            ASSERTL1(collectionIter != bconditions.end(),
                     "Unable to locate collection " + std::to_string(regionId));

            const SpatialDomains::BoundaryConditionMapShPtr bndCondMap =
                (*collectionIter).second;

            // Record what kind of condition each component carries here. The
            // values of every type are stored alike - what differs is how a
            // consumer must interpret them - so the type is kept alongside
            // rather than used to select what gets evaluated. The user defined
            // tag is kept with it: a condition such as an adiabatic wall is an
            // ordinary Dirichlet condition as far as this operator goes, and
            // only the tag says otherwise.
            std::vector<SpatialDomains::BoundaryConditionType> bndCondType(
                nComp, SpatialDomains::eNotDefined);
            std::vector<std::string> bndUserDefined(nComp);
            bool regionTimeDependent = false;
            unsigned int numPeriodic = 0;
            for (unsigned int nc = 0; nc < nComp; nc++)
            {
                auto cndMapIter = bndCondMap->find(components[nc]);

                ASSERTL1(cndMapIter != bndCondMap->end(),
                         "Unable to locate condition map.");

                bndCondType[nc] =
                    (*cndMapIter).second->GetBoundaryConditionType();
                bndUserDefined[nc] = (*cndMapIter).second->GetUserDefined();

                regionTimeDependent = regionTimeDependent ||
                                      (*cndMapIter).second->IsTimeDependent();

                numPeriodic += (bndCondType[nc] == SpatialDomains::ePeriodic);
            }

            // Periodic boundaries carry no values of their own: the interior
            // trace strategy pairs them with their partner instead.
            if (numPeriodic == nComp)
            {
                continue;
            }

            ASSERTL0(numPeriodic == 0,
                     "Boundary region " + std::to_string(regionId) +
                         " is periodic for some components but not others.");

            // Note use just the first component as we are just looking for the
            // boundary expansion list
            auto conditionMapIter = bndCondMap->find(components[0]);

            ASSERTL1(conditionMapIter != bndCondMap->end(),
                     "Unable to locate condition map.");

            // Get BoundaryCondition(SharedPtr)
            bc = (*conditionMapIter).second;

            // Create boundary condition expansion
            bcExpList = MemoryManager<MultiRegions::ExpList>::AllocateSharedPtr(
                session, *(it.second), graph, true, components[0], false,
                bc->GetComm(), Collections::eNoImpType);

            // Set data warehouse for device support operators
            bcExpList->SetDataWarehouse();

            // This region's storage block, and where its elements sit in it.
            const unsigned reg = m_regionNumPhys.size();
            m_regionNumPhys.push_back(bcExpList->GetTotPoints());
            m_regionOrdinal.push_back(thisOrdinal);
            m_bndCondType.push_back(bndCondType);
            {
                // Split "Cond:key=value:key=value" into the condition and its
                // parameters. Everything downstream compares against the
                // condition alone, so a region carrying parameters is claimed
                // by the same operator as one without.
                std::vector<std::string> cond(bndUserDefined.size());
                std::vector<std::map<std::string, std::string>> pars(
                    bndUserDefined.size());

                for (size_t c = 0; c < bndUserDefined.size(); ++c)
                {
                    SplitBndUserDefined(bndUserDefined[c], cond[c], pars[c]);
                }

                m_bndUserDefined.push_back(cond);
                m_bndUserDefinedParams.push_back(pars);
            }
            m_regionTimeDependent.push_back(regionTimeDependent);

            this->m_hasTimeDependentBndPhys =
                this->m_hasTimeDependentBndPhys || regionTimeDependent;

            for (size_t e = 0; e < bcExpList->GetExpSize(); ++e)
            {
                // A trace is read back out of the region it belongs to, at the
                // offset its boundary expansion already uses.
                const auto tid = bcExpList->GetExp(e)->GetGeom()->GetGlobalID();
                m_traceIdToBlk[tid]    = reg;
                m_traceIdToOffset[tid] = bcExpList->GetPhys_Offset(e);
            }

            // Create fields for evaluating BCs into
            auto blocks_phys =
                MultiRegions::GetBlockAttributes<TData, FieldState::Phys>(
                    bcExpList);
            this->m_wsp_phys.push_back(
                LibUtilities::Field<TData, FieldState::Phys>(
                    "Boundary condition phys", blocks_phys, nComp, nhomo));

            // Initialize memory regions
            m_wsp_phys.back().template Initialize<MemSpace>(0.0);

            // Create operators for this boundary condition. Every component of
            // a kept region is evaluated: each has a condition of some type,
            // and each type supplies an equation through GetEquation().
            this->m_expressionOps.push_back(
                ExpressionOp<TData>::Create(bcExpList, components, "Serial"));
            this->m_expressionOps.back()->SetComponentMask(
                std::vector<bool>(nComp, true));

            // Gather equations for each field/component
            std::vector<LibUtilities::EquationSharedPtr> listOfEquations;
            for (unsigned int nc = 0; nc < nComp; nc++)
            {
                // Get map for each field
                auto cndMapIter = bndCondMap->find(components[nc]);

                // Get BoundaryCondition and extract equation
                bc = (*cndMapIter).second;
                listOfEquations.push_back(bc->GetEquation());
            }

            // Set boundary conditions for each component in this boundary
            // region
            this->m_expressionOps.back()->SetExpressions(listOfEquations);
        }

        // Evaluate every region into its own storage block. Going through
        // UpdateBndPhys() rather than calling EvaluateRegion() directly
        // marks the values initialised, so a later call redoes only the
        // regions that depend on time.
        m_bndPhys.resize(m_regionNumPhys.size());
        this->UpdateBndPhys(static_cast<TData>(0.0));
    }

    /**
     * @brief Evaluate boundary region @p reg into its own storage block.
     *
     * Separate from the constructor so a region whose values change - time
     * dependent or user defined - can be refreshed on its own.
     */
    void EvaluateRegion(const unsigned int reg)
    {
        m_expressionOps[reg]->Apply(m_wsp_phys[reg], m_wsp_phys[reg]);

        // ToVector() strips the block padding, leaving the boundary
        // expansion's own component-major phys layout - which is the layout
        // the offsets in m_traceIdToOffset index into.
        std::vector<TData> vals = m_wsp_phys[reg].template ToVector<TData>();

        ASSERTL1(vals.size() == m_numComp * m_regionNumPhys[reg],
                 "Unexpected boundary phys vector size.");

        m_bndPhys[reg] =
            LibUtilities::MemoryRegion<TData>::template FromVector<MemSpace,
                                                                   TData>(vals);
    }

    // className - for OperatorFactory
    static std::string className;

    // instantiation function for CreatorFunction in OperatorFactory
    static std::unique_ptr<Operator<TData>> Instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList,
        const std::vector<std::string> &components)
    {
        return std::make_unique<BndCondPhysOpImpl<ExecSpace, TData>>(
            expansionList, components);
    }

protected:
    unsigned int m_numComp = 0;

    std::vector<LibUtilities::Field<TData, FieldState::Phys>> m_wsp_phys;
    std::vector<std::shared_ptr<ExpressionOp<TData>>> m_expressionOps;

    /// Boundary values, one storage block per non-periodic boundary region, in
    /// that region's own component-major phys layout. A region can therefore
    /// be re-evaluated on its own, and the memory manager sees the refresh of
    /// just that region.
    std::vector<LibUtilities::MemoryRegion<TData>> m_bndPhys;
    /// Phys points in each region, i.e. the stride between its components.
    std::vector<size_t> m_regionNumPhys;

    /// Position of each block's region in GetBndCondExpansions(), which the
    /// block index does not give once periodic regions are skipped.
    std::vector<unsigned int> m_regionOrdinal;
    /// Condition type each component of a block carries, indexed as m_bndPhys.
    /// The values themselves are stored the same way whatever the type; this
    /// is what tells a consumer how to interpret them.
    std::vector<std::vector<SpatialDomains::BoundaryConditionType>>
        m_bndCondType;
    /// USERDEFINEDTYPE of each component, indexed as m_bndCondType. Empty when
    /// the session gives none. Left as the session's string rather than mapped
    /// to a code: the tags are solver specific - "WallAdiabatic" means nothing
    /// here - so the solver that understands them does the mapping.
    /// Kept so that a boundary tag parameter can be evaluated against the
    /// session's own interpreter, long after construction.
    LibUtilities::SessionReaderSharedPtr m_session;
    std::vector<std::vector<std::string>> m_bndUserDefined;
    /// Parameters carried on each tag, by lower-cased name. Indexed as
    /// #m_bndUserDefined.
    std::vector<std::vector<std::map<std::string, std::string>>>
        m_bndUserDefinedParams;
    /// Whether any component of a region depends on time, so UpdateBndPhys()
    /// re-evaluates only the regions that can actually have changed.
    std::vector<bool> m_regionTimeDependent;

    // given trace elmtid (tid) return which region it is in
    std::map<size_t, unsigned> m_traceIdToBlk;
    // given trace elmtid (tid) offset in that region
    std::map<size_t, size_t> m_traceIdToOffset;

    // access the data associated with trace id == tid and the component offset
    size_t v_GetNumBlocks() const override
    {
        return m_bndPhys.size();
    }

    SpatialDomains::BoundaryConditionType v_GetBndCondType(
        const unsigned blk, const unsigned comp) const override
    {
        ASSERTL1(blk < m_bndCondType.size(), "Block index out of range");
        ASSERTL1(comp < m_bndCondType[blk].size(),
                 "Component index out of range");
        return m_bndCondType[blk][comp];
    }

    void v_UpdateBndPhys(const TData &time) override
    {
        for (unsigned int reg = 0; reg < m_bndPhys.size(); ++reg)
        {
            // The constructor has already evaluated everything, so on a
            // re-entry only the regions that depend on time can have moved.
            if (this->m_bndPhysInitialised && !m_regionTimeDependent[reg])
            {
                continue;
            }

            m_expressionOps[reg]->SetTime(time);
            EvaluateRegion(reg);
        }
    }

    /**
     * @brief Split a boundary tag into its condition and named parameters.
     *
     * `WallViscous:twall=300.15` gives `WallViscous` and `{twall: "300.15"}`.
     * The colon separation follows the `Rotated:` periodic tag; the names are
     * this convention's own, positional fields being unreadable once a
     * condition takes more than one optional parameter.
     *
     * A field with no `=` is kept as a parameter with an empty value rather
     * than rejected, so a future positional or flag-like use has somewhere to
     * go without changing the parsing.
     */
    static void SplitBndUserDefined(const std::string &tag,
                                    std::string &condition,
                                    std::map<std::string, std::string> &params)
    {
        std::vector<std::string> fields;
        boost::split(fields, tag, boost::is_any_of(":"));

        condition = fields.empty() ? std::string() : fields[0];
        boost::trim(condition);

        for (size_t f = 1; f < fields.size(); ++f)
        {
            const auto eq = fields[f].find('=');

            std::string key =
                (eq == std::string::npos) ? fields[f] : fields[f].substr(0, eq);
            std::string val = (eq == std::string::npos)
                                  ? std::string()
                                  : fields[f].substr(eq + 1);
            boost::trim(key);
            boost::trim(val);
            boost::to_lower(key);

            params[key] = val;
        }
    }

    bool v_GetBndUserDefinedParam(const unsigned blk, const unsigned comp,
                                  const std::string &name,
                                  TData &value) const override
    {
        ASSERTL1(blk < m_bndUserDefinedParams.size(),
                 "Block index out of range");
        ASSERTL1(comp < m_bndUserDefinedParams[blk].size(),
                 "Component index out of range");

        std::string key = name;
        boost::to_lower(key);

        const auto &params = m_bndUserDefinedParams[blk][comp];
        const auto it      = params.find(key);

        if (it == params.end())
        {
            return false;
        }

        ASSERTL0(!it->second.empty(), "Boundary tag parameter '" + name +
                                          "' on block " + std::to_string(blk) +
                                          " has no value; expected " + name +
                                          "=<expression>.");

        // Evaluated through the *session's* interpreter, not a bare one, so
        // that the expression may name session parameters - twall=TInf - and
        // not only arithmetic and built-in constants. The Rotated periodic tag
        // uses a bare interpreter and so gets PI/15 but would not get TInf;
        // this is the more useful of the two and costs nothing.
        ASSERTL0(m_session, "No session available to evaluate boundary tag "
                            "parameter '" +
                                name + "'.");

        auto interp = m_session->GetInterpreter();
        value       = interp->Evaluate(interp->DefineFunction("", it->second));

        return true;
    }

    const std::string &v_GetBndUserDefined(const unsigned blk,
                                           const unsigned comp) const override
    {
        ASSERTL1(blk < m_bndUserDefined.size(), "Block index out of range");
        ASSERTL1(comp < m_bndUserDefined[blk].size(),
                 "Component index out of range");
        return m_bndUserDefined[blk][comp];
    }

    void v_GetTraceLocation(const size_t tid, unsigned &blk, size_t &offset,
                            size_t &compOffset) override
    {
        ASSERTL1(m_traceIdToBlk.count(tid), " Cannot find trace id in map");
        blk    = m_traceIdToBlk[tid];
        offset = m_traceIdToOffset[tid];

        // the components of a region follow one another, so the stride is the
        // number of phys points in the region
        compOffset = m_regionNumPhys[blk];
    }

    bool v_HasTrace(const size_t tid) const override
    {
        return m_traceIdToBlk.count(tid) > 0;
    }

    const TData *v_GetBlockPtr(const unsigned blk) override
    {
        ASSERTL1(blk < m_bndPhys.size(), "Block index out of range");
        return m_bndPhys[blk].template GetPtr<MemSpace, ReadOnly>();
    }

    TData *v_UpdateBlockPtr(const unsigned blk) override
    {
        ASSERTL1(blk < m_bndPhys.size(), "Block index out of range");
        return m_bndPhys[blk].template GetPtr<MemSpace, ReadWrite>();
    }

    size_t v_GetBlockCompStride(const unsigned blk) const override
    {
        ASSERTL1(blk < m_regionNumPhys.size(), "Block index out of range");
        return m_regionNumPhys[blk];
    }

    const TData *v_GetBlockNormals(const unsigned blk) override
    {
        ASSERTL1(blk < m_regionOrdinal.size(), "Block index out of range");

        // Keyed on the region's position in GetBndCondExpansions() rather than
        // on the block, since the two part company as soon as the session has
        // a periodic boundary.
        return this->m_dataWarehouse->template GetData<MemSpace>(
            MultiRegions::BndCondNormalKey<TData>(m_regionOrdinal[blk]));
    }
};

} // namespace Nektar::Operators::detail
