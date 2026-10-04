///////////////////////////////////////////////////////////////////////////////
//
// File: GatherScatter.h
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
// Description: Gather-scatter handle selectable between gslib and
// SharedIdPlan.
//
///////////////////////////////////////////////////////////////////////////////

#ifndef NEKTAR_LIBUTILITIES_COMMUNICATION_GATHERSCATTER_H
#define NEKTAR_LIBUTILITIES_COMMUNICATION_GATHERSCATTER_H

#include <memory>
#include <string>

#include <LibUtilities/BasicUtils/SessionReader.h>
#include <LibUtilities/BasicUtils/SharedArray.hpp>
#include <LibUtilities/Communication/Comm.h>
#include <LibUtilities/LibUtilitiesDeclspec.h>

namespace Nektar::LibUtilities
{

/**
 * @brief Fold a #GatherScatter applies across every slot sharing an id.
 *
 * Mirrors the subset of gslib's `gs_op` Nektar uses, so that no caller needs
 * gslib's headers to ask for a gather.
 */
enum class GatherScatterOp
{
    eAdd,   ///< sum of the contributions
    eMul,   ///< product of the contributions
    eMin,   ///< smallest contribution
    eMax,   ///< largest contribution
    eAbsMax ///< the contribution of largest magnitude, sign kept
};

/**
 * @brief Which implementation a #GatherScatter uses, selected with the
 * `--gather-scatter` command-line option.
 */
enum class GatherScatterMethod
{
    eGslib,     ///< gslib's gs_setup / gs_gather / gs_unique (the default)
    ePairwise,  ///< SharedIdPlan over ExchangeBackend::ePairwise
    eNeighbour, ///< SharedIdPlan over ExchangeBackend::eNeighbourCollective
    eCrystal    ///< SharedIdPlan replaying through a CrystalRouterTransport
};

/**
 * @brief Transport a SharedIdPlan's one-off discovery runs over, selected with
 * `--gather-scatter-discovery`. Ignored for GatherScatterMethod::eGslib.
 */
enum class GatherScatterDiscovery
{
    eCrystalRouter, ///< CrystalRouterTransport: no O(P) arrays (the default)
    eAlltoallv      ///< AlltoallvTransport
};

/// A gather-scatter method and discovery transport, as chosen on the command
/// line.
struct GatherScatterConfig
{
    GatherScatterMethod method       = GatherScatterMethod::eGslib;
    GatherScatterDiscovery discovery = GatherScatterDiscovery::eCrystalRouter;

    /**
     * @brief Read `--gather-scatter` and `--gather-scatter-discovery` from
     * @p session. Anything not given keeps its default (gslib), as does
     * everything for a null session.
     */
    LIB_UTILITIES_EXPORT static GatherScatterConfig FromSession(
        const SessionReaderSharedPtr &session);

    /// @return A one-line description, e.g. for verbose output.
    LIB_UTILITIES_EXPORT std::string Describe() const;
};

/**
 * @class GatherScatter
 * @brief A gather-scatter over universal ids, mirroring Gs::Init(),
 * Gs::Gather() and Gs::Unique() behind an implementation-agnostic interface.
 *
 * @details
 * Create() picks the implementation from a #GatherScatterConfig: gslib's
 * `gs_data`, or a #SharedIdPlan over one of its exchange backends. The
 * subclasses live entirely in the implementation file, so a caller needs
 * neither gslib's headers nor the shared-id module's to use one. This is what
 * lets a solver be benchmarked with and without gslib just by adding, say,
 * `--gather-scatter pairwise` to its command line.
 *
 * On a serial communicator Create() returns the same do-nothing handle
 * CreateNoOp() gives, as gslib is a no-op there too. Create() and Gather()
 * are collective over the communicator, as is Unique() for gslib.
 */
class GatherScatter
{
public:
    /**
     * @brief Set up a gather-scatter over @p ids: the Gs::Init() equivalent.
     *
     * @param ids     One universal id per slot; 0 for slots to skip. The
     *                SharedIdPlan methods also skip negative ids, which
     *                gslib treats as "flagged" (see the implementation).
     * @param comm    Communicator to gather over.
     * @param config  Implementation to use.
     * @param verbose Print a setup summary on rank 0.
     * @return A handle, never null.
     */
    LIB_UTILITIES_EXPORT static std::shared_ptr<GatherScatter> Create(
        const Array<OneD, long> &ids, const CommSharedPtr &comm,
        const GatherScatterConfig &config, bool verbose = false);

    /**
     * @brief A handle whose Gather() and Unique() do nothing.
     *
     * For slots that never had a gather-scatter set up over them, which
     * callers previously expressed as a null `Gs::gs_data *` and relied on
     * Gs::Gather() ignoring.
     */
    LIB_UTILITIES_EXPORT static std::shared_ptr<GatherScatter> CreateNoOp();

    LIB_UTILITIES_EXPORT virtual ~GatherScatter() = default;

    GatherScatter(const GatherScatter &)            = delete;
    GatherScatter &operator=(const GatherScatter &) = delete;

    /**
     * @brief The Gs::Gather() equivalent: fold the entries of @p u, in
     * place, across every slot on every rank with the same id.
     *
     * As with gslib, a slot whose id is neither shared nor repeated locally
     * is never accessed, so @p u may be shorter than the id array this
     * handle was built from provided it covers every other slot.
     * AssemblyMap's multi-level static condensation relies on this.
     */
    LIB_UTILITIES_EXPORT virtual void Gather(Array<OneD, NekDouble> u,
                                             GatherScatterOp op) = 0;

    /**
     * @brief The Gs::Unique() equivalent: negate all but one occurrence of
     * each id across all ranks, in place. @p ids must hold the ids this
     * handle was built from. Which occurrence survives differs between
     * implementations; that exactly one does is what callers rely on.
     */
    LIB_UTILITIES_EXPORT virtual void Unique(Array<OneD, long> &ids) = 0;

protected:
    GatherScatter() = default;
};

typedef std::shared_ptr<GatherScatter> GatherScatterSharedPtr;

} // namespace Nektar::LibUtilities

#endif
