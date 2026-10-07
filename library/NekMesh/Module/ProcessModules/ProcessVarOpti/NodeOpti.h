////////////////////////////////////////////////////////////////////////////////
//
//  File: NodeOpti.h
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
//  Description:
//
////////////////////////////////////////////////////////////////////////////////

#ifndef UTILITIES_NEKMESH_NODEOPTI
#define UTILITIES_NEKMESH_NODEOPTI

#include <algorithm>
#include <mutex>
#include <ostream>

#include <LibUtilities/BasicUtils/HashUtils.hpp>
#include <LibUtilities/BasicUtils/Thread.h>

#include "Evaluator.hxx"
#include "Functionals.hxx"
#include "ProcessVarOpti.h"

namespace Nektar::NekMesh
{

class NodeOptiJob;

class NodeOpti
{
public:
    NodeOpti(SpatialDomains::PointGeom *n, std::vector<ElUtilSharedPtr> e,
             ResidualSharedPtr r,
             std::map<LibUtilities::ShapeType, DerivUtilSharedPtr> d,
             optiType o, int dim, SpatialDomains::MeshGraph *g)
        : m_node(n), m_graph(g), m_res(r), m_opti(o)
    {
        // Group the elements by shape. Held as a flat list with a block per
        // shape rather than as a map keyed by shape: this object exists once
        // per free node of the mesh, and a std::map of each thing it needs
        // came to more memory than the data.
        std::map<LibUtilities::ShapeType, std::vector<ElUtil *>> byShape;
        for (auto &el : e)
        {
            byShape[el->GetEl()->GetShapeType()].push_back(el.get());
        }

        for (auto &[shape, els] : byShape)
        {
            auto it = d.find(shape);
            ASSERTL0(it != d.end(),
                     std::string("No integration rule for element type ") +
                         LibUtilities::ShapeTypeMap[shape]);

            m_elements.insert(m_elements.end(), els.begin(), els.end());
        }

        // Where this node sits in each of those elements' node lists. Looked
        // up once here rather than on every evaluation: it is the column of
        // the derivative operator the whole local problem is built around.
        m_nodeIds.reserve(m_elements.size());
        for (ElUtil *el : m_elements)
        {
            m_nodeIds.push_back(el->NodeId(n));
        }

        (void)dim;
    }

    virtual ~NodeOpti() {};

    virtual void Optimise() = 0;

    /**
     * @brief Evaluate the functional at the node's current position, and with
     * @p gradient its derivatives too, which are left in m_grad.
     */
    template <int DIM>
    NekDouble GetFunctional(NekDouble &minJacNew, bool gradient = true);

    /**
     * @brief Evaluate the functional with the node displaced by @p offset,
     * without moving it.
     *
     * Only one node moves, and the mapping is linear in its position, so the
     * derivatives each element holds change by a rank-one update, which is
     * applied as they are read. A line search can therefore try a step
     * without touching the mesh, and a rejected one costs nothing to undo.
     */
    template <int DIM>
    NekDouble GetFunctionalAt(const NekDouble *offset, NekDouble &minJacNew);

    /**
     * @brief Move the node by @p offset, and carry the change through to the
     * derivatives its elements hold.
     */
    template <int DIM> void MoveNode(const NekDouble *offset)
    {
        for (int d = 0; d < DIM; ++d)
        {
            (*m_node)[d] += offset[d];
        }

        for (size_t i = 0; i < m_elements.size(); ++i)
        {
            m_elements[i]->MoveNode(m_nodeIds[i], offset, DIM);
        }
    }

    /**
     * @brief The gradient and Hessian that the last evaluation left behind:
     * DIM gradient entries followed by the upper triangle of the Hessian,
     * row by row.
     */
    const std::array<NekDouble, 9> &GetGrad() const
    {
        return m_grad;
    }

    template <int DIM> void MinEigen(NekDouble &val);

protected:
    SpatialDomains::PointGeom *m_node;
    /// Holds the node to CAD association; only the CAD-constrained subclasses
    /// use it.
    SpatialDomains::MeshGraph *m_graph;
    /// Guards the shared Residual. One mutex for all nodes: a per-object one
    /// locks nothing, since each node holds its own.
    NEKMESH_EXPORT static std::mutex mtx;
    /// The elements around the node, ordered so that each block of m_blocks
    /// is contiguous. Not owning: ProcessVarOpti::m_dataSet holds them.
    std::vector<ElUtil *> m_elements;
    /// This node's index within each of those elements' node lists.
    std::vector<int> m_nodeIds;
    /// The gradient and Hessian of the last evaluation. Only live while this
    /// node is being optimised, so it could belong to the thread rather than
    /// to the node -- but it is written in the innermost loop, and reaching
    /// it through thread-local storage there costs more than the seventy-two
    /// bytes per node are worth.
    std::array<NekDouble, 9> m_grad;

    template <int DIM> int IsIndefinite();

    template <int DIM>
    NekDouble Evaluate(const NekDouble *offset, NekDouble &minJacNew,
                       bool gradient);

    template <int DIM, typename Energy>
    NekDouble Integrate(const NekDouble *offset, NekDouble &minJacNew,
                        bool gradient);

    ResidualSharedPtr m_res;
    optiType m_opti;

    static NekDouble c1()
    {
        return 1e-3;
    }
    static NekDouble gradTol()
    {
        return 1e-20;
    }
    static NekDouble alphaTol()
    {
        return 1e-8;
    }
};

typedef std::shared_ptr<NodeOpti> NodeOptiSharedPtr;
typedef LibUtilities::NekFactory<
    int, NodeOpti, SpatialDomains::PointGeom *, std::vector<ElUtilSharedPtr>,
    ResidualSharedPtr, std::map<LibUtilities::ShapeType, DerivUtilSharedPtr>,
    optiType, SpatialDomains::MeshGraph *>
    NodeOptiFactory;

NEKMESH_EXPORT NodeOptiFactory &GetNodeOptiFactory();

class NodeOpti3D3D : public NodeOpti // 1D optimsation in 3D space
{
public:
    NodeOpti3D3D(SpatialDomains::PointGeom *n, std::vector<ElUtilSharedPtr> e,
                 ResidualSharedPtr r,
                 std::map<LibUtilities::ShapeType, DerivUtilSharedPtr> d,
                 optiType o, SpatialDomains::MeshGraph *g)
        : NodeOpti(n, e, r, d, o, 3, g)
    {
    }

    ~NodeOpti3D3D() override {};

    void Optimise() override;

    static int m_type;
    static NodeOptiSharedPtr create(
        SpatialDomains::PointGeom *n, std::vector<ElUtilSharedPtr> e,
        ResidualSharedPtr r,
        std::map<LibUtilities::ShapeType, DerivUtilSharedPtr> d, optiType o,
        SpatialDomains::MeshGraph *g)
    {
        return NodeOptiSharedPtr(new NodeOpti3D3D(n, e, r, d, o, g));
    }

private:
};

class NodeOpti2D2D : public NodeOpti // 1D optimsation in 3D space
{
public:
    NodeOpti2D2D(SpatialDomains::PointGeom *n, std::vector<ElUtilSharedPtr> e,
                 ResidualSharedPtr r,
                 std::map<LibUtilities::ShapeType, DerivUtilSharedPtr> d,
                 optiType o, SpatialDomains::MeshGraph *g)
        : NodeOpti(n, e, r, d, o, 2, g)
    {
    }

    ~NodeOpti2D2D() override {};

    void Optimise() override;

    static int m_type;
    static NodeOptiSharedPtr create(
        SpatialDomains::PointGeom *n, std::vector<ElUtilSharedPtr> e,
        ResidualSharedPtr r,
        std::map<LibUtilities::ShapeType, DerivUtilSharedPtr> d, optiType o,
        SpatialDomains::MeshGraph *g)
    {
        return NodeOptiSharedPtr(new NodeOpti2D2D(n, e, r, d, o, g));
    }

private:
};

/**
 * @brief Optimises a run of nodes from one colour set.
 *
 * A job per node meant one allocation, one lock of the queue and one
 * broadcast for every node of the mesh on every iteration, to do a few
 * microseconds of work. The nodes of a colour set are independent of each
 * other by construction, so any run of them can be handed over together.
 */
class NodeOptiJob : public Thread::ThreadJob
{
public:
    NodeOptiJob(NodeOptiSharedPtr *nodes, int count)
        : m_nodes(nodes), m_count(count)
    {
    }

    void Run() override
    {
        for (int i = 0; i < m_count; ++i)
        {
            m_nodes[i]->Optimise();
        }
    }

private:
    NodeOptiSharedPtr *m_nodes;
    int m_count;
};
/**
 * @brief Evaluate functional for elements connected to a node.
 *
 * @param minJacNew   Stores current minimum Jacobian for the element group
 * @param gradient    If true, calculate gradient.
 */
template <int DIM>
NekDouble NodeOpti::GetFunctional(NekDouble &minJacNew, bool gradient)
{
    const NekDouble noOffset[3] = {0.0, 0.0, 0.0};
    return Evaluate<DIM>(noOffset, minJacNew, gradient);
}

template <int DIM>
NekDouble NodeOpti::GetFunctionalAt(const NekDouble *offset,
                                    NekDouble &minJacNew)
{
    return Evaluate<DIM>(offset, minJacNew, false);
}

template <int DIM>
NekDouble NodeOpti::Evaluate(const NekDouble *offset, NekDouble &minJacNew,
                             bool gradient)
{
    switch (m_opti)
    {
        case eLinEl:
            return Integrate<DIM, LinearElasticEnergy>(offset, minJacNew,
                                                       gradient);
        case eHypEl:
            return Integrate<DIM, HyperElasticEnergy>(offset, minJacNew,
                                                      gradient);
        case eRoca:
            return Integrate<DIM, DistortionEnergy>(offset, minJacNew,
                                                    gradient);
        case eWins:
            return Integrate<DIM, WinslowEnergy>(offset, minJacNew, gradient);
    }

    NEKERROR(ErrorUtil::efatal, "unknown optimisation functional");
    return 0.0;
}

/**
 * @brief Integrate an energy functional, and its derivatives with respect to
 * the node being optimised, over the elements connected to that node.
 *
 * @param offset     Displacement of the node from where it currently is.
 * @param minJacNew  Set to the smallest Jacobian seen.
 * @param gradient   If true, leave the gradient and Hessian in m_grad.
 *
 * Everything here is common to the four functionals: all that distinguishes
 * them is how @p Energy combines the quantities gathered into Deformation.
 * They used to be four copies of this loop, and each copy had drifted --
 * two of them weighted the Hessian wrongly and a third built its strain
 * tensor wrongly, in each case without disturbing the other three.
 */
template <int DIM, typename Energy>
NekDouble NodeOpti::Integrate(const NekDouble *offset, NekDouble &minJacNew,
                              bool gradient)
{
    constexpr int nHess = HessianSize(DIM);

    minJacNew          = std::numeric_limits<double>::max();
    NekDouble integral = 0.0;

    // The Jacobian regularisation parameter. Keeping it small even where the
    // elements are valid is what lets the same functional untangle an invalid
    // one; see section 2.2 of the paper. It belongs to the mesh rather than
    // to this node, so that every local problem of an iteration minimises the
    // same functional.
    //
    // It is capped. Tying it to the worst Jacobian in the mesh keeps J_R
    // comfortably above zero for a badly inverted element, but J here is
    // normalised -- a perfect element has J = 1 -- and as delta approaches
    // that, J_R tends to delta for everything and the functional stops
    // telling a good element from a degenerate one. One catastrophically
    // inverted element can then flatten the energy over the whole mesh,
    // which is what a boundary layer mesh with a few broken elements does:
    // the minimum Jacobian runs away, delta follows it, and the optimiser
    // stalls. The cap is the largest delta that leaves J_R within about six
    // percent of J for a perfect element. Note that delta also had to track
    // the worst Jacobian to keep the old form of J_R away from its
    // cancellation, which is no longer a reason: it is computed stably below.
    constexpr NekDouble epMax = 0.25;

    const NekDouble minJac = m_res->minJac;
    const NekDouble ep =
        minJac < 0.0 ? std::min(sqrt(1e-8 + 0.04 * minJac * minJac), epMax)
                     : 1e-4;

    m_grad.fill(0.0);

    // Hoisted out of the quadrature loop below, and zeroed because the energy
    // leaves them alone when no derivatives were asked for.
    NekDouble dW[DIM]    = {};
    NekDouble d2W[nHess] = {};

    for (size_t e = 0; e < m_elements.size(); ++e)
    {
        ElUtil *el                  = m_elements[e];
        DerivUtil *derivUtil        = el->GetDerivUtil();
        const int pts               = derivUtil->pts;
        NekVector<NekDouble> &quadW = derivUtil->quadW;

        // The column of the derivative operator belonging to the node being
        // moved is the same at every integration point, and is both the
        // rank-one update of the element's derivatives and the basis
        // derivative the gradient needs.
        const int nodeId = m_nodeIds[e];

        const NekDouble *basis[DIM];
        for (int d = 0; d < DIM; ++d)
        {
            basis[d] = derivUtil->VdmD[d].GetRawPtr() + nodeId * pts;
        }

        const NekDouble *elDeriv = el->deriv.data();

        {
            for (int k = 0; k < pts; ++k)
            {
                const NekDouble *map = &el->maps[k * el->mapStride];

                Deformation<DIM> d;

                NekDouble basisDeriv[DIM];
                for (int m = 0; m < DIM; ++m)
                {
                    basisDeriv[m] = basis[m][k];
                }

                NekDouble phiM[DIM][DIM];
                for (int l = 0; l < DIM; ++l)
                {
                    for (int n = 0; n < DIM; ++n)
                    {
                        phiM[n][l] = elDeriv[(l * DIM + n) * pts + k] +
                                     offset[n] * basisDeriv[l];
                    }
                }

                for (int m = 0; m < DIM; ++m)
                {
                    for (int n = 0; n < DIM; ++n)
                    {
                        d.jacIdeal[n][m] = 0.0;
                        for (int l = 0; l < DIM; ++l)
                        {
                            d.jacIdeal[n][m] += phiM[n][l] * map[m * 3 + l];
                        }
                    }
                }

                d.jacDet  = Determinant<DIM>(d.jacIdeal);
                minJacNew = std::min(minJacNew, d.jacDet);

                // The ideal mapping does not move with the node, so its
                // determinant is a constant here. It is the volume element of
                // the integral, for which only its magnitude is wanted, and it
                // also divides det(grad phi_M) to give J, for which its sign
                // matters: a straight-sided element that is itself inverted
                // has a negative one, and taking the magnitude there points
                // dJ/dx the wrong way.
                const NekDouble idealMapDet    = map[9];
                const NekDouble absIdealMapDet = fabs(idealMapDet);

                // The regularised Jacobian, J_R = (J + sqrt(4 d^2 + J^2))/2.
                // Written that way it loses every significant digit once J is
                // negative and much larger than d: the square root rounds to
                // -J, the sum cancels to zero, ln(J_R) is -inf and the node
                // goes to NaN. The conjugate form is the same number without
                // the cancellation, and J is only ever negative while
                // untangling, which is where this has to hold up.
                const NekDouble root =
                    sqrt(d.jacDet * d.jacDet + 4.0 * ep * ep);

                d.sigma = d.jacDet >= 0.0 ? 0.5 * (d.jacDet + root)
                                          : 2.0 * ep * ep / (root - d.jacDet);

                // 2 J_R - J is the square root exactly; forming it by
                // subtraction would reintroduce the cancellation above.
                d.twoSigmaMinusJ = root;

                // sigma is now positive for any finite J, so this catches
                // only a node that has already been taken somewhere
                // non-finite. Refuse the configuration rather than bringing
                // the run down: in a line search the step is rejected, and
                // when the gradient was asked for the node is left alone,
                // since a zero gradient reads as already optimal.
                if (!(d.sigma > 0.0))
                {
                    m_grad.fill(0.0);

                    if (gradient)
                    {
                        // Only count it where the node is genuinely being
                        // left alone, rather than on a trial position in a
                        // line search, which is refused and moved on from.
                        mtx.lock();
                        m_res->nSkipped++;
                        mtx.unlock();
                    }

                    return std::numeric_limits<double>::max();
                }

                d.frob = FrobeniusNorm<DIM>(d.jacIdeal);

                if (gradient)
                {
                    for (int n = 0; n < DIM; ++n)
                    {
                        d.jacDerivPhi[n] = 0.0;
                        for (int l = 0; l < DIM; ++l)
                        {
                            d.jacDerivPhi[n] += basisDeriv[l] * map[l + 3 * n];
                        }
                    }

                    // d(det A)/dx = det(A) tr(A^-1 dA/dx), rescaled by the
                    // ideal mapping to give the derivative of J rather than
                    // of det(grad phi_M).
                    const NekDouble derivDet = Determinant<DIM>(phiM);
                    NekDouble jacInvTrans[DIM][DIM];
                    InvTrans<DIM>(phiM, jacInvTrans);

                    for (int m = 0; m < DIM; ++m)
                    {
                        d.jacDetDeriv[m] = 0.0;
                        for (int n = 0; n < DIM; ++n)
                        {
                            d.jacDetDeriv[m] +=
                                jacInvTrans[m][n] * basisDeriv[n];
                        }
                        d.jacDetDeriv[m] *= derivDet / idealMapDet;
                    }

                    for (int m = 0; m < DIM; ++m)
                    {
                        d.frobProd[m] =
                            ScalarProd<DIM>(d.jacIdeal[m], d.jacDerivPhi);
                    }
                    d.frobProdHes =
                        ScalarProd<DIM>(d.jacDerivPhi, d.jacDerivPhi);
                }

                const NekDouble W =
                    Energy::template Evaluate<DIM>(d, gradient, dW, d2W);

                // The quadrature weight is applied once, to the integrand and
                // to each of its derivatives alike.
                const NekDouble wq = quadW[k] * absIdealMapDet;

                integral += wq * W;

                if (gradient)
                {
                    for (int m = 0; m < DIM; ++m)
                    {
                        m_grad[m] += wq * dW[m];
                    }
                    for (int ct = 0; ct < nHess; ++ct)
                    {
                        m_grad[ct + DIM] += wq * d2W[ct];
                    }
                }
            }
        }
    }

    return integral;
}

} // namespace Nektar::NekMesh

#endif
