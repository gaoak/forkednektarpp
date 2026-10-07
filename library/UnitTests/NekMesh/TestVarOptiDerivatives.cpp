///////////////////////////////////////////////////////////////////////////////
//
// File: TestVarOptiDerivatives.cpp
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
// Description: Tests that the analytic gradient and Hessian of the variational
// optimiser's energy functionals agree with the functional they differentiate.
//
///////////////////////////////////////////////////////////////////////////////

#include <LibUtilities/BasicUtils/Log.hpp>
#include <NekMesh/MeshElements/Element.h>
#include <NekMesh/MeshElements/Mesh.h>
#include <NekMesh/Module/ProcessModules/ProcessVarOpti/ElUtil.h>
#include <NekMesh/Module/ProcessModules/ProcessVarOpti/Evaluator.hxx>
#include <NekMesh/Module/ProcessModules/ProcessVarOpti/NodeOpti.h>
#include <NekMesh/Module/ProcessModules/ProcessVarOpti/ProcessVarOpti.h>

#include <boost/test/unit_test.hpp>

#include <array>
#include <random>
#include <set>
#include <string>
#include <vector>

namespace Nektar::NekMeshVarOptiDerivativeUnitTest
{
using namespace Nektar::NekMesh;

/**
 * @brief The functional, its gradient and its Hessian have to describe the
 * same function.
 *
 * NodeOpti::GetFunctional() returns the energy of the elements around a node
 * and leaves its analytic first and second derivatives with respect to that
 * node's position in m_grad. The optimiser trusts both: the Hessian picks the
 * Newton direction and the gradient decides whether the step that results is
 * downhill. Neither is checked anywhere else, and an error in either is close
 * to invisible from the outside -- the mesh still improves, just more slowly
 * and less reliably, and the module reports nothing unusual.
 *
 * So here the derivatives are differenced against the functional the same
 * code evaluates. Three separate errors were found this way, each in one
 * functional only: the Hessian of the distortion and Winslow energies carried
 * the quadrature weight three times over rather than once, and EMatrix<3> --
 * used by the linear elastic energy, and by nothing else -- had one wrong
 * term, which vanishes for an undeformed element and so left the derivatives
 * disagreeing with the functional only once the mesh was curved.
 *
 * Each case is run on a valid element and again on a tangled one, since the
 * Jacobian regularisation that untangling depends on only takes effect when
 * the Jacobian is negative, and is exactly where the derivatives are hardest.
 *
 * A triangle and a tetrahedron cover both dimensions and every branch of
 * GetFunctional(). The remaining shapes differ only in the ideal mapping,
 * which varies across the element for a quadrilateral and a prism rather than
 * being constant; those two are covered end to end by the varopti tests in
 * utilities/NekMesh/Tests instead, as a single quadrilateral or prism built
 * here through CreateElementLite and Mesh::MakeOrder does not come out with a
 * sound node list, which is a separate problem from the one under test.
 */

namespace
{

/// A single element, with the mesh that owns it.
struct TestElement
{
    MeshSharedPtr mesh;
    SpatialDomains::Geometry *el;
    std::vector<SpatialDomains::PointGeom *> nodes;
};

/// Reproducible pseudo-random numbers in [-0.5, 0.5]. std::mt19937 is
/// specified exactly by the standard; the distributions over it are not.
class Jitter
{
public:
    Jitter(unsigned seed) : m_rng(seed)
    {
    }

    NekDouble operator()()
    {
        return NekDouble(m_rng()) / NekDouble(std::mt19937::max()) - 0.5;
    }

private:
    std::mt19937 m_rng;
};

/**
 * @brief What state the element under test is put in.
 */
enum class Config
{
    /// Valid, and curved enough that nothing is symmetric.
    eValid,
    /// A node displaced far enough to invert the element, so that the
    /// Jacobian regularisation is doing the work.
    eTangled,
    /// The straight-sided element itself inverted, which leaves the
    /// determinant of the ideal mapping negative.
    eInverted
};

std::vector<std::vector<NekDouble>> VertexPositions(
    LibUtilities::ShapeType shape)
{
    switch (shape)
    {
        case LibUtilities::eTriangle:
            return {{0.0, 0.0, 0.0}, {1.0, 0.1, 0.0}, {0.2, 1.0, 0.0}};
        case LibUtilities::eTetrahedron:
            return {{0.0, 0.0, 0.0},
                    {1.0, 0.1, 0.05},
                    {0.1, 1.0, -0.05},
                    {0.2, 0.15, 1.0}};
        default:
            BOOST_FAIL("unsupported shape");
            return {};
    }
}

/**
 * @brief Build a mesh of one curved element of @p shape at order
 * @p nummode - 1, with every node jittered so that no symmetry of the element
 * can hide an error in a cross derivative.
 */
TestElement MakeElement(LibUtilities::ShapeType shape, int nummode,
                        unsigned seed, bool invert)
{
    const int dim = (shape == LibUtilities::eTriangle) ? 2 : 3;

    TestElement ret;
    ret.mesh    = std::make_shared<Mesh>();
    auto &graph = ret.mesh->m_meshGraph;
    graph->SetMeshDimension(dim);
    graph->SetSpaceDimension(dim);

    std::vector<std::vector<NekDouble>> verts = VertexPositions(shape);

    // Exchanging two vertices turns the element inside out, which is how a
    // mesh comes to hold a straight-sided element of negative volume.
    if (invert)
    {
        std::swap(verts[1], verts[2]);
    }
    std::vector<SpatialDomains::PointGeom *> nodeList(verts.size());
    for (size_t i = 0; i < verts.size(); ++i)
    {
        nodeList[i] = graph->CreatePointGeom(dim, i, verts[i][0], verts[i][1],
                                             verts[i][2]);
    }

    ret.el = CreateElementLite(shape, nodeList, graph, ret.mesh->m_edgeSet,
                               ret.mesh->m_faceSet);
    ret.mesh->m_elementTags[dim][ret.el] = {0};

    Logger log;
    ret.mesh->MakeOrder(nummode - 1, LibUtilities::eGaussLobattoLegendre, log);

    ret.nodes = GetCurvedNodes(ret.el);

    Jitter jitter(seed);
    for (auto *n : ret.nodes)
    {
        for (int d = 0; d < dim; ++d)
        {
            (*n)[d] += 0.08 * jitter();
        }
    }

    return ret;
}

/// The packed gradient and Hessian the optimiser works with, unpacked.
struct Derivatives
{
    std::vector<NekDouble> grad;
    std::vector<std::vector<NekDouble>> hess;
};

template <int DIM> Derivatives Unpack(const std::array<NekDouble, 9> &g)
{
    Derivatives ret;
    ret.grad.assign(g.begin(), g.begin() + DIM);
    ret.hess.assign(DIM, std::vector<NekDouble>(DIM, 0.0));

    int ct = DIM;
    for (int m = 0; m < DIM; ++m)
    {
        for (int l = m; l < DIM; ++l, ++ct)
        {
            ret.hess[m][l] = ret.hess[l][m] = g[ct];
        }
    }
    return ret;
}

NekDouble MaxAbs(const std::vector<NekDouble> &v)
{
    NekDouble ret = 0.0;
    for (auto x : v)
    {
        ret = std::max(ret, std::fabs(x));
    }
    return ret;
}

/**
 * @brief Difference the functional around @p node and compare with the
 * analytic derivatives the same evaluation produced.
 *
 */
template <int DIM>
void CheckDerivatives(NodeOptiSharedPtr opti, SpatialDomains::PointGeom *node,
                      const std::string &what)
{
    NekDouble minJac;
    opti->GetFunctional<DIM>(minJac, true);
    Derivatives analytic = Unpack<DIM>(opti->GetGrad());

    std::vector<NekDouble> x0(DIM);
    for (int d = 0; d < DIM; ++d)
    {
        x0[d] = (*node)[d];
    }

    // Central differences at a step h, leaving the node where it was found.
    auto difference = [&](NekDouble h) {
        auto eval = [&](std::vector<int> steps) {
            NekDouble offset[3] = {};
            for (int d = 0; d < DIM; ++d)
            {
                offset[d] = steps[d] * h;
            }
            NekDouble dummy;
            return opti->GetFunctionalAt<DIM>(offset, dummy);
        };

        Derivatives ret;
        ret.grad.assign(DIM, 0.0);
        ret.hess.assign(DIM, std::vector<NekDouble>(DIM, 0.0));

        for (int i = 0; i < DIM; ++i)
        {
            std::vector<int> fwd(DIM, 0), bwd(DIM, 0);
            fwd[i]      = 1;
            bwd[i]      = -1;
            ret.grad[i] = (eval(fwd) - eval(bwd)) / (2.0 * h);
        }

        for (int i = 0; i < DIM; ++i)
        {
            for (int j = 0; j < DIM; ++j)
            {
                NekDouble sum = 0.0;
                for (int si : {1, -1})
                {
                    for (int sj : {1, -1})
                    {
                        std::vector<int> steps(DIM, 0);
                        steps[i] += si;
                        steps[j] += sj;
                        sum += si * sj * eval(steps);
                    }
                }
                ret.hess[i][j] = sum / (4.0 * h * h);
            }
        }
        return ret;
    };

    // Both differences carry an error in h^2, so Richardson extrapolation of
    // the two leaves one in h^4. Without it the regularised functional of a
    // tangled element -- which is where this test is most worth having -- is
    // curved enough that the difference itself is only good to a few parts in
    // a thousand, and the tolerance would have to be loose enough to let a
    // real error through.
    const NekDouble h = 2e-4;
    Derivatives d1    = difference(h);
    Derivatives d2    = difference(0.5 * h);

    Derivatives fd;
    fd.grad.assign(DIM, 0.0);
    fd.hess.assign(DIM, std::vector<NekDouble>(DIM, 0.0));
    for (int i = 0; i < DIM; ++i)
    {
        fd.grad[i] = (4.0 * d2.grad[i] - d1.grad[i]) / 3.0;
        for (int j = 0; j < DIM; ++j)
        {
            fd.hess[i][j] = (4.0 * d2.hess[i][j] - d1.hess[i][j]) / 3.0;
        }
    }

    const NekDouble gScale = std::max(MaxAbs(fd.grad), 1e-12);
    for (int i = 0; i < DIM; ++i)
    {
        BOOST_CHECK_MESSAGE(
            std::fabs(analytic.grad[i] - fd.grad[i]) < 1e-6 * gScale,
            what << ": gradient component " << i << " is " << analytic.grad[i]
                 << ", differencing the functional gives " << fd.grad[i]);
    }

    NekDouble hScale = 1e-12;
    for (int i = 0; i < DIM; ++i)
    {
        hScale = std::max(hScale, MaxAbs(fd.hess[i]));
    }

    for (int i = 0; i < DIM; ++i)
    {
        for (int j = 0; j < DIM; ++j)
        {
            BOOST_CHECK_MESSAGE(std::fabs(analytic.hess[i][j] - fd.hess[i][j]) <
                                    1e-4 * hScale,
                                what << ": Hessian entry (" << i << "," << j
                                     << ") is " << analytic.hess[i][j]
                                     << ", differencing the functional gives "
                                     << fd.hess[i][j]);
        }
    }
}

/**
 * @brief Set up one element, one node of it, and the optimiser object that
 * owns the functional, then hand it to CheckDerivatives().
 *
 * @param tangle  Displace the node far enough to invert the element first, so
 *                that the Jacobian regularisation is active.
 */
template <int DIM>
void RunCase(LibUtilities::ShapeType shape, optiType opti,
             const std::string &optiName, Config config)
{
    const int nummode = 4;
    const int overInt = 2;

    TestElement te =
        MakeElement(shape, nummode, 20251005u, config == Config::eInverted);

    std::map<LibUtilities::ShapeType, DerivUtilSharedPtr> derivUtils =
        BuildDerivUtil(nummode, overInt);
    BOOST_REQUIRE(derivUtils.count(shape) == 1);

    ResidualSharedPtr res = std::make_shared<Residual>();
    res->val              = 1.0;
    res->startInv         = 0;
    res->worstJac         = std::numeric_limits<NekDouble>::max();

    ElUtilSharedPtr elUtil = std::make_shared<ElUtil>(te.el, derivUtils[shape],
                                                      res, nummode, overInt);

    // Move a node that is not one of the element's vertices, so that the
    // straight-sided element the functional measures deformation against --
    // which ElUtil builds from the vertices once, and does not revisit -- is
    // genuinely independent of the node being differentiated.
    std::set<SpatialDomains::PointGeom *> verts;
    for (int i = 0; i < te.el->GetNumVerts(); ++i)
    {
        verts.insert(te.el->GetVertex(i));
    }

    SpatialDomains::PointGeom *node = nullptr;
    for (auto *n : te.nodes)
    {
        if (verts.count(n) == 0)
        {
            node = n;
        }
    }
    BOOST_REQUIRE(node != nullptr);

    if (config == Config::eTangled)
    {
        (*node)[0] -= 0.9;
        (*node)[1] -= 0.9;
    }

    // The regularisation parameter is set from the smallest Jacobian in the
    // mesh, which CalcMinJac() reduces into the residual. It stays fixed
    // across the differences below, so the functional really is the smooth
    // function of the node position that the analytic expressions claim to
    // differentiate.
    res->minJac = std::numeric_limits<NekDouble>::max();
    elUtil->CalcDeriv();
    elUtil->CalcMinJac();

    std::vector<ElUtilSharedPtr> els{elUtil};
    NodeOptiSharedPtr nodeOpti = GetNodeOptiFactory().CreateInstance(
        DIM * 11, node, els, res, derivUtils, opti, te.mesh->m_meshGraph.get());

    const char *configName = config == Config::eValid     ? " (valid)"
                             : config == Config::eTangled ? " (tangled)"
                                                          : " (inverted)";
    std::string what =
        optiName + " " + LibUtilities::ShapeTypeMap[shape] + configName;

    if (config == Config::eTangled)
    {
        NekDouble minJac;
        nodeOpti->GetFunctional<DIM>(minJac, false);
        BOOST_REQUIRE_MESSAGE(minJac < 0.0,
                              what + ": element was meant to be tangled");
    }

    if (config == Config::eInverted)
    {
        BOOST_REQUIRE_MESSAGE(
            elUtil->maps[9] < 0.0,
            what + ": the ideal mapping was meant to be inverted");
    }

    CheckDerivatives<DIM>(nodeOpti, node, what);
}

const std::vector<std::pair<optiType, std::string>> functionals = {
    {eLinEl, "linearelastic"},
    {eHypEl, "hyperelastic"},
    {eRoca, "distortion"},
    {eWins, "winslow"}};

} // namespace

BOOST_AUTO_TEST_CASE(TestDerivatives2D)
{
    for (auto &[opti, name] : functionals)
    {
        {
            const LibUtilities::ShapeType shape = LibUtilities::eTriangle;
            for (Config config :
                 {Config::eValid, Config::eTangled, Config::eInverted})
            {
                RunCase<2>(shape, opti, name, config);
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(TestDerivatives3D)
{
    for (auto &[opti, name] : functionals)
    {
        {
            const LibUtilities::ShapeType shape = LibUtilities::eTetrahedron;
            for (Config config :
                 {Config::eValid, Config::eTangled, Config::eInverted})
            {
                RunCase<3>(shape, opti, name, config);
            }
        }
    }
}

/**
 * @brief Moving a node has to leave the derivatives its elements hold equal
 * to what differentiating the moved mesh would give.
 *
 * Each element keeps the derivatives of its mapping at the integration
 * points, worked out once per iteration, and the optimiser carries a node's
 * movement through to them with a rank-one update rather than recomputing
 * them. That update is what the functional is then evaluated from, so if it
 * and the full computation ever disagree the optimiser is minimising
 * something other than the energy of the mesh it is holding -- and it would
 * not show, because the next iteration recomputes them from scratch and
 * quietly papers over it.
 */
BOOST_AUTO_TEST_CASE(TestMovedNodeDerivatives)
{
    const int nummode = 4;
    const int overInt = 2;

    for (auto shape : {LibUtilities::eTriangle, LibUtilities::eTetrahedron})
    {
        const int dim = (shape == LibUtilities::eTriangle) ? 2 : 3;

        TestElement te = MakeElement(shape, nummode, 7u, false);

        std::map<LibUtilities::ShapeType, DerivUtilSharedPtr> derivUtils =
            BuildDerivUtil(nummode, overInt);

        ResidualSharedPtr res = std::make_shared<Residual>();
        res->worstJac         = std::numeric_limits<NekDouble>::max();
        res->minJac           = std::numeric_limits<NekDouble>::max();

        ElUtilSharedPtr elUtil = std::make_shared<ElUtil>(
            te.el, derivUtils[shape], res, nummode, overInt);

        elUtil->CalcDeriv();

        // Move a node, and bring the stored derivatives along with it.
        SpatialDomains::PointGeom *node = te.nodes.back();
        const NekDouble offset[3]       = {0.031, -0.047, 0.013};

        elUtil->MoveNode(elUtil->NodeId(node), offset, dim);
        for (int d = 0; d < dim; ++d)
        {
            (*node)[d] += offset[d];
        }

        std::vector<NekDouble> updated = elUtil->deriv;

        // Now work them out again from the moved mesh.
        elUtil->CalcDeriv();

        BOOST_REQUIRE_EQUAL(updated.size(), elUtil->deriv.size());

        NekDouble scale = 0.0, worst = 0.0;
        for (size_t i = 0; i < updated.size(); ++i)
        {
            scale = std::max(scale, std::fabs(elUtil->deriv[i]));
            worst = std::max(worst, std::fabs(updated[i] - elUtil->deriv[i]));
        }

        BOOST_CHECK_MESSAGE(worst < 1e-12 * std::max(scale, 1e-12),
                            LibUtilities::ShapeTypeMap[shape]
                                << ": the rank-one update left the "
                                << "derivatives differing by " << worst);
    }
}

/**
 * @brief The Newton direction has to solve H sk = -G.
 *
 * NodeOpti3D3D::Optimise() inverts the 3x3 Hessian through its cofactors.
 * Two of the nine were wrong, which left the search direction unrelated to
 * the Newton direction; on the badly conditioned Hessians that the
 * regularisation produces while untangling, roughly a third of the directions
 * that resulted pointed uphill, and the Wolfe condition accepts an uphill
 * step whenever the direction it is handed is not a descent direction. The
 * mesh still improved, so nothing about the result said that the optimiser
 * was no longer doing Newton's method.
 *
 * The Hessians used here have the eigenvalue spread the regularisation leaves
 * behind, one of them sitting at the floor of 1e-6.
 */
BOOST_AUTO_TEST_CASE(TestNewtonDirection)
{
    Jitter jitter(99u);

    for (int t = 0; t < 500; ++t)
    {
        NekDouble a[3][3];
        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                a[i][j] = jitter();
            }
        }

        NekDouble H[3][3];
        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                H[i][j] = (i == j) ? 1e-6 : 0.0;
                for (int k = 0; k < 3; ++k)
                {
                    H[i][j] += a[i][k] * a[j][k];
                }
            }
        }

        std::vector<NekDouble> grad3 = {jitter(), jitter(), jitter(),
                                        H[0][0],  H[0][1],  H[0][2],
                                        H[1][1],  H[1][2],  H[2][2]};

        NekDouble sk3[3];
        NewtonDirection<3>(grad3.data(), sk3);

        NekDouble normH = 0.0, normSk = 0.0;
        for (int i = 0; i < 3; ++i)
        {
            normSk = std::max(normSk, std::fabs(sk3[i]));
            for (int j = 0; j < 3; ++j)
            {
                normH = std::max(normH, std::fabs(H[i][j]));
            }
        }

        for (int i = 0; i < 3; ++i)
        {
            NekDouble Hsk = 0.0;
            for (int j = 0; j < 3; ++j)
            {
                Hsk += H[i][j] * sk3[j];
            }
            BOOST_CHECK_SMALL(Hsk + grad3[i], 1e-9 * normH * normSk);
        }

        BOOST_CHECK_LT(
            grad3[0] * sk3[0] + grad3[1] * sk3[1] + grad3[2] * sk3[2], 0.0);

        // The same again in two dimensions, on the leading 2x2 block.
        std::array<NekDouble, 5> grad2 = {grad3[0], grad3[1], H[0][0], H[0][1],
                                          H[1][1]};
        NekDouble sk2[2];
        NewtonDirection<2>(grad2.data(), sk2);

        for (int i = 0; i < 2; ++i)
        {
            NekDouble Hsk = 0.0;
            for (int j = 0; j < 2; ++j)
            {
                Hsk += H[i][j] * sk2[j];
            }
            BOOST_CHECK_SMALL(
                Hsk + grad2[i],
                1e-9 * normH * std::max(std::fabs(sk2[0]), std::fabs(sk2[1])));
        }

        BOOST_CHECK_LT(grad2[0] * sk2[0] + grad2[1] * sk2[1], 0.0);
    }
}

} // namespace Nektar::NekMeshVarOptiDerivativeUnitTest
