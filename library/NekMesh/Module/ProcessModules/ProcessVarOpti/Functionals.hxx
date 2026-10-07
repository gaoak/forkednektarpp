////////////////////////////////////////////////////////////////////////////////
//
//  File: Functionals.hxx
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
//  Description: The energy functionals the variational optimiser minimises.
//
////////////////////////////////////////////////////////////////////////////////

#ifndef UTILITIES_NEKMESH_NODEOPTI_FUNCTIONALS
#define UTILITIES_NEKMESH_NODEOPTI_FUNCTIONALS

#include "Evaluator.hxx"

#include <cmath>

namespace Nektar::NekMesh
{

/**
 * @brief The deformation at one quadrature point of one element, and the
 * quantities every energy functional is built from.
 *
 * The four functionals differ only in how they combine these; the work of
 * producing them is the same for all of them and belongs to the caller. The
 * derivatives are with respect to the position of the single node being
 * optimised, and are not Cartesian derivatives: because
 * \f$ \partial (\nabla\phi_M)_{ij} / \partial x^n_m = \delta_{im}
 * \partial_j \ell_n \f$ has only one non-zero row, several quantities that
 * are tensors in general collapse to vectors or scalars here.
 */
template <int DIM> struct Deformation
{
    /// \f$ \nabla\phi \f$, the deformation from the ideal element.
    NekDouble jacIdeal[DIM][DIM];
    /// \f$ J = \det\nabla\phi \f$, negative where the element is inverted.
    NekDouble jacDet = 0.0;
    /// The regularised Jacobian \f$ J_R \f$, which stays positive where J
    /// does not and so keeps \f$ \ln J \f$ and \f$ 1/J \f$ defined while
    /// untangling.
    NekDouble sigma = 0.0;
    /// \f$ 2J_R - J = \sqrt{4\delta^2 + J^2} \f$. Every derivative of
    /// \f$ J_R \f$ is \f$ \partial J \f$ over this.
    NekDouble twoSigmaMinusJ = 0.0;
    /// \f$ \|\nabla\phi\|_f^2 \f$.
    NekDouble frob = 0.0;

    // Set only when derivatives are wanted.

    /// \f$ \partial(\nabla\phi_M)/\partial x^n_m \nabla\phi_I^{-1} \f$, which
    /// the delta function above reduces to one vector rather than one tensor
    /// per direction.
    NekDouble jacDerivPhi[DIM] = {};
    /// \f$ \partial J / \partial x^n_m \f$.
    NekDouble jacDetDeriv[DIM] = {};
    /// Half of \f$ \partial \|\nabla\phi\|_f^2 / \partial x^n_m \f$.
    NekDouble frobProd[DIM] = {};
    /// The diagonal of the second derivative of \f$ \|\nabla\phi\|_f^2 \f$,
    /// which is zero off the diagonal for the same reason.
    NekDouble frobProdHes = 0.0;
};

/// The number of distinct entries of a symmetric DIM x DIM Hessian.
constexpr int HessianSize(int DIM)
{
    return DIM * (DIM + 1) / 2;
}

/**
 * @brief The Lame constants the two elasticity functionals use.
 *
 * A constant Young's modulus only scales the energy, so the functionals
 * depend on Poisson's ratio alone.
 */
struct Lame
{
    static constexpr NekDouble nu = 0.49;
    static constexpr NekDouble mu = 1.0 / 2.0 / (1.0 + nu);
    static constexpr NekDouble K  = 1.0 / 3.0 / (1.0 - 2.0 * nu);
};

/**
 * @brief Linear elasticity,
 * \f$ W = \frac{\kappa}{2}(\ln J_R)^2
 *         + \mu \|\tfrac{1}{2}(\nabla\phi^\top\nabla\phi - I)\|_f^2 \f$.
 *
 * The strain is the Green-Lagrange one rather than
 * \f$ \tfrac{1}{2}(\nabla u + \nabla u^\top) \f$: the logarithmic bulk term
 * replaces \f$ \tfrac{\lambda}{2}(\operatorname{tr} E)^2 \f$ so that
 * \f$ W \to \infty \f$ as \f$ J \to 0^+ \f$, which is what stops the mesh
 * inverting.
 */
struct LinearElasticEnergy
{
    template <int DIM>
    static NekDouble Evaluate(const Deformation<DIM> &d, bool gradient,
                              NekDouble (&dW)[DIM],
                              NekDouble (&d2W)[HessianSize(DIM)])
    {
        NekDouble emat[DIM][DIM];
        EMatrix<DIM>(d.jacIdeal, emat);

        const NekDouble lsigma = log(d.sigma);
        const NekDouble W      = Lame::K * 0.5 * lsigma * lsigma +
                            Lame::mu * FrobProd<DIM>(emat, emat);

        if (!gradient)
        {
            return W;
        }

        // The derivative of the strain with respect to each direction. As
        // with jacDerivPhi this would be a rank-three tensor were it not for
        // the delta function.
        NekDouble M2[DIM][DIM][DIM];
        for (int m = 0; m < DIM; ++m)
        {
            for (int p = 0; p < DIM; ++p)
            {
                for (int q = 0; q < DIM; ++q)
                {
                    M2[m][p][q] = 0.5 * (d.jacDerivPhi[p] * d.jacIdeal[m][q] +
                                         d.jacIdeal[m][p] * d.jacDerivPhi[q]);
                }
            }
        }

        for (int m = 0; m < DIM; ++m)
        {
            dW[m] = 2.0 * Lame::mu * FrobProd<DIM>(M2[m], emat) +
                    Lame::K * lsigma * d.jacDetDeriv[m] / d.twoSigmaMinusJ;
        }

        int ct = 0;
        for (int m = 0; m < DIM; ++m)
        {
            for (int l = m; l < DIM; ++l, ct++)
            {
                NekDouble frobProdBC = FrobProd<DIM>(M2[m], M2[l]);

                if (m == l)
                {
                    NekDouble M3[DIM][DIM];
                    for (int p = 0; p < DIM; ++p)
                    {
                        for (int q = 0; q < DIM; ++q)
                        {
                            M3[p][q] = d.jacDerivPhi[p] * d.jacDerivPhi[q];
                        }
                    }
                    frobProdBC += FrobProd<DIM>(M3, emat);
                }

                d2W[ct] = 2.0 * Lame::mu * frobProdBC +
                          d.jacDetDeriv[m] * d.jacDetDeriv[l] * Lame::K /
                              d.twoSigmaMinusJ / d.twoSigmaMinusJ *
                              (1.0 - d.jacDet * lsigma / d.twoSigmaMinusJ);
            }
        }

        return W;
    }
};

/**
 * @brief Compressible neo-Hookean hyperelasticity,
 * \f$ W = \frac{\mu}{2}(\|\nabla\phi\|_f^2 - 3) - \mu\ln J_R
 *         + \frac{\kappa}{2}(\ln J_R)^2 \f$.
 */
struct HyperElasticEnergy
{
    template <int DIM>
    static NekDouble Evaluate(const Deformation<DIM> &d, bool gradient,
                              NekDouble (&dW)[DIM],
                              NekDouble (&d2W)[HessianSize(DIM)])
    {
        const NekDouble lsigma = log(d.sigma);
        const NekDouble W = 0.5 * Lame::mu * (d.frob - 3.0 - 2.0 * lsigma) +
                            0.5 * Lame::K * lsigma * lsigma;

        if (!gradient)
        {
            return W;
        }

        for (int m = 0; m < DIM; ++m)
        {
            dW[m] = Lame::mu * d.frobProd[m] +
                    (d.jacDetDeriv[m] / d.twoSigmaMinusJ *
                     (Lame::K * lsigma - Lame::mu));
        }

        int ct = 0;
        for (int m = 0; m < DIM; ++m)
        {
            for (int l = m; l < DIM; ++l, ct++)
            {
                d2W[ct] =
                    Lame::mu * (m == l ? d.frobProdHes : 0.0) +
                    d.jacDetDeriv[m] * d.jacDetDeriv[l] / d.twoSigmaMinusJ /
                        d.twoSigmaMinusJ *
                        (Lame::K - d.jacDet * (Lame::K * lsigma - Lame::mu) /
                                       d.twoSigmaMinusJ);
            }
        }

        return W;
    }
};

/**
 * @brief The shape distortion measure,
 * \f$ W = \|\nabla\phi\|_f^2 / (n |J_R|^{2/n}) \f$.
 *
 * In two dimensions this and the Winslow energy differ only by a factor of
 * one half wherever J is positive, and so optimise to the same mesh.
 */
struct DistortionEnergy
{
    template <int DIM>
    static NekDouble Evaluate(const Deformation<DIM> &d, bool gradient,
                              NekDouble (&dW)[DIM],
                              NekDouble (&d2W)[HessianSize(DIM)])
    {
        const NekDouble W = d.frob / DIM / pow(fabs(d.sigma), 2.0 / DIM);

        if (!gradient)
        {
            return W;
        }

        for (int m = 0; m < DIM; ++m)
        {
            dW[m] = 2.0 * W *
                    (d.frobProd[m] / d.frob -
                     d.jacDetDeriv[m] / DIM / d.twoSigmaMinusJ);
        }

        int ct = 0;
        for (int m = 0; m < DIM; ++m)
        {
            for (int l = m; l < DIM; ++l, ct++)
            {
                d2W[ct] =
                    dW[m] * dW[l] / W +
                    2.0 * W *
                        ((m == l ? d.frobProdHes : 0.0) / d.frob -
                         2.0 * d.frobProd[m] * d.frobProd[l] / d.frob / d.frob +
                         d.jacDetDeriv[m] * d.jacDetDeriv[l] * d.jacDet /
                             d.twoSigmaMinusJ / d.twoSigmaMinusJ /
                             d.twoSigmaMinusJ / DIM);
            }
        }

        return W;
    }
};

/**
 * @brief The Winslow energy, \f$ W = \|\nabla\phi\|_f^2 / J_R \f$.
 */
struct WinslowEnergy
{
    template <int DIM>
    static NekDouble Evaluate(const Deformation<DIM> &d, bool gradient,
                              NekDouble (&dW)[DIM],
                              NekDouble (&d2W)[HessianSize(DIM)])
    {
        const NekDouble W = d.frob / d.sigma;

        if (!gradient)
        {
            return W;
        }

        for (int m = 0; m < DIM; ++m)
        {
            dW[m] = W * (2.0 * d.frobProd[m] / d.frob -
                         d.jacDetDeriv[m] / d.twoSigmaMinusJ);
        }

        int ct = 0;
        for (int m = 0; m < DIM; ++m)
        {
            for (int l = m; l < DIM; ++l, ct++)
            {
                d2W[ct] =
                    dW[m] * dW[l] / W +
                    2.0 * W *
                        ((m == l ? d.frobProdHes : 0.0) / d.frob -
                         2.0 * d.frobProd[m] * d.frobProd[l] / d.frob / d.frob +
                         0.5 * d.jacDetDeriv[m] * d.jacDetDeriv[l] * d.jacDet /
                             d.twoSigmaMinusJ / d.twoSigmaMinusJ /
                             d.twoSigmaMinusJ);
            }
        }

        return W;
    }
};

} // namespace Nektar::NekMesh

#endif
