#include "IProductWRTBaseSumFacKernels.hpp"
#include "Operators/OperatorIProductWRTBase.hpp"
#include <StdRegions/StdExpansion.h>

namespace Nektar::Operators::detail
{

// standard matrix implementation
template <typename TData>
class OperatorIProductWRTBaseImpl<TData, ImplSumFac>
    : public OperatorIProductWRTBase<TData>
{
public:
    OperatorIProductWRTBaseImpl(
        const MultiRegions::ExpListSharedPtr &expansionList)
        : OperatorIProductWRTBase<TData>(expansionList)
    {
        // Initialise jacobian.
        size_t jacSize = Operator<TData>::GetGeometricFactorSize();
        m_jac          = Operator<TData>::SetJacobian(jacSize);
    }

    void apply(Field<TData, FieldState::Phys> &in,
               Field<TData, FieldState::Coeff> &out,
               [[maybe_unused]] const TData lambda = 1.0) override
    {
        auto const *inptr = in.GetStorage().GetCPUPtr();
        auto *outptr      = out.GetStorage().GetCPUPtr();

        size_t exp_idx = 0;
        size_t jac_idx = 0;
        for (size_t block_idx = 0; block_idx < in.GetBlocks().size();
             ++block_idx)
        {
            // block dependent
            auto const block    = in.GetBlocks()[block_idx];
            auto const numElmts = block.num_elements;
            auto expPtr         = this->m_expansionList->GetExp(exp_idx);
            auto shapeType      = this->m_expansionList->GetExp(exp_idx)
                                 ->GetStdExp()
                                 ->DetShapeType();

            switch (shapeType)
            {
                // Segment
                case LibUtilities::Seg:
                    IProductWRTBaseSumFacSegKernel(inptr, outptr, expPtr, m_jac,
                                                   numElmts, jac_idx);
                    break;
                // Triangles
                case LibUtilities::Tri:
                    IProductWRTBaseSumFacTriKernel(inptr, outptr, expPtr, m_jac,
                                                   numElmts, jac_idx);
                    break;
                // Quads
                case LibUtilities::Quad:
                    IProductWRTBaseSumFacQuadKernel(inptr, outptr, expPtr,
                                                    m_jac, numElmts, jac_idx);
                    break;
                // Tet
                case LibUtilities::Tet:
                    IProductWRTBaseSumFacTetKernel(inptr, outptr, expPtr, m_jac,
                                                   numElmts, jac_idx);
                    break;
                // Pyr
                case LibUtilities::Pyr:
                    IProductWRTBaseSumFacPyrKernel(inptr, outptr, expPtr, m_jac,
                                                   numElmts, jac_idx);
                    break;
                // Prism
                case LibUtilities::Prism:
                    IProductWRTBaseSumFacPrismKernel(inptr, outptr, expPtr,
                                                     m_jac, numElmts, jac_idx);
                    break;
                // Hexes
                case LibUtilities::Hex:
                    IProductWRTBaseSumFacHexKernel(inptr, outptr, expPtr, m_jac,
                                                   numElmts, jac_idx);
                    break;
                default:
                    std::cout << "shapetype not implemented" << std::endl;

            }

            inptr += in.GetBlocks()[block_idx].block_size;
            outptr += out.GetBlocks()[block_idx].block_size;
            exp_idx += numElmts;
        }
    }

    static std::unique_ptr<Operator<TData>> instantiate(
        const MultiRegions::ExpListSharedPtr &expansionList)
    {
        return std::make_unique<OperatorIProductWRTBaseImpl<TData, ImplSumFac>>(
            expansionList);
    }

    static std::string className;

private:
    Array<OneD, TData> m_jac;
};

} // namespace Nektar::Operators::detail
