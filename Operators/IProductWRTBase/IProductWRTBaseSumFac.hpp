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
        : OperatorIProductWRTBase<TData>(std::move(expansionList))
    {
        // Initialise jacobian.
        size_t jacSize = Operator<TData>::GetGeometricFactorSize();
        m_jac          = Operator<TData>::SetJacobian(jacSize);
    }

    void apply(Field<TData, FieldState::Phys> &in,
               Field<TData, FieldState::Coeff> &out,
               const TData lambda = 1.0) override
    {
        boost::ignore_unused(lambda);
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
            auto nqTot = expPtr->GetTotPoints();
            auto nmTot = expPtr->GetNcoeffs();

            switch (shapeType)
            {
                // Segment
                case LibUtilities::Seg:
                    std::cout << "Sumfac-Seg" << std::endl;
                    IProductWRTBaseSumFacSegKernel(inptr, outptr, expPtr, m_jac,
                                                   numElmts, jac_idx);
                    break;
                // Quads
                case LibUtilities::Quad:
                    std::cout << "Sumfac-Quad" << std::endl;
                    IProductWRTBaseSumFacQuadKernel(inptr, outptr, expPtr,
                                                    m_jac, numElmts, jac_idx);
                    break;
                // Hexes
                case LibUtilities::Hex:
                    std::cout << "Sumfac-Hex" << std::endl;
                    IProductWRTBaseSumFacHexKernel(inptr, outptr, expPtr, m_jac,
                                                   numElmts, jac_idx);
                    break;
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
            std::move(expansionList));
    }

    static std::string className;

private:
    Array<OneD, TData> m_jac;
};

} // namespace Nektar::Operators::detail
