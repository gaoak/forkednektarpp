#include "ErrorUtil.hpp"

namespace Nektar
{
    std::ostream *ErrorUtil::m_outStream = &std::cerr;
    bool ErrorUtil::m_printBacktrace     = true;
}