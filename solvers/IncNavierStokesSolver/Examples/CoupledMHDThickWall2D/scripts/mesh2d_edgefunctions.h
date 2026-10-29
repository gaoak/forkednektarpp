#ifndef COUPLEDMHD_EDGEFUNCTIONS_H
#define COUPLEDMHD_EDGEFUNCTIONS_H

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

// Adaptation of Mesh2D's square branch. Coordinates are element boundaries.
// Keep fluid/solid interface coordinates 0 and 1 in y_nodes.
const std::array<double, 3> x_nodes = {{0.0, 0.5, 1.0}};
const std::array<double, 5> y_nodes = {{-0.25, 0.0, 0.5, 1.0, 1.25}};
const int Nx = static_cast<int>(x_nodes.size()) - 1;
const int Ny = static_cast<int>(y_nodes.size()) - 1;

template <std::size_t N>
double coordinateFromNodes(const std::array<double, N> &nodes, double s)
{
    const double index =
        0.5 * (std::max(-1.0, std::min(1.0, s)) + 1.0) * (N - 1);
    const std::size_t lower = static_cast<std::size_t>(std::floor(index));
    if (lower >= N - 1)
    {
        return nodes.back();
    }
    const double weight = index - lower;
    return (1.0 - weight) * nodes[lower] + weight * nodes[lower + 1];
}

inline std::vector<double> edge03(double s)
{
    return {coordinateFromNodes(x_nodes, s), y_nodes.front()};
}
inline std::vector<double> edge69(double s)
{
    return {coordinateFromNodes(x_nodes, s), y_nodes.back()};
}
inline std::vector<double> edge06(double s)
{
    return {x_nodes.front(), coordinateFromNodes(y_nodes, s)};
}
inline std::vector<double> edge39(double s)
{
    return {x_nodes.back(), coordinateFromNodes(y_nodes, s)};
}
#endif
