#pragma once

#include <Eigen/Dense>
#include <optional>
#include <vector>
#include "common.h"

namespace vr_tokenizer::cgal
{

    Stats edge_collapse_with_record(
        const Eigen::ArrayX3d &vertices,
        const Eigen::ArrayX3i &faces,
        std::size_t target_number_of_vertices,
        std::size_t target_number_of_triangles,
        bool no_placement = false,
        double sharp_angle_threshold = -1,
        bool strict = false,
        bool record_full_info = false,
        // Indices into `vertices` that must survive the simplification.
        // Empty (the default) reproduces upstream behaviour exactly.
        const std::vector<int> &pinned_vertices = {});

    std::optional<PolygonSoup> vertex_split(
        const Eigen::ArrayX3d &vertices,
        const Eigen::ArrayX3i &faces,
        std::size_t v_s,
        std::optional<std::size_t> v_l,
        std::optional<std::size_t> v_r,
        const Eigen::Vector3d &v_t);

} // namespace vr_tokenizer::cgal
