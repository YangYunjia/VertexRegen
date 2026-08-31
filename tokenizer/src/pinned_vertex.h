#pragma once

#include <CGAL/Unique_hash_map.h>
#include <optional>
#include "common.h"

namespace vr_tokenizer::cgal
{

    // Vertices that must survive the simplification.
    //
    // Note this is deliberately NOT expressed with `edge_is_constrained_map`.
    // CGAL rejects a collapse when *both* endpoints touch a constrained edge
    // (Edge_collapse.h:823, via is_constrained(vertex) at :687), so constraining
    // the incident edges of a vertex freezes its entire one-ring: the mesh floors
    // at the union of the pinned stars (measured: 21-25 verts on our brackets)
    // and keeps the fine fan geometry around each pin. Pinning the *vertices* and
    // vetoing only pin-to-pin collapses lets a pinned vertex keep absorbing its
    // unpinned neighbours, which is what drives the mesh down to the seed.
    using Pin_map = CGAL::Unique_hash_map<vertex_descriptor, bool>;

    // Veto a collapse whose two endpoints are both pinned; allow everything else.
    //
    // The filter runs after the placement is computed and its return value is only
    // tested against nullopt (Edge_collapse.h:633) -- it can reject a collapse but
    // it cannot override where the survivor lands. That is why the placement below
    // exists separately.
    struct Pinned_vertex_filter
    {
        const Pin_map *pins = nullptr;

        template <typename Profile>
        std::optional<typename Profile::Point>
        operator()(const Profile &profile, const std::optional<typename Profile::Point> &op) const
        {
            if (op && pins != nullptr &&
                pins->is_defined(profile.v0()) && pins->is_defined(profile.v1()))
            {
                return std::optional<typename Profile::Point>();
            }
            return op;
        }
    };

    // Keep a pinned endpoint exactly where it is.
    //
    // Euler::collapse_edge always removes `source` and returns `target`, and SMS
    // never flips the halfedge to protect a particular vertex, so a pinned vertex
    // can still be the one erased. That is harmless here because the tokenizer
    // identifies vertices by their quantized position (collapse.py builds
    // `vertex_map = {tuple(v): i}`), not by descriptor -- preserving the *point*
    // is what keeps the pin addressable downstream.
    //
    // Modelled on CGAL's Constrained_placement, which does the same thing keyed on
    // edges instead of vertices.
    template <class BasePlacement>
    class Pinned_vertex_placement : public BasePlacement
    {
    public:
        Pinned_vertex_placement(const Pin_map *pins = nullptr,
                                const BasePlacement &base = BasePlacement())
            : BasePlacement(base), m_pins(pins)
        {
        }

        template <typename Profile>
        std::optional<typename Profile::Point> operator()(const Profile &profile) const
        {
            if (m_pins != nullptr)
            {
                const bool p0 = m_pins->is_defined(profile.v0());
                const bool p1 = m_pins->is_defined(profile.v1());
                if (p0 && !p1)
                    return get(profile.vertex_point_map(), profile.v0());
                if (p1 && !p0)
                    return get(profile.vertex_point_map(), profile.v1());
            }
            return static_cast<const BasePlacement *>(this)->operator()(profile);
        }

    private:
        const Pin_map *m_pins;
    };

} // namespace vr_tokenizer::cgal
