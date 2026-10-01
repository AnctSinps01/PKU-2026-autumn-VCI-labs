#include <unordered_map>

#include <glm/gtc/matrix_inverse.hpp>
#include <spdlog/spdlog.h>

#include "Labs/2-GeometryProcessing/DCEL.hpp"
#include "Labs/2-GeometryProcessing/tasks.h"

namespace VCX::Labs::GeometryProcessing {

#include "Labs/2-GeometryProcessing/marching_cubes_table.h"

    /******************* 1. Mesh Subdivision *****************/
    void SubdivisionMesh(Engine::SurfaceMesh const & input, Engine::SurfaceMesh & output, std::uint32_t numIterations) {
        Engine::SurfaceMesh curr_mesh = input;
        // We do subdivison iteratively.
        for (std::uint32_t it = 0; it < numIterations; ++it) {
            // During each iteration, we first move curr_mesh into prev_mesh.
            Engine::SurfaceMesh prev_mesh;
            prev_mesh.Swap(curr_mesh);
            // Then we create doubly connected edge list.
            DCEL G(prev_mesh);
            if (! G.IsManifold()) {
                spdlog::warn("VCX::Labs::GeometryProcessing::SubdivisionMesh(..): Non-manifold mesh.");
                return;
            }
            // Note that here curr_mesh has already been empty.
            // We reserve memory first for efficiency.
            curr_mesh.Positions.reserve(prev_mesh.Positions.size() * 3 / 2);
            curr_mesh.Indices.reserve(prev_mesh.Indices.size() * 4);
            // Then we iteratively update currently existing vertices.
            for (std::size_t i = 0; i < prev_mesh.Positions.size(); ++i) {
                // Update the currently existing vetex v from prev_mesh.Positions.
                // Then add the updated vertex into curr_mesh.Positions.
                auto v           = G.Vertex(i);
                auto neighbors   = v->Neighbors();
                // your code here:
                if (v->OnBoundary()) {
                    auto [nb0, nb1] = v->BoundaryNeighbors();
                    curr_mesh.Positions.emplace_back(
                        0.75f * prev_mesh.Positions[i] +
                        0.125f * (prev_mesh.Positions[nb0] + prev_mesh.Positions[nb1]));
                } else if (neighbors.size() >= 3u) {
                    float n = neighbors.size();
                    float beta = (0.625f - std::pow(0.375f + 0.25f * std::cos(2.f * 3.14159265f / n), 2.f)) / n;
                    glm::vec3 sum(0.f);
                    for (auto j : neighbors) sum += prev_mesh.Positions[j];
                    curr_mesh.Positions.emplace_back((1.f - n * beta) * prev_mesh.Positions[i] + beta * sum);
                } else {
                    curr_mesh.Positions.emplace_back(prev_mesh.Positions[i]);
                }
            }
            // We create an array to store indices of the newly generated vertices.
            // Note: newIndices[i][j] is the index of vertex generated on the "opposite edge" of j-th
            //       vertex in the i-th triangle.
            std::vector<std::array<std::uint32_t, 3U>> newIndices(prev_mesh.Indices.size() / 3, { ~0U, ~0U, ~0U });
            // Iteratively process each halfedge.
            for (auto e : G.Edges()) {
                // newIndices[face index][vertex index] = index of the newly generated vertex
                newIndices[G.IndexOf(e->Face())][e->EdgeLabel()] = curr_mesh.Positions.size();
                auto eTwin                                       = e->TwinEdgeOr(nullptr);
                // eTwin stores the twin halfedge.
                if (! eTwin) {
                    // When there is no twin halfedge (so, e is a boundary edge):
                    // your code here: generate the new vertex and add it into curr_mesh.Positions.
                    // edge
                    curr_mesh.Positions.emplace_back(
                        0.5f * (prev_mesh.Positions[e->From()] + prev_mesh.Positions[e->To()]));
                } else {
                    // When the twin halfedge exists, we should also record:
                    //     newIndices[face index][vertex index] = index of the newly generated vertex
                    // Because G.Edges() will only traverse once for two halfedges,
                    //     we have to record twice.
                    newIndices[G.IndexOf(eTwin->Face())][e->TwinEdge()->EdgeLabel()] = curr_mesh.Positions.size();
                    // your code here: generate the new vertex and add it into curr_mesh.Positions.
                    // inside
                    curr_mesh.Positions.emplace_back(
                        0.375f * (prev_mesh.Positions[e->From()] + prev_mesh.Positions[e->To()]) +
                        0.125f * (prev_mesh.Positions[e->OppositeVertex()] + prev_mesh.Positions[e->TwinOppositeVertex()]));
                }
            }

            // Here we've already build all the vertices.
            // Next, it's time to reconstruct face indices.
            for (std::size_t i = 0; i < prev_mesh.Indices.size(); i += 3U) {
                // For each face F in prev_mesh, we should create 4 sub-faces.
                // v0,v1,v2 are indices of vertices in F.
                // m0,m1,m2 are generated vertices on the edges of F.
                auto v0           = prev_mesh.Indices[i + 0U];
                auto v1           = prev_mesh.Indices[i + 1U];
                auto v2           = prev_mesh.Indices[i + 2U];
                auto [m0, m1, m2] = newIndices[i / 3U];
                // Note: m0 is on the opposite edge (v1-v2) to v0.
                // Please keep the correct indices order (consistent with order v0-v1-v2)
                //     when inserting new face indices.
                // toInsert[i][j] stores the j-th vertex index of the i-th sub-face.
                std::uint32_t toInsert[4][3] = {
                    // your code here:
                    { v0, m2, m1 },
                    { v1, m0, m2 },
                    { v2, m1, m0 },
                    { m0, m1, m2 }
                };
                // Do insertion.
                curr_mesh.Indices.insert(
                    curr_mesh.Indices.end(),
                    reinterpret_cast<std::uint32_t *>(toInsert),
                    reinterpret_cast<std::uint32_t *>(toInsert) + 12U
                );
            }

            if (curr_mesh.Positions.size() == 0) {
                spdlog::warn("VCX::Labs::GeometryProcessing::SubdivisionMesh(..): Empty mesh.");
                output = input;
                return;
            }
        }
        // Update output.
        output.Swap(curr_mesh);
    }

    /******************* 2. Mesh Parameterization *****************/
    void Parameterization(Engine::SurfaceMesh const & input, Engine::SurfaceMesh & output, const std::uint32_t numIterations) {
        // Copy.
        output = input;
        // Reset output.TexCoords.
        output.TexCoords.resize(input.Positions.size(), glm::vec2 { 0 });

        // Build DCEL.
        DCEL G(input);
        if (! G.IsManifold()) {
            spdlog::warn("VCX::Labs::GeometryProcessing::Parameterization(..): non-manifold mesh.");
            return;
        }

        // Set boundary UVs for boundary vertices.
        // your code here: directly edit output.TexCoords
        size_t start = input.Positions.size();
        for (size_t i = 0; i < input.Positions.size(); ++i)
            if (G.Vertex(i)->OnBoundary()) { start = i; break; }

        std::vector<size_t> boundary;
        if (start < input.Positions.size()) {
            size_t prev = input.Positions.size(), curr = start;
            do {
                boundary.emplace_back(curr);
                auto [n0, n1] = G.Vertex(curr)->BoundaryNeighbors();
                size_t next = (n0 == prev) ? n1 : n0;
                prev = curr;
                curr = next;
            } while (curr != start);
        }
        if (!boundary.empty()) {
            size_t B = boundary.size();
            std::vector<float> cum(B +1, 0.f);
            for (size_t i = 0; i < B; ++i)
                cum[i +1] = cum[i] + glm::length(input.Positions[boundary[(i +1) % B]] - input.Positions[boundary[i]]);
            float total = cum.back();
            float invTotal = total > 0.f ? 4.f / total : 0.f;
            for (size_t i = 0; i < B; ++i) {
                float s = cum[i] * invTotal;
                glm::vec2 uv(0.f);
                if (s < 1.f)      uv = { s, 0.f };
                else if (s < 2.f) uv = { 1.f, s - 1.f };
                else if (s < 3.f) uv = { 3.f - s, 1.f };
                else              uv = { 0.f, 4.f - s };
                output.TexCoords[boundary[i]] = uv;
            }
        }

        // Solve equation via Gauss-Seidel Iterative Method.
        for (int k = 0; k < numIterations; ++k) {
            // your code here:
            for (std::size_t i = 0; i < input.Positions.size(); ++i) {
                auto v = G.Vertex(i);
                if (v->OnBoundary()) continue;
                auto neighbors = v->Neighbors();
                glm::vec2 sum { 0.f };
                for (auto j : neighbors) sum += output.TexCoords[j];
                output.TexCoords[i] = sum / (float)(neighbors.size());
            }
        }
    }

    /******************* 3. Mesh Simplification *****************/
    void SimplifyMesh(Engine::SurfaceMesh const & input, Engine::SurfaceMesh & output, float simplification_ratio) {

        DCEL G(input);
        if (! G.IsManifold()) {
            spdlog::warn("VCX::Labs::GeometryProcessing::SimplifyMesh(..): Non-manifold mesh.");
            return;
        }
        // We only allow watertight mesh.
        if (! G.IsWatertight()) {
            spdlog::warn("VCX::Labs::GeometryProcessing::SimplifyMesh(..): Non-watertight mesh.");
            return;
        }

        // Copy.
        output = input;

        // Compute Kp matrix of the face f.
        auto UpdateQ {
            [&G, &output] (DCEL::Triangle const * f) -> glm::mat4 {
                glm::mat4 Kp;
                // your code here:
                glm::vec3& p0 = output.Positions[f->VertexIndex(0)];
                glm::vec3& p1 = output.Positions[f->VertexIndex(1)];
                glm::vec3& p2 = output.Positions[f->VertexIndex(2)];
                glm::vec3 n = glm::cross(p1 - p0, p2 - p0);
                float len = glm::length(n);
                if (len < 1e-12f) return glm::mat4(0.f);
                n /= len;
                float d = -glm::dot(n, p0);
                glm::vec4 p(n, d);
                for (int i = 0; i < 4; ++i)
                    for (int j = 0; j < 4; ++j)
                        Kp[i][j] = p[i] * p[j];
                return Kp;
            }
        };

        // The struct to record contraction info.
        struct ContractionPair {
            DCEL::HalfEdge const * edge;            // which edge to contract; if $edge == nullptr$, it means this pair is no longer valid
            glm::vec4              targetPosition;  // the targetPosition $v$ for vertex $edge->From()$ to move to
            float                  cost;            // the cost $v.T * Qbar * v$
        };

        // Given an edge (v1->v2), the positions of its two endpoints (p1, p2) and the Q matrix (Q1+Q2),
        //     return the ContractionPair struct.
        static constexpr auto MakePair {
            [] (DCEL::HalfEdge const * edge,
                glm::vec3 const & p1,
                glm::vec3 const & p2,
                glm::mat4 const & Q
            ) -> ContractionPair {
                // your code here:
                glm::mat3 A {
                    glm::vec3(Q[0]),
                    glm::vec3(Q[1]),
                    glm::vec3(Q[2])
                };
                glm::vec3 b(Q[3]);
                glm::vec3 target;
                if (glm::abs(glm::determinant(A)) > 1e-3f) {
                    target = glm::inverse(A) * (-b);
                } else {
                    const auto Cost = [&Q] (const glm::vec3& x) -> float {
                        glm::vec4 v { x, 1.f };
                        return glm::dot(v, Q * v);
                    };
                    glm::vec3 const mid = 0.5f * (p1 + p2);
                    float c1 = Cost(p1);
                    float c2 = Cost(p2);
                    float cm = Cost(mid);
                    if (c1 <= c2 && c1 <= cm)      target = p1;
                    else if (c2 <= c1 && c2 <= cm) target = p2;
                    else                           target = mid;
                }
                glm::vec4 v {target, 1.f};
                return ContractionPair {edge, v, glm::dot(v, Q * v)};
            }
        };

        // pair_map: map EdgeIdx to index of $pairs$
        // pairs:    store ContractionPair
        // Qv:       $Qv[idx]$ is the Q matrix of vertex with index $idx$
        // Kf:       $Kf[idx]$ is the Kp matrix of face with index $idx$
        std::unordered_map<DCEL::EdgeIdx, std::size_t> pair_map; 
        std::vector<ContractionPair>                  pairs; 
        std::vector<glm::mat4>                         Qv(G.NumOfVertices(), glm::mat4(0));
        std::vector<glm::mat4>                         Kf(G.NumOfFaces(),    glm::mat4(0));

        // Initially, we compute Q matrix for each faces and it accumulates at each vertex.
        for (auto f : G.Faces()) {
            auto Q                 = UpdateQ(f);
            Qv[f->VertexIndex(0)] += Q;
            Qv[f->VertexIndex(1)] += Q;
            Qv[f->VertexIndex(2)] += Q;
            Kf[G.IndexOf(f)]       = Q;
        }

        pair_map.reserve(G.NumOfFaces() * 3);
        pairs.reserve(G.NumOfFaces() * 3 / 2);

        // Initially, we make pairs from all the contractable edges.
        for (auto e : G.Edges()) {
            if (! G.IsContractable(e)) continue;
            auto v1                            = e->From();
            auto v2                            = e->To();
            auto pair                          = MakePair(e, input.Positions[v1], input.Positions[v2], Qv[v1] + Qv[v2]);
            pair_map[G.IndexOf(e)]             = pairs.size();
            pair_map[G.IndexOf(e->TwinEdge())] = pairs.size();
            pairs.emplace_back(pair);
        }

        // Loop until the number of vertices is less than $simplification_ratio * initial_size$.
        while (G.NumOfVertices() > simplification_ratio * Qv.size()) {
            // Find the contractable pair with minimal cost.
            std::size_t min_idx = ~0;
            for (std::size_t i = 0; i < pairs.size(); ++i) {
                if (! pairs[i].edge) continue;
                if (!~min_idx || pairs[i].cost < pairs[min_idx].cost) {
                    if (G.IsContractable(pairs[i].edge)) min_idx = i;
                    else pairs[i].edge = nullptr;
                }
            }
            if (!~min_idx) break;

            // top:    the contractable pair with minimal cost
            // v1:     the reserved vertex
            // v2:     the removed vertex
            // result: the contract result
            // ring:   the edge ring of vertex v1
            ContractionPair & top    = pairs[min_idx];
            auto               v1     = top.edge->From();
            auto               v2     = top.edge->To();
            auto               result = G.Contract(top.edge);
            auto               ring   = G.Vertex(v1)->Ring();

            top.edge             = nullptr;            // The contraction has already been done, so the pair is no longer valid. Mark it as invalid.
            output.Positions[v1] = top.targetPosition; // Update the positions.

            // We do something to repair $pair_map$ and $pairs$ because some edges and vertices no longer exist.
            for (int i = 0; i < 2; ++i) {
                DCEL::EdgeIdx removed           = G.IndexOf(result.removed_edges[i].first);
                DCEL::EdgeIdx collapsed         = G.IndexOf(result.collapsed_edges[i].second);
                pairs[pair_map[removed]].edge   = result.collapsed_edges[i].first;
                pairs[pair_map[collapsed]].edge = nullptr;
                pair_map[collapsed]             = pair_map[G.IndexOf(result.collapsed_edges[i].first)];
            }

            // For the two wing vertices, each of them lose one incident face.
            // So, we update the Q matrix.
            Qv[result.removed_faces[0].first] -= Kf[G.IndexOf(result.removed_faces[0].second)];
            Qv[result.removed_faces[1].first] -= Kf[G.IndexOf(result.removed_faces[1].second)];

            // For the vertex v1, Q matrix should be recomputed.
            // And as the position of v1 changed, all the vertices which are on the ring of v1 should update their Q matrix as well.
            Qv[v1] = glm::mat4(0);
            for (auto e : ring) {
                // your code here:
                //     1. Compute the new Kp matrix for $e->Face()$.
                //     2. According to the difference between the old Kp (in $Kf$) and the new Kp (computed in step 1),
                //        update Q matrix of each vertex on the ring (update $Qv$).
                //     3. Update Q matrix of vertex v1 as well (update $Qv$).
                //     4. Update $Kf$.
                const DCEL::Triangle *f  = e->Face();
                DCEL::FaceIdx fi = G.IndexOf(f);
                glm::mat4 newKp = UpdateQ(f);
                glm::mat4 delta = newKp - Kf[fi];
                for (int i = 0; i < 3; ++i) {
                    DCEL::VertexIdx vi = f->VertexIndex(i);
                    if (vi == v1) Qv[vi] += newKp;
                    else          Qv[vi] += delta;
                }
                Kf[fi] = newKp;
            }

            // Finally, as the Q matrix changed, we should update the relative $ContractionPair$ in $pairs$.
            // Any pair with the Q matrix of its endpoints changed, should be remade by $MakePair$.
            // your code here:
            std::set<DCEL::VertexIdx> affected;
            affected.insert(v1);
            for (auto e : ring) {
                affected.insert(e->From());
                affected.insert(e->To());
            }
            for (DCEL::VertexIdx u : affected) {
                for (auto re : G.Vertex(u)->Ring()) {
                    DCEL::HalfEdge const * edge = re->PrevEdge();
                    auto it = pair_map.find(G.IndexOf(edge));
                    if (it == pair_map.end()) continue;
                    std::size_t idx = it->second;
                    if (! pairs[idx].edge) continue;
                    auto u1 = edge->From();
                    auto u2 = edge->To();
                    pairs[idx] = MakePair(
                        edge, output.Positions[u1], output.Positions[u2], Qv[u1] + Qv[u2]);
                }
            }

        }

        // In the end, we check if the result mesh is watertight and manifold.
        if (! G.DebugWatertightManifold()) {
            spdlog::warn("VCX::Labs::GeometryProcessing::SimplifyMesh(..): Result is not watertight manifold.");
        }

        auto exported = G.ExportMesh();
        output.Indices.swap(exported.Indices);
    }

    /******************* 4. Mesh Smoothing *****************/
    void SmoothMesh(Engine::SurfaceMesh const & input, Engine::SurfaceMesh & output, std::uint32_t numIterations, float lambda, bool useUniformWeight) {
        // Define function to compute cotangent value of the angle v1-vAngle-v2
        static constexpr auto GetCotangent {
            [] (glm::vec3 vAngle, glm::vec3 v1, glm::vec3 v2) -> float {
                // your code here:
                glm::vec3 u = v1 - vAngle;
                glm::vec3 v = v2 - vAngle;
                float s = glm::length(glm::cross(u, v));
                if (s < 1e-8f) return 0.0f;
                float cot = glm::dot(u, v) / s;
                return glm::clamp(cot, 0.0f, 1e3f);
            }
        };

        DCEL G(input);
        if (! G.IsManifold()) {
            spdlog::warn("VCX::Labs::GeometryProcessing::SmoothMesh(..): Non-manifold mesh.");
            return;
        }
        // We only allow watertight mesh.
        if (! G.IsWatertight()) {
            spdlog::warn("VCX::Labs::GeometryProcessing::SmoothMesh(..): Non-watertight mesh.");
            return;
        }

        Engine::SurfaceMesh prev_mesh;
        prev_mesh.Positions = input.Positions;
        for (std::uint32_t iter = 0; iter < numIterations; ++iter) {
            Engine::SurfaceMesh curr_mesh = prev_mesh;
            for (std::size_t i = 0; i < input.Positions.size(); ++i) {
                // your code here: curr_mesh.Positions[i] = ...
                auto v = G.Vertex(i);
                std::unordered_map<uint32_t, float> weights;
                if (useUniformWeight) {
                    for (auto j : v->Neighbors())
                        weights[j] = 1.f;
                } else {
                    for (auto e : v->Ring()) {
                        auto j = e->From();
                        auto k = e->To();
                        weights[j] += GetCotangent(
                            prev_mesh.Positions[k],
                            prev_mesh.Positions[i],
                            prev_mesh.Positions[j]
                        );
                        weights[k] += GetCotangent(
                            prev_mesh.Positions[j],
                            prev_mesh.Positions[i],
                            prev_mesh.Positions[k]
                        );
                    }
                }
                glm::vec3 sum(0.0f);
                float wsum = 0.0f;
                for (auto & [j, w] : weights) {
                    sum  += w * prev_mesh.Positions[j];
                    wsum += w;
                }
                if (wsum > 1e-8f)
                    curr_mesh.Positions[i] = (1.0f - lambda) * prev_mesh.Positions[i] + lambda * (sum / wsum);
                else
                    curr_mesh.Positions[i] = prev_mesh.Positions[i];
            }
            // Move curr_mesh to prev_mesh.
            prev_mesh.Swap(curr_mesh);
        }
        // Move prev_mesh to output.
        output.Swap(prev_mesh);
        // Copy indices from input.
        output.Indices = input.Indices;
    }

    /******************* 5. Marching Cubes *****************/
    void MarchingCubes(Engine::SurfaceMesh & output, const std::function<float(const glm::vec3 &)> & sdf, const glm::vec3 & grid_min, const float dx, const int n) {
        // your code here:
        int N = n + 1;

        auto gridPos = [&](int x, int y, int z) {
            return grid_min + dx * glm::vec3(x, y, z);
        };
        auto gridIdx = [&](int x, int y, int z) {
            return (z * N + y) * N + x;
        };

        std::vector<float> field(N * N * N);
        for (int z = 0; z < N; ++z)
            for (int y = 0; y < N; ++y)
                for (int x = 0; x < N; ++x)
                    field[gridIdx(x, y, z)] = sdf(gridPos(x, y, z));

        output.Positions.clear();
        output.Indices.clear();

        std::vector<std::uint32_t> edgeVtx((std::size_t)N * N * N * 3, ~0u);
        for (int axis = 0; axis < 3; ++axis)
            for (int z = 0; z < N; ++z)
                for (int y = 0; y < N; ++y)
                    for (int x = 0; x < N; ++x) {
                        if (axis == 0 && x == n) continue;
                        if (axis == 1 && y == n) continue;
                        if (axis == 2 && z == n) continue;
                        int x1 = x, y1 = y, z1 = z;
                        if (axis == 0) ++x1;
                        if (axis == 1) ++y1;
                        if (axis == 2) ++z1;
                        float s0 = field[gridIdx(x, y, z)];
                        float s1 = field[gridIdx(x1, y1, z1)];
                        if ((s0 < 0.f) == (s1 < 0.f)) continue;
                        float ratio = s0 / (s0 - s1);
                        edgeVtx[(std::size_t)gridIdx(x, y, z) * 3 + axis] = (std::uint32_t)output.Positions.size();
                        output.Positions.emplace_back(gridPos(x, y, z) + ratio * (gridPos(x1, y1, z1) - gridPos(x, y, z)));
                    }

        for (int z = 0; z < n; ++z)
            for (int y = 0; y < n; ++y)
                for (int x = 0; x < n; ++x) {
                    std::uint32_t mask = 0;
                    for (int c = 0; c < 8; ++c) {
                        int cx = x + (c & 1);
                        int cy = y + ((c >> 1) & 1);
                        int cz = z + (c >> 2);
                        if (field[gridIdx(cx, cy, cz)] < 0.f) mask |= 1u << c;
                    }
                    if (c_EdgeStateTable[mask] == 0) continue;
                    for (int t = 0; t < 16; t += 3) {
                        if (c_EdgeOrdsTable[mask][t] < 0) break;
                        for (int m = 0; m < 3; ++m) {
                            int e = c_EdgeOrdsTable[mask][t + m];
                            int axis = e >> 2;
                            int c0 = ((e & 1) << ((axis + 1) % 3)) | (((e >> 1) & 1) << ((axis + 2) % 3));
                            int gx = x + (c0 & 1);
                            int gy = y + ((c0 >> 1) & 1);
                            int gz = z + (c0 >> 2);
                            output.Indices.emplace_back(edgeVtx[(std::size_t)gridIdx(gx, gy, gz) * 3 + axis]);
                        }
                    }
                }
    }
} // namespace VCX::Labs::GeometryProcessing
