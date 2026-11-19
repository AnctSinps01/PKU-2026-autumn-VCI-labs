#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <algorithm>
#include "Labs/4-Animation/tasks.h"

namespace VCX::Labs::Animation {
    static glm::quat RotationBetween(glm::vec3 const & from, glm::vec3 const & to) {
        float const cosine = glm::dot(from, to);
        if (cosine < -0.999999f) return glm::rotation(from, to);
        glm::vec3 const axis = glm::cross(from, to);
        return glm::normalize(glm::quat(1.f + cosine, axis.x, axis.y, axis.z));
    }

    void ForwardKinematics(IKSystem & ik, int StartIndex) {
        if (StartIndex == 0) {
            ik.JointGlobalRotation[0] = ik.JointLocalRotation[0];
            ik.JointGlobalPosition[0] = ik.JointLocalOffset[0];
            StartIndex = 1;
        }
        for (int i = StartIndex; i < ik.NumJoints(); ++i) {
            ik.JointGlobalPosition[i] = ik.JointGlobalPosition[i - 1]
                + ik.JointGlobalRotation[i - 1] * ik.JointLocalOffset[i];
            ik.JointGlobalRotation[i] = glm::normalize(ik.JointGlobalRotation[i - 1] * ik.JointLocalRotation[i]);
        }
    }

    static void PrepareIK(IKSystem & ik, glm::vec3 const & target, float eps) {
        ForwardKinematics(ik, 0);
        glm::vec3 const direction = ik.EndEffectorPosition() - ik.JointGlobalPosition[0];
        glm::vec3 const toTarget = target - ik.JointGlobalPosition[0];
        if (glm::length2(glm::cross(direction, toTarget)) < 1e-12f && glm::distance(ik.EndEffectorPosition(), target) > eps) {
            glm::vec3 axis = glm::cross(glm::normalize(direction), glm::vec3(1, 0, 0));
            if (glm::length2(axis) < 1e-12f) axis = glm::cross(glm::normalize(direction), glm::vec3(0, 0, 1));
            glm::vec3 const localAxis = glm::inverse(ik.JointGlobalRotation[0]) * glm::normalize(axis);
            ik.JointLocalRotation[1] = glm::angleAxis(.01f, localAxis) * ik.JointLocalRotation[1];
            ForwardKinematics(ik, 0);
        }
    }

    void InverseKinematicsCCD(IKSystem & ik, const glm::vec3 & EndPosition, int maxCCDIKIteration, float eps) {
        PrepareIK(ik, EndPosition, eps);
        for (int iteration = 0; iteration < maxCCDIKIteration && glm::distance(ik.EndEffectorPosition(), EndPosition) > eps; ++iteration) {
            for (int i = ik.NumJoints() - 2; i >= 0; --i) {
                glm::vec3 toEnd = ik.EndEffectorPosition() - ik.JointGlobalPosition[i];
                glm::vec3 toTarget = EndPosition - ik.JointGlobalPosition[i];
                if (glm::length2(toEnd) < 1e-12f || glm::length2(toTarget) < 1e-12f) continue;
                glm::quat rotation = RotationBetween(glm::normalize(toEnd), glm::normalize(toTarget));
                glm::quat global = rotation * ik.JointGlobalRotation[i];
                ik.JointLocalRotation[i] = glm::normalize(i == 0 ? global : glm::inverse(ik.JointGlobalRotation[i - 1]) * global);
                ForwardKinematics(ik, i);
            }
        }
    }

    void InverseKinematicsFABR(IKSystem & ik, const glm::vec3 & EndPosition, int maxFABRIKIteration, float eps) {
        PrepareIK(ik, EndPosition, eps);
        int const n = ik.NumJoints();
        glm::vec3 const root = ik.JointGlobalPosition[0];
        float length = 0;
        for (int i = 1; i < n; ++i) length += ik.JointOffsetLength[i];
        if (glm::distance(root, EndPosition) >= length) {
            glm::vec3 direction = glm::normalize(EndPosition - root);
            for (int i = 1; i < n; ++i)
                ik.JointGlobalPosition[i] = ik.JointGlobalPosition[i - 1] + ik.JointOffsetLength[i] * direction;
        } else {
            for (int iteration = 0; iteration < maxFABRIKIteration && glm::distance(ik.EndEffectorPosition(), EndPosition) > eps; ++iteration) {
                auto const previous = ik.JointGlobalPosition;
                ik.JointGlobalPosition[n - 1] = EndPosition;
                for (int i = n - 2; i >= 0; --i) {
                    glm::vec3 direction = ik.JointGlobalPosition[i] - ik.JointGlobalPosition[i + 1];
                    if (glm::length2(direction) < 1e-12f) direction = previous[i] - previous[i + 1];
                    ik.JointGlobalPosition[i] = ik.JointGlobalPosition[i + 1] + ik.JointOffsetLength[i + 1] * glm::normalize(direction);
                }
                ik.JointGlobalPosition[0] = root;
                for (int i = 1; i < n; ++i) {
                    glm::vec3 direction = ik.JointGlobalPosition[i] - ik.JointGlobalPosition[i - 1];
                    if (glm::length2(direction) < 1e-12f) direction = previous[i] - previous[i - 1];
                    ik.JointGlobalPosition[i] = ik.JointGlobalPosition[i - 1] + ik.JointOffsetLength[i] * glm::normalize(direction);
                }
            }
        }
        for (int i = 0; i < n - 1; ++i)
            ik.JointGlobalRotation[i] = RotationBetween(glm::normalize(ik.JointLocalOffset[i + 1]),
                glm::normalize(ik.JointGlobalPosition[i + 1] - ik.JointGlobalPosition[i]));
        ik.JointLocalRotation[0] = ik.JointGlobalRotation[0];
        for (int i = 1; i < n - 1; ++i)
            ik.JointLocalRotation[i] = glm::normalize(glm::inverse(ik.JointGlobalRotation[i - 1]) * ik.JointGlobalRotation[i]);
        ForwardKinematics(ik, 0);
    }

    IKSystem::Vec3ArrPtr IKSystem::BuildCustomTargetPosition() {
        auto points = std::make_shared<Vec3Arr>();
        auto line = [&](glm::vec3 const & a, glm::vec3 const & b, int samples) {
            for (int i = 0; i < samples; ++i)
                points->push_back(glm::mix(a, b, float(i) / samples));
        };
        glm::vec3 const v[] = {{-.9f, 0, .4f}, {-.6f, 0, -.4f}, {-.3f, 0, .4f}};
        line(v[0], v[1], 80);
        line(v[1], v[2], 80);
        for (int i = 0; i < 160; ++i) {
            float angle = glm::radians(45.f + 270.f * i / 160.f);
            points->emplace_back(.15f + .424264f * std::cos(angle), 0, .494975f * std::sin(angle));
        }
        line({.65f, 0, -.4f}, {.95f, 0, -.4f}, 30);
        line({.95f, 0, -.4f}, {.8f, 0, -.4f}, 15);
        line({.8f, 0, -.4f}, {.8f, 0, .4f}, 80);
        line({.8f, 0, .4f}, {.65f, 0, .4f}, 15);
        line({.65f, 0, .4f}, {.95f, 0, .4f}, 30);
        return points;
    }

    static Eigen::VectorXf glm2eigen(std::vector<glm::vec3> const & glm_v) {
        Eigen::VectorXf v = Eigen::Map<Eigen::VectorXf const, Eigen::Aligned>(reinterpret_cast<float const *>(glm_v.data()), static_cast<int>(glm_v.size() * 3));
        return v;
    }

    static std::vector<glm::vec3> eigen2glm(Eigen::VectorXf const & eigen_v) {
        return std::vector<glm::vec3>(
            reinterpret_cast<glm::vec3 const *>(eigen_v.data()),
            reinterpret_cast<glm::vec3 const *>(eigen_v.data() + eigen_v.size())
        );
    }

    static Eigen::SparseMatrix<float> CreateEigenSparseMatrix(std::size_t n, std::vector<Eigen::Triplet<float>> const & triplets) {
        Eigen::SparseMatrix<float> matLinearized(n, n);
        matLinearized.setFromTriplets(triplets.begin(), triplets.end());
        return matLinearized;
    }

    // solve Ax = b and return x
    static Eigen::VectorXf ComputeSimplicialLLT(
        Eigen::SparseMatrix<float> const & A,
        Eigen::VectorXf const & b) {
        auto solver = Eigen::SimplicialLLT<Eigen::SparseMatrix<float>>(A);
        return solver.solve(b);
    }

    void AdvanceMassSpringSystem(MassSpringSystem & system, float const dt) {
        int const n = static_cast<int>(system.Positions.size());
        Eigen::VectorXf rhs = Eigen::VectorXf::Zero(3 * n);
        std::vector<Eigen::Triplet<float>> triplets;
        triplets.reserve(3 * n + 36 * system.Springs.size());
        for (int i = 0; i < n; ++i) {
            for (int axis = 0; axis < 3; ++axis)
                triplets.emplace_back(3 * i + axis, 3 * i + axis, system.Fixed[i] ? 1.f : system.Mass);
            if (system.Fixed[i]) system.Velocities[i] = glm::vec3(0);
            else rhs[3 * i + 1] = -dt * system.Mass * system.Gravity;
        }
        auto addBlock = [&](int i, int j, Eigen::Matrix3f const & block) {
            if (system.Fixed[i] || system.Fixed[j]) return;
            for (int row = 0; row < 3; ++row)
                for (int col = 0; col < 3; ++col)
                    triplets.emplace_back(3 * i + row, 3 * j + col, block(row, col));
        };
        for (auto const & spring : system.Springs) {
            int const i = static_cast<int>(spring.AdjIdx.first);
            int const j = static_cast<int>(spring.AdjIdx.second);
            glm::vec3 const displacement = system.Positions[j] - system.Positions[i];
            float const length = glm::length(displacement);
            glm::vec3 const direction = displacement / length;
            glm::vec3 const relativeVelocity = system.Velocities[j] - system.Velocities[i];
            Eigen::Vector3f const e(direction.x, direction.y, direction.z);
            Eigen::Vector3f const v(relativeVelocity.x, relativeVelocity.y, relativeVelocity.z);
            Eigen::Matrix3f const outer = e * e.transpose();
            Eigen::Matrix3f const stiffness = system.Stiffness * (outer
                + std::max(0.f, 1.f - spring.RestLength / length) * (Eigen::Matrix3f::Identity() - outer));
            Eigen::Matrix3f const damping = system.Damping * outer;
            Eigen::Vector3f const force = (system.Stiffness * (length - spring.RestLength)
                + system.Damping * e.dot(v)) * e;
            Eigen::Vector3f const impulse = dt * force + dt * dt * stiffness * v;
            if (!system.Fixed[i]) rhs.segment<3>(3 * i) += impulse;
            if (!system.Fixed[j]) rhs.segment<3>(3 * j) -= impulse;
            Eigen::Matrix3f const block = dt * damping + dt * dt * stiffness;
            addBlock(i, i, block);
            addBlock(j, j, block);
            addBlock(i, j, -block);
            addBlock(j, i, -block);
        }
        auto const matrix = CreateEigenSparseMatrix(3 * n, triplets);
        auto const deltaVelocity = eigen2glm(ComputeSimplicialLLT(matrix, rhs));
        for (int i = 0; i < n; ++i) {
            if (system.Fixed[i]) continue;
            system.Velocities[i] += deltaVelocity[i];
            system.Positions[i] += dt * system.Velocities[i];
        }
    }
}
