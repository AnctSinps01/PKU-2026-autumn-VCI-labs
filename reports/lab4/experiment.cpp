#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include "Labs/4-Animation/tasks.h"
using namespace VCX::Labs::Animation;
using Clock = std::chrono::steady_clock;

MassSpringSystem Cloth(float mass = 1, float stiffness = 100, float damping = .2f, float gravity = .3f) {
    MassSpringSystem s;
    s.Mass = mass; s.Stiffness = stiffness; s.Damping = damping; s.Gravity = gravity;
    auto id = [](int i, int j) { return i * 11 + j; };
    for (int i = 0; i <= 10; ++i) for (int j = 0; j <= 10; ++j) {
        s.AddParticle({i * .2f, 1.5f, j * .2f - 1.f});
        if (i > 0) s.AddSpring(id(i,j), id(i-1,j));
        if (i > 1) s.AddSpring(id(i,j), id(i-2,j));
        if (j > 0) s.AddSpring(id(i,j), id(i,j-1));
        if (j > 1) s.AddSpring(id(i,j), id(i,j-2));
        if (i > 0 && j > 0) s.AddSpring(id(i,j), id(i-1,j-1));
        if (i > 0 && j < 10) s.AddSpring(id(i,j), id(i-1,j+1));
    }
    s.Fixed[0] = s.Fixed[10] = true;
    return s;
}

void OriginalEuler(MassSpringSystem & s, float dt) {
    float h = dt / 1000;
    for (int step = 0; step < 1000; ++step) {
        std::vector<glm::vec3> forces(s.Positions.size(), glm::vec3(0));
        for (auto const & spring : s.Springs) {
            auto [i, j] = spring.AdjIdx;
            auto d = s.Positions[j] - s.Positions[i];
            auto e = glm::normalize(d);
            auto f = (s.Stiffness * (glm::length(d) - spring.RestLength)
                + s.Damping * glm::dot(s.Velocities[j] - s.Velocities[i], e)) * e;
            forces[i] += f; forces[j] -= f;
        }
        for (int i = 0; i < s.Positions.size(); ++i) if (!s.Fixed[i]) {
            s.Velocities[i] += h * (forces[i] / s.Mass + glm::vec3(0, -s.Gravity, 0));
            s.Positions[i] += h * s.Velocities[i];
        }
    }
}

float Error(IKSystem const & ik, glm::vec3 target) { return glm::distance(ik.JointGlobalPosition.back(), target); }
float LengthError(IKSystem const & ik) {
    float result = 0;
    for (int i = 1; i < ik.NumJoints(); ++i)
        result = std::max(result, std::abs(glm::distance(ik.JointGlobalPosition[i], ik.JointGlobalPosition[i-1]) - ik.JointOffsetLength[i]));
    return result;
}
using Solver = void(*)(IKSystem &, glm::vec3 const &, int, float);

int main(int argc, char ** argv) {
    std::string out = argc > 1 ? argv[1] : ".";
    std::cout << std::setprecision(9);
    int failures = 0;
    auto check = [&](bool condition, std::string const & name) {
        std::cout << "CHECK," << name << ',' << (condition ? "PASS" : "FAIL") << '\n';
        failures += !condition;
    };
    IKSystem fk;
    fk.JointLocalRotation[0] = glm::angleAxis(glm::half_pi<float>(), glm::vec3(0,0,1));
    ForwardKinematics(fk, 0);
    check(glm::distance(fk.JointGlobalPosition.back(), glm::vec3(2.6f,1.7f,0)) < 2e-6f, "FK_root_rotation");
    fk.JointLocalRotation[2] = glm::angleAxis(.3f, glm::normalize(glm::vec3(1,2,3)));
    auto partial = fk;
    ForwardKinematics(fk, 0); ForwardKinematics(partial, 2);
    float partialError = 0;
    for (int i = 0; i < fk.NumJoints(); ++i) partialError = std::max(partialError, glm::distance(fk.JointGlobalPosition[i], partial.JointGlobalPosition[i]));
    check(partialError < 1e-6f && LengthError(fk) < 1e-6f, "FK_partial_and_lengths");

    glm::vec3 targets[] = {{.6f,0,.3f}, {1.6f,.2f,.2f}, {2.2f,.6f,.8f}, {3,0,0}, {0,0,0}};
    for (auto algorithm : {0,1}) {
        Solver solve = algorithm == 0 ? InverseKinematicsCCD : InverseKinematicsFABR;
        std::string name = algorithm == 0 ? "CCD" : "FABR";
        for (int t = 0; t < 5; ++t) {
            int iterations = 100;
            IKSystem ik;
            for (int cap = 0; cap <= 100; ++cap) {
                ik.JointLocalRotation.assign(ik.NumJoints(), glm::quat(1,0,0,0));
                solve(ik, targets[t], cap, 1e-4f);
                if (Error(ik, targets[t]) <= 1e-4f) { iterations = cap; break; }
            }
            std::cout << "IK_TARGET," << name << ',' << t << ',' << iterations << ',' << Error(ik,targets[t]) << ',' << LengthError(ik) << '\n';
            check(std::isfinite(Error(ik,targets[t])) && LengthError(ik) < 3e-6f && glm::distance(ik.JointGlobalPosition[0],ik.JointLocalOffset[0]) < 1e-7f, name + "_target_" + std::to_string(t) + "_invariants");
        }
        for (int shape = 0; shape < 6; ++shape) {
            IKSystem ik;
            float maxError = 0, sumError = 0, boneError = 0;
            int misses = 0;
            auto const points = ik.TargetPositionList[shape];
            auto start = Clock::now();
            for (auto const & target : *points) {
                solve(ik,target,100,1e-4f);
                float e = Error(ik,target);
                maxError = std::max(maxError,e); sumError += e; misses += e > 1e-4f;
                boneError = std::max(boneError,LengthError(ik));
            }
            auto ms = std::chrono::duration<double,std::milli>(Clock::now()-start).count();
            std::cout << "IK_PATH," << name << ',' << shape << ',' << points->size() << ',' << maxError << ',' << sumError / points->size() << ',' << misses << ',' << boneError << ',' << ms / points->size() << '\n';
            check(std::isfinite(sumError) && maxError < 1.01e-4f && boneError < 4e-6f, name + "_path_" + std::to_string(shape));
            if (shape == 5) {
                std::ofstream positions(out + "/trajectory_" + name + ".csv");
                positions << "tx,ty,tz,x,y,z\n";
                IKSystem rendered;
                for (auto const & target : *points) {
                    solve(rendered,target,100,1e-4f); auto p = rendered.EndEffectorPosition();
                    positions << target.x << ',' << target.y << ',' << target.z << ',' << p.x << ',' << p.y << ',' << p.z << '\n';
                }
            }
        }
    }

    IKSystem rotated;
    rotated.JointLocalRotation[0] = glm::angleAxis(glm::half_pi<float>(), glm::vec3(0,0,1));
    auto rotatedTarget = rotated.JointLocalOffset[0] + glm::vec3(1.7f,0,0);
    InverseKinematicsCCD(rotated,rotatedTarget,100,1e-4f);
    check(Error(rotated,rotatedTarget) < 1e-4f, "CCD_rotated_collinear_pose");
    IKSystem coincident;
    coincident.JointLocalRotation[1] = glm::angleAxis(.5f,glm::vec3(0,0,1));
    ForwardKinematics(coincident,0);
    auto coincidentTarget = coincident.JointGlobalPosition[3];
    InverseKinematicsFABR(coincident,coincidentTarget,100,1e-4f);
    check(std::isfinite(Error(coincident,coincidentTarget)) && LengthError(coincident) < 3e-6f, "FABR_target_on_existing_joint");

    MassSpringSystem freefall;
    freefall.AddParticle({0,0,0});
    AdvanceMassSpringSystem(freefall,.1f);
    std::cout << "FREEFALL," << freefall.Positions[0].y << ',' << freefall.Velocities[0].y << '\n';
    check(std::abs(freefall.Velocities[0].y + .03f) < 1e-6f && std::abs(freefall.Positions[0].y + .003f) < 1e-6f, "backward_euler_freefall");
    MassSpringSystem pair;
    pair.Gravity = 0; pair.Stiffness = 100; pair.Damping = 0;
    pair.AddParticle({0,0,0}); pair.AddParticle({1.2f,0,0}); pair.AddSpring(0,1,1);
    pair.Fixed[0] = true;
    AdvanceMassSpringSystem(pair,.1f);
    std::cout << "PAIR," << pair.Positions[1].x << ',' << pair.Velocities[1].x << '\n';
    check(std::abs(pair.Positions[1].x - 1.1f) < 1e-6f && std::abs(pair.Velocities[1].x + 1.f) < 1e-6f && pair.Positions[0] == glm::vec3(0), "backward_euler_analytic_spring");

    for (int scenario = 0; scenario < 3; ++scenario) {
        auto s = scenario == 0 ? Cloth() : scenario == 1 ? Cloth(.5f,300,.1f,1) : Cloth(10,10,10,.1f);
        float h = scenario == 0 ? 1.f/60 : 1.f/30;
        auto initial = s.Positions;
        int frames = scenario == 0 ? 600 : 300;
        auto start = Clock::now();
        bool finite = true;
        float maxStretch = 0, maxFixedError = 0, maxSpeed = 0;
        std::ofstream history(out + "/cloth_" + std::to_string(scenario) + ".csv");
        history << "time,center_y,free_corner_y,max_speed\n";
        for (int frame = 0; frame < frames; ++frame) {
            AdvanceMassSpringSystem(s,h);
            float speed = 0;
            for (int i = 0; i < s.Positions.size(); ++i) {
                auto p = s.Positions[i];
                finite &= std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z);
                speed = std::max(speed,glm::length(s.Velocities[i]));
                if (s.Fixed[i]) maxFixedError = std::max(maxFixedError,glm::distance(p,initial[i]));
            }
            maxSpeed = std::max(maxSpeed,speed);
            for (auto const & spring : s.Springs) maxStretch = std::max(maxStretch,glm::distance(s.Positions[spring.AdjIdx.first],s.Positions[spring.AdjIdx.second])/spring.RestLength);
            history << (frame+1)*h << ',' << s.Positions[60].y << ',' << s.Positions[120].y << ',' << speed << '\n';
            if (scenario == 0 && (frame == 59 || frame == 179 || frame == 599)) {
                std::ofstream snapshot(out + "/cloth_t" + std::to_string(frame+1) + ".csv");
                snapshot << "x,y,z\n";
                for (auto p : s.Positions) snapshot << p.x << ',' << p.y << ',' << p.z << '\n';
            }
        }
        auto ms = std::chrono::duration<double,std::milli>(Clock::now()-start).count()/frames;
        std::cout << "CLOTH," << scenario << ',' << s.Positions.size() << ',' << s.Springs.size() << ',' << frames << ',' << h << ',' << ms << ',' << finite << ',' << maxFixedError << ',' << maxStretch << ',' << maxSpeed << ',' << s.Positions[60].y << '\n';
        check(finite && maxFixedError == 0, "cloth_scenario_" + std::to_string(scenario));
    }
    auto explicitCloth = Cloth();
    auto start = Clock::now();
    for (int i = 0; i < 120; ++i) OriginalEuler(explicitCloth,1.f/60);
    auto originalMs = std::chrono::duration<double,std::milli>(Clock::now()-start).count()/120;
    std::cout << "ORIGINAL_EULER_MS," << originalMs << '\n';
    std::cout << "FAILURES," << failures << '\n';
    return failures ? 1 : 0;
}
