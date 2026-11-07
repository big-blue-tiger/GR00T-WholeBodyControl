// Simulation-clock SONIC control. History layout matches G1Deploy's enabled
// ten-frame gatherers: token, angular velocity, q, dq, last action, gravity.
#pragma once

#include <algorithm>
#include <array>
#include <deque>
#include <stdexcept>
#include <string>
#include <vector>
#include "robot_parameters.hpp"
#include "policy_parameters.hpp"

inline MotorCommand body_command(const float* action) {
    MotorCommand command;
    for (int i = 0; i < 29; ++i) {
        command.q_target[i] = default_angles[i] + action[isaaclab_to_mujoco[i]] * g1_action_scale[i];
        command.kp[i] = kps[i];
        command.kd[i] = kds[i];
    }
    return command;
}

inline double hand_target(double desired, double current, bool left, int joint,
                          double close_ratio = 1.0) {
    constexpr double hi[2][7] = {{1.05, 1.05, 1.75, 0, 0, 0, 0},
                                {1.05, .742, 0, 1.57, 1.75, 1.57, 1.75}};
    constexpr double lo[2][7] = {{-1.05, -.724, 0, -1.57, -1.75, -1.57, -1.75},
                                {-1.05, -1.05, -1.75, 0, 0, 0, 0}};
    int side = left ? 0 : 1;
    if (desired > 0 && hi[side][joint] > 0)
        desired = std::min(desired, close_ratio * hi[side][joint]);
    else if (desired < 0 && lo[side][joint] < 0)
        desired = std::max(desired, close_ratio * lo[side][joint]);
    return current + std::clamp(desired - current, -.25, .25);
}

struct SimulationFrame {
    std::array<double, 3> angular{}, gravity{};
    std::array<double, 29> q{}, dq{}, last_action{};
};

struct SimulationContext {
    int episode = -1;
    int next_step = 0;
    std::array<double, 29> last_action{};
    std::deque<SimulationFrame> history;

    std::vector<float> observe(int episode_id, int step, const std::vector<double>& q,
        const std::vector<double>& dq, const std::vector<double>& quat,
        const std::vector<double>& angular, const std::vector<double>& action) {
        if (q.size()!=29 || dq.size()!=29 || quat.size()!=4 || angular.size()!=3 || action.size()!=78)
            throw std::runtime_error("invalid SONIC state/action dimensions");
        if (episode_id != episode) {
            if (step != 0) throw std::runtime_error("episode must begin at step zero");
            episode = episode_id;
            next_step = 0;
            history.clear();
            last_action.fill(0);
        }
        if (step != next_step) throw std::runtime_error("unexpected control step");
        SimulationFrame frame;
        for (int i=0; i<29; ++i) {
            int hw = mujoco_to_isaaclab[i];
            frame.q[i] = q[hw] - default_angles[hw];
            frame.dq[i] = dq[hw];
        }
        std::copy_n(angular.begin(), 3, frame.angular.begin());
        const auto w=quat[0], x=quat[1], y=quat[2], z=quat[3];
        // quat_rotate(conjugate(q), [0,0,-1]), identical to utils.hpp.
        frame.gravity = {2*(w*y-x*z), -2*(y*z+w*x), 2*(x*x+y*y)-1};
        frame.last_action = last_action;
        if (history.empty()) history.assign(10, frame); // frozen-state preparation
        else { history.pop_front(); history.push_back(frame); }
        std::vector<float> obs(action.begin(), action.begin()+64);
        auto append = [&](auto member) {
            for (const auto& f: history)
                for (double v: f.*member) obs.push_back(static_cast<float>(v));
        };
        append(&SimulationFrame::angular);
        append(&SimulationFrame::q);
        append(&SimulationFrame::dq);
        append(&SimulationFrame::last_action);
        append(&SimulationFrame::gravity);
        ++next_step;
        return obs;
    }
};
