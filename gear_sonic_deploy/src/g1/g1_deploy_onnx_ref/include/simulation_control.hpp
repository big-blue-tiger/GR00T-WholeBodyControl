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

inline std::array<double, 29> recorded_body_action(
    const std::vector<double>& raw, const std::vector<double>& targets) {
    std::array<double, 29> action{};
    if (raw.size() == action.size() && targets.empty()) {
        std::copy(raw.begin(), raw.end(), action.begin());
    } else if (targets.size() == action.size() && raw.empty()) {
        for (int i = 0; i < 29; ++i) {
            action[isaaclab_to_mujoco[i]] = (targets[i] - default_angles[i]) / g1_action_scale[i];
        }
    } else {
        throw std::runtime_error("expected one recorded SONIC action or body target");
    }
    return action;
}

inline SimulationFrame simulation_frame(const std::vector<double>& q,
    const std::vector<double>& dq, const std::vector<double>& quat,
    const std::vector<double>& angular, const std::array<double, 29>& last_action) {
    if (q.size()!=29 || dq.size()!=29 || quat.size()!=4 || angular.size()!=3)
        throw std::runtime_error("invalid SONIC state dimensions");
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
    return frame;
}

struct SimulationContext {
    int episode = -1;
    int next_step = 0;
    std::array<double, 29> last_action{};
    std::deque<SimulationFrame> history;

    void push(const SimulationFrame& frame) {
        if (history.empty()) history.assign(10, frame);
        else { history.pop_front(); history.push_back(frame); }
    }

    void seed(int episode_id, int step, const std::vector<SimulationFrame>& frames,
              const std::array<double, 29>& previous_action) {
        if (step < 1 || frames.size() != static_cast<size_t>(std::min(step, 9)))
            throw std::runtime_error("invalid recorded SONIC history length");
        episode = episode_id;
        next_step = step;
        history.clear();
        for (const auto& frame : frames) push(frame);
        last_action = previous_action;
    }

    std::vector<float> observe(int episode_id, int step, const std::vector<double>& q,
        const std::vector<double>& dq, const std::vector<double>& quat,
        const std::vector<double>& angular, const std::vector<double>& action) {
        if (action.size()!=78) throw std::runtime_error("invalid SONIC action dimensions");
        if (episode_id != episode) {
            if (step != 0) throw std::runtime_error("episode must begin at step zero");
            episode = episode_id;
            next_step = 0;
            history.clear();
            last_action.fill(0);
        }
        if (step != next_step) throw std::runtime_error("unexpected control step");
        push(simulation_frame(q, dq, quat, angular, last_action));
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
