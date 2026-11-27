// One TensorRT engine, independent simulation-clock contexts, atomic replies.
#include <chrono>
#include <iostream>
#include <unordered_map>
#include <msgpack.hpp>
#include <zmq.hpp>
#include "control_policy.hpp"
#include "simulation_control.hpp"

struct RecordedAction {
    std::vector<double> raw, targets;
    MSGPACK_DEFINE_MAP(raw, targets);
};
struct RecordedFrame {
    std::vector<double> q, dq, quat, angular;
    RecordedAction previous;
    MSGPACK_DEFINE_MAP(q, dq, quat, angular, previous);
};
struct Request {
    int env_id, episode_id, step;
    std::vector<double> q, dq, quat, angular, hand_q, action;
    std::vector<RecordedFrame> history;
    RecordedAction previous;
    MSGPACK_DEFINE_MAP(env_id, episode_id, step, q, dq, quat, angular, hand_q, action, history, previous);
};
struct Reply {
    int env_id, episode_id, step;
    std::vector<double> q, dq, kp, kd, tau, raw_action;
    std::vector<float> observation;
    double inference_s;
    MSGPACK_DEFINE_MAP(env_id, episode_id, step, q, dq, kp, kd, tau, raw_action, observation, inference_s);
};

int main(int argc, char** argv) try {
    if (argc != 3) throw std::runtime_error("usage: grail_offline_decoder MODEL.onnx IPC_ENDPOINT");
    PolicyEngine engine;
    if (!engine.Initialize(argv[1], false) || engine.GetInputDimension()!=994 || engine.GetActionDimension()!=29)
        throw std::runtime_error("expected GRAIL SONIC decoder 994 -> 29");
    std::fill(engine.GetInputBuffer().begin(), engine.GetInputBuffer().end(), 0);
    if (!engine.Infer() || !engine.CaptureGraph()) throw std::runtime_error("decoder warmup failed");
    zmq::context_t transport(1);
    zmq::socket_t socket(transport, zmq::socket_type::rep);
    socket.set(zmq::sockopt::linger, 1000); // Deliver the final shutdown acknowledgement.
    socket.bind(argv[2]);
    std::unordered_map<int, SimulationContext> contexts;
    std::cout << "[OfflineDecoder] ready" << std::endl;
    while (true) {
        zmq::message_t message;
        if (!socket.recv(message)) break;
        auto parsed = msgpack::unpack(static_cast<const char*>(message.data()), message.size());
        auto requests = parsed.get().as<std::vector<Request>>();
        std::vector<Reply> replies;
        for (const auto& request: requests) {
            if (request.hand_q.size()!=14) throw std::runtime_error("expected 14 measured hand joints");
            auto& context = contexts[request.env_id];
            if (!request.history.empty()) {
                std::vector<SimulationFrame> frames;
                frames.reserve(request.history.size());
                for (const auto& frame : request.history) {
                    frames.push_back(simulation_frame(frame.q, frame.dq, frame.quat, frame.angular,
                        recorded_body_action(frame.previous.raw, frame.previous.targets)));
                }
                context.seed(request.episode_id, request.step, frames,
                    recorded_body_action(request.previous.raw, request.previous.targets));
            }
            Reply reply;
            reply.env_id=request.env_id; reply.episode_id=request.episode_id; reply.step=request.step;
            reply.observation=context.observe(request.episode_id, request.step, request.q,
                request.dq, request.quat, request.angular, request.action);
            std::copy(reply.observation.begin(), reply.observation.end(), engine.GetInputBuffer().begin());
            const auto start=std::chrono::steady_clock::now();
            if (!engine.Infer()) throw std::runtime_error("decoder inference failed");
            reply.inference_s=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
            const auto& output=engine.GetActionBuffer();
            auto body=body_command(output.data());
            reply.q.assign(body.q_target.begin(), body.q_target.end());
            reply.dq.assign(body.dq_target.begin(), body.dq_target.end());
            reply.kp.assign(body.kp.begin(), body.kp.end());
            reply.kd.assign(body.kd.begin(), body.kd.end());
            reply.tau.assign(body.tau_ff.begin(), body.tau_ff.end());
            reply.raw_action.assign(output.begin(), output.end());
            std::copy(output.begin(), output.end(), context.last_action.begin());
            for (int j=0; j<14; ++j) {
                reply.q.push_back(hand_target(request.action[64+j], request.hand_q[j], j<7, j%7));
                reply.dq.push_back(0); reply.kp.push_back(1.5); reply.kd.push_back(.1); reply.tau.push_back(0);
            }
            replies.push_back(std::move(reply));
        }
        msgpack::sbuffer buffer;
        msgpack::pack(buffer, replies);
        socket.send(zmq::buffer(buffer.data(), buffer.size()), zmq::send_flags::none);
        if (requests.empty()) break;
    }
    return 0;
} catch (const std::exception& error) {
    std::cerr << "[OfflineDecoder] " << error.what() << std::endl;
    return 1;
}
