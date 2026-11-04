"""Independent deterministic GRAIL PyTorch actor, no Isaac Lab environment."""

import argparse
import signal
import time

import numpy as np
import torch
import zmq

from gear_sonic.utils.grail.policy import GrailPolicy, finger_targets, fsq
from gear_sonic.utils.grail.transport import pack, socket, unpack


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--no-release-override",
        action="store_true",
        help="Disable the temporary forced-open hand override on drop requests",
    )
    parser.add_argument("--observation-endpoint", default="tcp://127.0.0.1:5560")
    parser.add_argument("--action-endpoint", default="tcp://127.0.0.1:5561")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    policy = GrailPolicy(args.checkpoint, args.device)
    policy(np.zeros((5, 161)), np.zeros(87))  # Warm CPU/GPU before subscribing.
    context = zmq.Context()
    sub = socket(context, zmq.SUB, args.observation_endpoint)
    pub = socket(context, zmq.PUB, args.action_endpoint, bind=True)
    stopping = False

    def stop(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    count, latency, report = 0, 0.0, time.monotonic()
    print("[GRAIL] Actor ready: 892 -> 2048 -> 2048 -> 512 -> 66; deterministic mean", flush=True)
    if not args.no_release_override:
        print("[GRAIL] Release override enabled: drop >= 0.5 forces both hands open", flush=True)
    try:
        while not stopping:
            if not sub.poll(100):
                continue
            packet = unpack(sub.recv())
            started = time.monotonic()
            action = policy(packet["proprio"], packet["task"])
            token = fsq(action[:, :64])[0].cpu().numpy()
            raw = action[0].cpu().numpy().copy()
            # The final task feature is drop_button. Negative hand actions select
            # the checkpoint's open pos_0 primitive; body tokens stay unchanged.
            if not args.no_release_override and packet["task"][-1] >= 0.5:
                raw[64:66] = -1.0
            left, right = finger_targets(raw, policy.config)
            elapsed = time.monotonic() - started
            pub.send(
                pack(
                    dict(
                        episode_id=packet["episode_id"],
                        sequence=packet["sequence"],
                        sim_time=packet["sim_time"],
                        observation_time=packet["observation_time"],
                        inference_s=elapsed,
                    ),
                    action=raw,
                    token=token,
                    left=left,
                    right=right,
                ),
                zmq.NOBLOCK,
            )
            count, latency = count + 1, latency + elapsed
            if time.monotonic() - report >= 5:
                print(
                    f"[GRAIL] {count / (time.monotonic() - report):.1f} Hz; "
                    f"inference mean {latency / count * 1000:.2f} ms",
                    flush=True,
                )
                count, latency, report = 0, 0.0, time.monotonic()
    finally:
        sub.close(0)
        pub.close(0)
        context.term()


if __name__ == "__main__":
    main()
