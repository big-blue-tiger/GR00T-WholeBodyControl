"""Independent deterministic GRAIL PyTorch actor, no Isaac Lab environment."""

import argparse
import signal
import time

import numpy as np
import torch
import zmq

from gear_sonic.utils.grail.policy import GrailPolicy, finger_targets, fsq
from gear_sonic.utils.grail.checkpoints import DEFAULT_CHECKPOINT
from gear_sonic.utils.grail.transport import pack, socket, unpack


class GrailActor:
    """Hold the last decoder latent from release until the episode resets."""

    def __init__(self, policy, release_override=True):
        self.policy = policy
        self.release_override = release_override
        self.episode_id = None
        self.last_raw = self.last_token = None
        self.released = False

    def act(self, packet):
        if packet["episode_id"] != self.episode_id:
            self.episode_id = packet["episode_id"]
            self.last_raw = self.last_token = None
            self.released = False

        self.released |= self.release_override and packet["release_requested"]
        # On the first release request, reuse the last published latent rather
        # than evaluating a new one. If the first packet already requests
        # release (e.g. actor restart), evaluate once to initialize the hold.
        if not self.released or self.last_raw is None:
            action = self.policy(packet["proprio"], packet["task"])
            self.last_token = fsq(action[:, :64])[0].cpu().numpy().copy()
            self.last_raw = action[0].cpu().numpy().copy()
        if self.released:
            # Keep the raw latent used by proprio history consistent with the
            # frozen FSQ token, while holding both open-hand primitives.
            self.last_raw[64:66] = -1.0
        return self.last_raw.copy(), self.last_token.copy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                        help=".pt file or directory (newest complete .pt)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--no-release-override",
        action="store_true",
        help="Disable forced-open hands and latent hold on release requests",
    )
    parser.add_argument("--observation-endpoint", default="tcp://127.0.0.1:5560")
    parser.add_argument("--action-endpoint", default="tcp://127.0.0.1:5561")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    policy = GrailPolicy(args.checkpoint, args.device)
    policy(np.zeros((5, 161)), np.zeros(79))  # Warm CPU/GPU before subscribing.
    actor = GrailActor(policy, release_override=not args.no_release_override)
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
    print(f"[GRAIL] Checkpoint: {policy.checkpoint}", flush=True)
    print("[GRAIL] Actor ready: 884 -> 2048 -> 2048 -> 512 -> 66; deterministic mean", flush=True)
    if not args.no_release_override:
        print("[GRAIL] Release override enabled: open both hands and hold the last decoder latent until reset", flush=True)
    try:
        while not stopping:
            if not sub.poll(100):
                continue
            packet = unpack(sub.recv())
            started = time.monotonic()
            raw, token = actor.act(packet)
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
