"""Versioned latest-value IPC. Episode identity is independent of sim time."""

import msgpack
import numpy as np
import zmq

VERSION = 1


def pack(packet, **arrays):
    packet = dict(packet, version=VERSION)
    packet["arrays"] = {
        key: dict(
            shape=list(np.asarray(value).shape), data=np.asarray(value, dtype="<f4").tobytes()
        )
        for key, value in arrays.items()
    }
    return msgpack.packb(packet, use_bin_type=True)


def unpack(raw):
    packet = msgpack.unpackb(raw, raw=False)
    if packet["version"] != VERSION:
        raise ValueError("GRAIL IPC version mismatch")
    for key, value in packet.pop("arrays").items():
        packet[key] = np.frombuffer(value["data"], dtype="<f4").reshape(value["shape"]).copy()
        if not np.isfinite(packet[key]).all():
            raise ValueError(f"Nonfinite IPC array {key}")
    return packet


def socket(context, kind, endpoint, bind=False, topic=b""):
    result = context.socket(kind)
    result.setsockopt(zmq.LINGER, 0)
    if kind == zmq.SUB:
        result.setsockopt(zmq.SUBSCRIBE, topic)
        result.setsockopt(zmq.CONFLATE, 1)
    else:
        result.setsockopt(zmq.SNDHWM, 1)
    (result.bind if bind else result.connect)(endpoint)
    return result
