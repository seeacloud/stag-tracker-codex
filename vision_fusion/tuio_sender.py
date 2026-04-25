from __future__ import annotations

import math
import socket
import struct
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .models import Track, bbox_center
from .screen_mapper import ScreenMapper


@dataclass(slots=True)
class TuioObjectState:
    session_id: int
    symbol_id: int
    x: float
    y: float
    angle: float
    x_velocity: float = 0.0
    y_velocity: float = 0.0
    angle_velocity: float = 0.0
    motion_accel: float = 0.0
    rotation_accel: float = 0.0


class TuioSender:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 3333,
        source: str = "vision_fusion",
    ) -> None:
        self.host = host
        self.port = port
        self.source = source
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._frame = 0

    def close(self) -> None:
        self._socket.close()

    def send(self, objects: Iterable[TuioObjectState]) -> None:
        object_list = list(objects)
        messages = [
            osc_message("/tuio/2Dobj", ["source", self.source]),
            *[
                osc_message(
                    "/tuio/2Dobj",
                    [
                        "set",
                        obj.session_id,
                        obj.symbol_id,
                        obj.x,
                        obj.y,
                        obj.angle,
                        obj.x_velocity,
                        obj.y_velocity,
                        obj.angle_velocity,
                        obj.motion_accel,
                        obj.rotation_accel,
                    ],
                )
                for obj in object_list
            ],
            osc_message("/tuio/2Dobj", ["alive", *[obj.session_id for obj in object_list]]),
            osc_message("/tuio/2Dobj", ["fseq", self._frame]),
        ]
        self._socket.sendto(osc_bundle(messages), (self.host, self.port))
        self._frame += 1


def tracks_to_tuio_objects(
    tracks: list[Track],
    frame_shape: tuple[int, ...],
    mapper: ScreenMapper | None = None,
) -> list[TuioObjectState]:
    return [
        obj
        for obj in (track_to_tuio_object(track, frame_shape, mapper) for track in tracks)
        if obj is not None
    ]


def track_to_tuio_object(
    track: Track,
    frame_shape: tuple[int, ...],
    mapper: ScreenMapper | None,
) -> TuioObjectState | None:
    if track.marker_id is None:
        return None

    points = track_points(track)
    if mapper is not None:
        points = mapper.transform_points(points)
        norm_width, norm_height = mapper.output_size
    else:
        norm_height, norm_width = frame_shape[:2]

    center = points.mean(axis=0)
    x = clamp01(float(center[0]) / max(norm_width - 1, 1))
    y = clamp01(float(center[1]) / max(norm_height - 1, 1))
    angle = marker_angle(points)

    vx, vy = track.velocity
    x_velocity = float(vx) / max(norm_width, 1)
    y_velocity = float(vy) / max(norm_height, 1)
    if mapper is not None:
        x_velocity = 0.0
        y_velocity = 0.0

    return TuioObjectState(
        session_id=track.track_id,
        symbol_id=int(track.marker_id),
        x=x,
        y=y,
        angle=angle,
        x_velocity=x_velocity,
        y_velocity=y_velocity,
    )


def track_points(track: Track) -> np.ndarray:
    if track.display_corners is not None:
        return np.asarray(track.display_corners, dtype=np.float32).reshape(-1, 2)
    if track.corners is not None:
        return np.asarray(track.corners, dtype=np.float32).reshape(-1, 2)
    bbox = track.display_bbox if track.display_bbox is not None else track.bbox
    x, y, w, h = bbox
    return np.asarray(
        [
            [x, y],
            [x + w, y],
            [x + w, y + h],
            [x, y + h],
        ],
        dtype=np.float32,
    )


def marker_angle(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(pts) < 2:
        return 0.0
    dx, dy = pts[1] - pts[0]
    return float(math.atan2(dy, dx))


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def osc_bundle(messages: list[bytes]) -> bytes:
    data = osc_string("#bundle") + struct.pack(">q", 1)
    for message in messages:
        data += struct.pack(">i", len(message)) + message
    return data


def osc_message(address: str, args: list[object]) -> bytes:
    tags = ","
    values = b""
    for arg in args:
        if isinstance(arg, str):
            tags += "s"
            values += osc_string(arg)
        elif isinstance(arg, int):
            tags += "i"
            values += struct.pack(">i", arg)
        elif isinstance(arg, float):
            tags += "f"
            values += struct.pack(">f", arg)
        else:
            raise TypeError(f"Unsupported OSC argument type: {type(arg)!r}")
    return osc_string(address) + osc_string(tags) + values


def osc_string(value: str) -> bytes:
    raw = value.encode("utf-8") + b"\x00"
    padding = (4 - len(raw) % 4) % 4
    return raw + (b"\x00" * padding)
