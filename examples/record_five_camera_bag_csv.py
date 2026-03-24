"""
Running Instructions:

python examples/record_five_camera_bag_csv.py
"""

import csv
import time
from contextlib import suppress
from pathlib import Path
from threading import Lock

from pyorbbecsdk import (
    Config,
    Context,
    OBError,
    OBFormat,
    OBMultiDeviceSyncMode,
    OBSensorType,
    Pipeline,
    RecordDevice,
)

# Local setup: 2 devices total (1 PRIMARY + 1 SECONDARY).
# Prefer serial numbers over enumeration order for a stable local rig.
# Fill in the real serial numbers for your setup. `device_index` is only used
# as a fallback if `serial_number` is left blank.
LOCAL_DEVICES = [
    {
        "device_index": 0,
        "serial_number": "CL8S16100G1",
        "name": "primary",
        "mode": OBMultiDeviceSyncMode.PRIMARY,
        "color_delay_us": 0,
        "depth_delay_us": 0,
        "trigger_to_image_delay_us": 0,
        "trigger_out_enable": True,
        "trigger_out_delay_us": 0,
        "frames_per_trigger": 1,
    },
    {
        "device_index": 1,
        "serial_number": "CL8RC5301ZH",
        "name": "secondary_1",
        "mode": OBMultiDeviceSyncMode.SECONDARY,
        "color_delay_us": 200,
        "depth_delay_us": 200,
        "trigger_to_image_delay_us": 0,
        "trigger_out_enable": True,
        "trigger_out_delay_us": 0,
        "frames_per_trigger": 1,
    },
]

OUT_DIR = Path("multi_bag/two_camera_pc")
OUT_DIR.mkdir(parents=True, exist_ok=True)

COLOR_PROFILE_CANDIDATES = [
    (0, 0, OBFormat.MJPG, 15),
    (0, 0, OBFormat.RGB, 15),
    (0, 0, OBFormat.MJPG, 30),
    (0, 0, OBFormat.RGB, 30),
]

DEPTH_PROFILE_CANDIDATES = [
    (0, 0, OBFormat.Y16, 15),
    (0, 0, OBFormat.Y16, 30),
]


def pick_video_profile(pipeline: Pipeline, sensor_type: OBSensorType, candidates):
    profile_list = pipeline.get_stream_profile_list(sensor_type)
    for width, height, fmt, fps in candidates:
        try:
            return profile_list.get_video_stream_profile(width, height, fmt, fps)
        except OBError:
            continue
    return profile_list.get_default_video_stream_profile()


def build_config(pipeline: Pipeline) -> Config:
    config = Config()
    color_profile = pick_video_profile(
        pipeline, OBSensorType.COLOR_SENSOR, COLOR_PROFILE_CANDIDATES
    )
    depth_profile = pick_video_profile(
        pipeline, OBSensorType.DEPTH_SENSOR, DEPTH_PROFILE_CANDIDATES
    )
    config.enable_stream(color_profile)
    config.enable_stream(depth_profile)
    return config


def resolve_device(device_list, spec):
    serial_number = spec["serial_number"].strip()
    if serial_number:
        return device_list.get_device_by_serial_number(serial_number)
    return device_list.get_device_by_index(spec["device_index"])


def apply_sync_config(device, spec) -> None:
    sync_config = device.get_multi_device_sync_config()
    sync_config.mode = spec["mode"]
    sync_config.color_delay_us = spec["color_delay_us"]
    sync_config.depth_delay_us = spec["depth_delay_us"]
    sync_config.trigger_to_image_delay_us = spec["trigger_to_image_delay_us"]
    sync_config.trigger_out_enable = spec["trigger_out_enable"]
    sync_config.trigger_out_delay_us = spec["trigger_out_delay_us"]
    sync_config.frames_per_trigger = spec["frames_per_trigger"]
    try:
        device.set_multi_device_sync_config(sync_config)
    except OBError as exc:
        raise RuntimeError(
            "Failed to apply multi-device sync config for "
            f"{device.get_device_info().get_serial_number()} "
            f"(mode={spec['mode'].name}). "
            "This often means a previous run did not release the device cleanly. "
            "Unplug/replug the cameras once, then retry."
        ) from exc


def main() -> None:
    ctx = Context()
    device_list = ctx.query_devices()
    available = device_list.get_count()
    if available < len(LOCAL_DEVICES):
        raise RuntimeError(f"Expected at least {len(LOCAL_DEVICES)} devices, found {available}")

    cameras = []
    for spec in LOCAL_DEVICES:
        device = resolve_device(device_list, spec)
        info = device.get_device_info()
        serial_number = info.get_serial_number()
        apply_sync_config(device, spec)

        pipeline = Pipeline(device)
        config = build_config(pipeline)

        bag_path = OUT_DIR / f"{spec['name']}_dev{spec['device_index']}_{serial_number}.bag"
        ts_csv_path = bag_path.with_suffix(".save_timestamps.csv")

        recorder = RecordDevice(device, str(bag_path))
        ts_fh = open(ts_csv_path, "w", newline="", encoding="utf-8")
        ts_writer = csv.writer(ts_fh)
        ts_writer.writerow(
            [
                "frame_idx",
                "callback_timestamp_ns",
                "color_timestamp_usec",
                "color_system_timestamp_usec",
                "depth_timestamp_usec",
                "depth_system_timestamp_usec",
            ]
        )

        cameras.append(
            {
                "spec": spec,
                "device": device,
                "serial_number": serial_number,
                "pipeline": pipeline,
                "config": config,
                "recorder": recorder,
                "bag_path": bag_path,
                "ts_csv_path": ts_csv_path,
                "ts_fh": ts_fh,
                "ts_writer": ts_writer,
                "frame_idx": 0,
                "saved_frames": 0,
                "lock": Lock(),
            }
        )

    start_order = [c for c in cameras if c["spec"]["mode"] != OBMultiDeviceSyncMode.PRIMARY]
    start_order += [c for c in cameras if c["spec"]["mode"] == OBMultiDeviceSyncMode.PRIMARY]

    def make_callback(camera):
        def on_new_frame(frames):
            if frames is None:
                return

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if color_frame is None or depth_frame is None:
                print(
                    f"skipping frame on serial={camera['serial_number']} "
                    f"name={camera['spec']['name']} because color/depth is missing"
                )
                return

            callback_timestamp_ns = time.time_ns()
            with camera["lock"]:
                camera["ts_writer"].writerow(
                    [
                        camera["frame_idx"],
                        callback_timestamp_ns,
                        color_frame.get_timestamp_us(),
                        color_frame.get_system_timestamp_us(),
                        depth_frame.get_timestamp_us(),
                        depth_frame.get_system_timestamp_us(),
                    ]
                )
                camera["frame_idx"] += 1
                camera["saved_frames"] += 1
                if camera["saved_frames"] % 30 == 0:
                    camera["ts_fh"].flush()

        return on_new_frame

    for camera in start_order:
        camera["pipeline"].start(camera["config"], make_callback(camera))
        print(
            f"started serial={camera['serial_number']} "
            f"name={camera['spec']['name']} mode={camera['spec']['mode'].name} "
            f"color_delay_us={camera['spec']['color_delay_us']} "
            f"depth_delay_us={camera['spec']['depth_delay_us']} "
            f"bag={camera['bag_path']}"
        )

    ctx.enable_multi_device_sync(60000)

    try:
        print("Recording BAG on local 2-camera rig with sidecar timestamps...")
        print("Press CTRL-C to stop.")
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("Stopping 2-camera recording")
    finally:
        for camera in cameras:
            with suppress(Exception):
                camera["ts_fh"].flush()
            with suppress(Exception):
                camera["ts_fh"].close()

            # Release the recorder first so the SDK can finalize the .bag
            # before the underlying device stream is torn down.
            with suppress(Exception, KeyboardInterrupt):
                camera["recorder"] = None

            with suppress(Exception, KeyboardInterrupt):
                camera["pipeline"].stop()

            print(
                f"saved {camera['bag_path']} frames={camera['saved_frames']} "
                f"timestamps={camera['ts_csv_path']}"
            )


if __name__ == "__main__":
    main()
