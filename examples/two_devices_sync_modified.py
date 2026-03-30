import json
import os
import time
from datetime import datetime
from typing import List

from pyorbbecsdk import *

MAX_DEVICES = 2
WFOV_BINNED_DEPTH_WIDTH = 512
WFOV_BINNED_DEPTH_HEIGHT = 512
RECORD_FPS = 15

multi_device_sync_config = {}
recorders: List[RecordDevice | None] = [None for _ in range(MAX_DEVICES)]

config_file_path = os.path.join(
    os.path.abspath(os.path.dirname(__file__)),
    "../config/multi_device_sync_config.json",
)


def sync_mode_from_str(sync_mode_str: str) -> OBMultiDeviceSyncMode:
    sync_mode_str = sync_mode_str.upper()
    if sync_mode_str == "FREE_RUN":
        return OBMultiDeviceSyncMode.FREE_RUN
    if sync_mode_str == "STANDALONE":
        return OBMultiDeviceSyncMode.STANDALONE
    if sync_mode_str == "PRIMARY":
        return OBMultiDeviceSyncMode.PRIMARY
    if sync_mode_str == "SECONDARY":
        return OBMultiDeviceSyncMode.SECONDARY
    if sync_mode_str == "SECONDARY_SYNCED":
        return OBMultiDeviceSyncMode.SECONDARY_SYNCED
    if sync_mode_str == "SOFTWARE_TRIGGERING":
        return OBMultiDeviceSyncMode.SOFTWARE_TRIGGERING
    if sync_mode_str == "HARDWARE_TRIGGERING":
        return OBMultiDeviceSyncMode.HARDWARE_TRIGGERING
    raise ValueError(f"Invalid sync mode: {sync_mode_str}")


def read_config(config_file: str):
    global multi_device_sync_config
    with open(config_file, "r", encoding="utf-8") as f:
        config = json.load(f)
    for device in config["devices"]:
        multi_device_sync_config[device["serial_number"]] = device


def default_sync_config_for_index(index: int):
    mode = "PRIMARY" if index == 0 else "SECONDARY"
    return {
        "serial_number": "",
        "config": {
            "mode": mode,
            "depth_delay_us": 0,
            "color_delay_us": 0,
            "trigger_to_image_delay_us": 0,
            "trigger_out_enable": index == 0,
            "trigger_out_delay_us": 0,
            "frames_per_trigger": 1,
        },
    }


def make_recording_dir() -> str:
    default_dir = os.path.join(
        os.path.abspath(os.path.dirname(__file__)),
        "..",
        "recordings",
        datetime.now().strftime("two_devices_sync_%Y%m%d_%H%M%S"),
    )
    user_input = input(
        f"Output directory for .bag recordings [{default_dir}]: "
    ).strip()
    output_dir = user_input or default_dir
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def stop_recorders():
    global recorders
    for i in range(len(recorders)):
        recorders[i] = None


def select_wfov_binned_depth_profile(pipeline: Pipeline) -> VideoStreamProfile:
    depth_profiles = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
    # Femto Bolt WFOV binned target: 512x512 Y16 @ configured record FPS
    return depth_profiles.get_video_stream_profile(
        WFOV_BINNED_DEPTH_WIDTH,
        WFOV_BINNED_DEPTH_HEIGHT,
        OBFormat.Y16,
        RECORD_FPS,
    )


def enable_default_streams(pipeline: Pipeline, config: Config, serial_number: str):
    try:
        color_profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        default_color_profile: VideoStreamProfile = (
            color_profiles.get_default_video_stream_profile()
        )
        color_profile = color_profiles.get_video_stream_profile(
            default_color_profile.get_width(),
            default_color_profile.get_height(),
            default_color_profile.get_format(),
            RECORD_FPS,
        )
        config.enable_stream(color_profile)
        print(
            f"{serial_number}: color "
            f"{color_profile.get_width()}x{color_profile.get_height()} "
            f"{color_profile.get_format()} @{color_profile.get_fps()}fps"
        )
    except OBError:
        pass

    try:
        depth_profile = select_wfov_binned_depth_profile(pipeline)
        print(
            f"{serial_number}: using WFOV binned depth "
            f"{depth_profile.get_width()}x{depth_profile.get_height()} "
            f"{depth_profile.get_format()} @{depth_profile.get_fps()}fps"
        )
    except OBError:
        raise RuntimeError(
            f"{serial_number}: required depth profile "
            f"{WFOV_BINNED_DEPTH_WIDTH}x{WFOV_BINNED_DEPTH_HEIGHT} "
            f"{OBFormat.Y16} @{RECORD_FPS}fps is unavailable"
        )
    config.enable_stream(depth_profile)


def configure_device(device: Device, index: int):
    serial_number = device.get_device_info().get_serial_number()
    sync_config_json = multi_device_sync_config.get(serial_number)
    if sync_config_json is None:
        sync_config_json = default_sync_config_for_index(index)
        sync_config_json["serial_number"] = serial_number

    sync_config = device.get_multi_device_sync_config()
    sync_config.mode = sync_mode_from_str(sync_config_json["config"]["mode"])
    sync_config.color_delay_us = sync_config_json["config"]["color_delay_us"]
    sync_config.depth_delay_us = sync_config_json["config"]["depth_delay_us"]
    sync_config.trigger_out_enable = sync_config_json["config"]["trigger_out_enable"]
    sync_config.trigger_out_delay_us = sync_config_json["config"]["trigger_out_delay_us"]
    sync_config.frames_per_trigger = sync_config_json["config"]["frames_per_trigger"]
    device.set_multi_device_sync_config(sync_config)
    return serial_number


def main():
    read_config(config_file_path)
    recording_dir = make_recording_dir()

    ctx = Context()
    device_list = ctx.query_devices()
    device_count = device_list.get_count()

    if device_count == 0:
        print("No device connected")
        return
    if device_count > MAX_DEVICES:
        print("Too many devices connected")
        return

    pipelines: List[Pipeline] = []
    configs: List[Config] = []

    for i in range(device_count):
        device = device_list.get_device_by_index(i)
        serial_number = configure_device(device, i)

        pipeline = Pipeline(device)
        config = Config()
        enable_default_streams(pipeline, config, serial_number)

        bag_path = os.path.join(recording_dir, f"{i}_{serial_number}.bag")
        recorders[i] = RecordDevice(device, bag_path)

        pipelines.append(pipeline)
        configs.append(config)

    for pipeline, config in zip(pipelines, configs):
        pipeline.start(config)
        try:
            pipeline.enable_frame_sync()
        except OBError:
            pass

    ctx.enable_multi_device_sync(60000)

    print("Recording. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        for pipeline in pipelines:
            pipeline.stop()
        stop_recorders()


if __name__ == "__main__":
    main()
