import argparse
import os
from pathlib import Path

import cv2
import numpy as np

from pyorbbecsdk import Config, OBSensorType, OBFormat, Pipeline, PlaybackDevice
from utils import frame_to_bgr_image


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export RGB frames and depth maps from Orbbec .bag recordings."
    )
    parser.add_argument(
        "bag_files",
        nargs="+",
        help="One or more .bag files recorded with RecordDevice.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Directory for exported files. Defaults to <bag_stem>_export next to each bag.",
    )
    parser.add_argument(
        "--every-n",
        type=int,
        default=1,
        help="Save every Nth frameset. Default: 1",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap on saved frames per bag.",
    )
    return parser.parse_args()


def build_output_dirs(base_dir: Path):
    color_dir = base_dir / "color"
    depth_raw_dir = base_dir / "depth_raw"
    depth_vis_dir = base_dir / "depth_vis"
    color_dir.mkdir(parents=True, exist_ok=True)
    depth_raw_dir.mkdir(parents=True, exist_ok=True)
    depth_vis_dir.mkdir(parents=True, exist_ok=True)
    return color_dir, depth_raw_dir, depth_vis_dir


def depth_frame_to_images(depth_frame):
    if depth_frame is None:
        return None, None
    if depth_frame.get_format() != OBFormat.Y16:
        print(f"Skipping unsupported depth format: {depth_frame.get_format()}")
        return None, None

    depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
    depth_data = depth_data.reshape((depth_frame.get_height(), depth_frame.get_width()))

    depth_vis = cv2.normalize(
        depth_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )
    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
    return depth_data, depth_vis


def enable_available_streams(config: Config, playback: PlaybackDevice):
    sensor_list = playback.get_sensor_list()
    available_types = {
        sensor_list.get_type_by_index(i) for i in range(sensor_list.get_count())
    }

    if OBSensorType.COLOR_SENSOR in available_types:
        config.enable_stream(OBSensorType.COLOR_SENSOR)
    if OBSensorType.DEPTH_SENSOR in available_types:
        config.enable_stream(OBSensorType.DEPTH_SENSOR)


def export_bag(bag_path: Path, output_root: str | None, every_n: int, max_frames: int | None):
    if not bag_path.exists():
        print(f"Missing bag file: {bag_path}")
        return

    base_dir = (
        Path(output_root) / bag_path.stem
        if output_root
        else bag_path.with_name(f"{bag_path.stem}_export")
    )
    color_dir, depth_raw_dir, depth_vis_dir = build_output_dirs(base_dir)

    playback = PlaybackDevice(str(bag_path))
    pipeline = Pipeline(playback)
    config = Config()
    enable_available_streams(config, playback)
    pipeline.start(config)

    print(f"Exporting {bag_path} -> {base_dir}")
    saved_frames = 0
    seen_frames = 0
    idle_loops = 0

    try:
        while True:
            try:
                frames = pipeline.wait_for_frames(100)
            except Exception:
                frames = None

            if frames is None:
                idle_loops += 1
                if idle_loops > 50:
                    break
                continue

            idle_loops = 0
            seen_frames += 1
            if seen_frames % every_n != 0:
                continue

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if color_frame is None and depth_frame is None:
                continue

            if color_frame is not None:
                color_image = frame_to_bgr_image(color_frame)
                if color_image is not None:
                    color_path = color_dir / (
                        f"color_{saved_frames:06d}_{color_frame.get_timestamp_us()}.png"
                    )
                    cv2.imwrite(str(color_path), color_image)

            if depth_frame is not None:
                depth_raw, depth_vis = depth_frame_to_images(depth_frame)
                if depth_raw is not None:
                    raw_path = depth_raw_dir / (
                        f"depth_{saved_frames:06d}_{depth_frame.get_timestamp_us()}.png"
                    )
                    vis_path = depth_vis_dir / (
                        f"depth_{saved_frames:06d}_{depth_frame.get_timestamp_us()}.png"
                    )
                    cv2.imwrite(str(raw_path), depth_raw)
                    cv2.imwrite(str(vis_path), depth_vis)

            saved_frames += 1
            if max_frames is not None and saved_frames >= max_frames:
                break
    finally:
        pipeline.stop()
        playback = None

    print(f"Saved {saved_frames} framesets from {bag_path.name}")


def main():
    args = parse_args()
    for bag_file in args.bag_files:
        export_bag(Path(bag_file), args.output_root, args.every_n, args.max_frames)


if __name__ == "__main__":
    main()
