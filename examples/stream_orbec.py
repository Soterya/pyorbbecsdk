"""
Run from an environment where `pyorbbecsdk` is installed.

Example:
python scripts/stream_orbbec_device1.py
"""

import cv2
import numpy as np
from pyorbbecsdk import Config, Context, OBError, OBFormat, OBSensorType, Pipeline


ESC_KEY = 27
DEVICE_INDEX = 0


def frame_to_bgr_image(color_frame):
    width = color_frame.get_width()
    height = color_frame.get_height()
    color_format = color_frame.get_format()
    data = np.asanyarray(color_frame.get_data())

    if color_format == OBFormat.RGB:
        image = np.resize(data, (height, width, 3))
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if color_format == OBFormat.BGR:
        image = np.resize(data, (height, width, 3))
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if color_format == OBFormat.YUYV:
        image = np.resize(data, (height, width, 2))
        return cv2.cvtColor(image, cv2.COLOR_YUV2BGR_YUYV)
    if color_format == OBFormat.MJPG:
        return cv2.imdecode(data, cv2.IMREAD_COLOR)

    print(f"Unsupported color format: {color_format}")
    return None


def main() -> int:
    ctx = Context()
    device_list = ctx.query_devices()
    count = device_list.get_count()
    if count <= DEVICE_INDEX:
        print(f"Device index {DEVICE_INDEX} is not available. Found {count} device(s).")
        return 1

    device = device_list.get_device_by_index(DEVICE_INDEX)
    info = device.get_device_info()
    print(
        f"Streaming device {DEVICE_INDEX}: "
        f"name={info.get_name()} serial={info.get_serial_number()} pid={info.get_pid()}"
    )

    pipeline = Pipeline(device)
    config = Config()

    try:
        profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        try:
            color_profile = profile_list.get_video_stream_profile(640, 0, OBFormat.RGB, 15)
        except OBError:
            color_profile = profile_list.get_default_video_stream_profile()
        config.enable_stream(color_profile)
        print(
            "Using color profile: "
            f"width={color_profile.get_width()} "
            f"height={color_profile.get_height()} "
            f"format={color_profile.get_format()} "
            f"fps={color_profile.get_fps()}"
        )
    except Exception as exc:
        print(f"Failed to configure color stream for device {DEVICE_INDEX}: {exc}")
        return 1

    pipeline.start(config)
    print("Press 'q' or ESC to quit.")
    consecutive_timeouts = 0
    displayed_frames = 0

    try:
        while True:
            frames = pipeline.wait_for_frames(1000)
            if frames is None:
                consecutive_timeouts += 1
                if consecutive_timeouts % 5 == 0:
                    print(f"Timed out waiting for frames {consecutive_timeouts} time(s)")
                continue

            consecutive_timeouts = 0
            color_frame = frames.get_color_frame()
            if color_frame is None:
                print("Received frameset without a color frame")
                continue

            color_image = frame_to_bgr_image(color_frame)
            if color_image is None:
                print(f"Failed to convert color frame with format {color_frame.get_format()}")
                continue

            displayed_frames += 1
            if displayed_frames == 1:
                print(
                    "First color frame: "
                    f"width={color_frame.get_width()} "
                    f"height={color_frame.get_height()} "
                    f"format={color_frame.get_format()} "
                    f"timestamp_us={color_frame.get_timestamp_us()}"
                )

            cv2.imshow("Orbbec Device 1", color_image)
            key = cv2.waitKey(1)
            if key in (ord("q"), ESC_KEY):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        pipeline.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
