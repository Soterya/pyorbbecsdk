"""
Running Instructions:

python scripts/record_subordinate_pc_csv.py
"""

import csv
import time
from pathlib import Path

from pyk4a import (
    ColorResolution,
    Config,
    DepthMode,
    FPS,
    ImageFormat,
    K4ATimeoutException,
    PyK4A,
    PyK4ARecord,
    WiredSyncMode,
    connected_device_count,
)
from pyk4a.errors import K4AException

# Local Orbbec K4A-wrapper setup: one local master and one local subordinate.
# IMPORTANT: delays must be globally unique across all subordinates in the full rig.
LOCAL_DEVICES = [
    {
        "device_id": 0,
        "expected_serial": "CL8S16100G1",
        "name": "master",
        "mode": WiredSyncMode.MASTER,
        "sub_delay_usec": 0,
        "depth_mode": DepthMode.WFOV_2X2BINNED,
        "timestamp_attr": "depth_timestamp_usec",
    },
    {
        "device_id": 1,
        "expected_serial": "CL8RC5301ZH",
        "name": "subordinate",
        "mode": WiredSyncMode.SUBORDINATE,
        "sub_delay_usec": 200,
        "depth_mode": DepthMode.OFF,
        "timestamp_attr": "color_timestamp_usec",
    },
]

OUT_DIR = Path("multi_mkv/subordinate_pc")
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_CONFIG = dict(
    color_format=ImageFormat.COLOR_MJPG,
    color_resolution=ColorResolution.RES_1080P,
    camera_fps=FPS.FPS_15,
    synchronized_images_only=False,
)


def build_config(mode: WiredSyncMode, sub_delay_usec: int, depth_mode: DepthMode) -> Config:
    if mode == WiredSyncMode.MASTER:
        return Config(**BASE_CONFIG, depth_mode=depth_mode, wired_sync_mode=WiredSyncMode.MASTER)
    return Config(
        **BASE_CONFIG,
        depth_mode=depth_mode,
        wired_sync_mode=WiredSyncMode.SUBORDINATE,
        subordinate_delay_off_master_usec=sub_delay_usec,
    )


def main() -> None:
    available = connected_device_count()
    if available < len(LOCAL_DEVICES):
        raise RuntimeError(f"Expected at least {len(LOCAL_DEVICES)} devices on subordinate PC, found {available}")

    cameras = []
    for spec in LOCAL_DEVICES:
        cfg = build_config(spec["mode"], spec["sub_delay_usec"], spec["depth_mode"])
        dev = PyK4A(config=cfg, device_id=spec["device_id"])
        cameras.append({"spec": spec, "device": dev, "config": cfg})

    # Start subordinate first so it is ready and waiting when the local master begins driving sync.
    start_order = [c for c in cameras if c["spec"]["mode"] == WiredSyncMode.SUBORDINATE]
    start_order += [c for c in cameras if c["spec"]["mode"] == WiredSyncMode.MASTER]

    for c in start_order:
        try:
            c["device"].start()
        except K4AException as exc:
            raise RuntimeError(
                f"Failed to start device_id={c['spec']['device_id']} "
                f"name={c['spec']['name']} mode={c['spec']['mode'].name}. "
                "The color camera may already be in use or blocked by Windows camera privacy settings."
            ) from exc
        actual_serial = c["device"].serial
        expected_serial = c["spec"]["expected_serial"]
        if actual_serial != expected_serial:
            raise RuntimeError(
                f"Device mapping mismatch for {c['spec']['name']}: "
                f"device_id={c['spec']['device_id']} expected serial {expected_serial}, got {actual_serial}"
            )
        print(
            f"started device_id={c['spec']['device_id']} serial={actual_serial} "
            f"name={c['spec']['name']} mode={c['spec']['mode'].name} "
            f"sub_delay_usec={c['spec']['sub_delay_usec']} sync_jack={c['device'].sync_jack_status}"
        )

    outputs = []
    for c in cameras:
        serial = c["device"].serial
        mkv_path = OUT_DIR / f"{c['spec']['name']}_dev{c['spec']['device_id']}_{serial}.mkv"
        ts_csv_path = mkv_path.with_suffix(".save_timestamps.csv")

        record = PyK4ARecord(path=mkv_path, config=c["config"], device=c["device"])
        record.create()

        ts_fh = open(ts_csv_path, "w", newline="", encoding="utf-8")
        ts_writer = csv.writer(ts_fh)
        ts_writer.writerow(["frame_idx", "save_timestamp_ns", "device_timestamp_usec"])

        outputs.append(
            {
                "camera": c,
                "record": record,
                "mkv_path": mkv_path,
                "ts_csv_path": ts_csv_path,
                "ts_fh": ts_fh,
                "ts_writer": ts_writer,
                "frame_idx": 0,
            }
        )

    try:
        print("Recording mixed MKV on subordinate PC (master depth ON, subordinate depth OFF)...")
        print("Press CTRL-C to stop.")
        while True:
            for out in outputs:
                try:
                    cap = out["camera"]["device"].get_capture(timeout=10000)
                except K4ATimeoutException:
                    print(
                        f"timed out waiting for frame on device {out['camera']['spec']['device_id']} "
                        f"serial={out['camera']['device'].serial}"
                    )
                    continue

                if cap.color is None:
                    print(
                        f"skipping frame on device {out['camera']['spec']['device_id']} "
                        "because color is missing"
                    )
                    continue
                if out["camera"]["spec"]["timestamp_attr"] == "depth_timestamp_usec" and cap.depth is None:
                    print(
                        f"skipping frame on device {out['camera']['spec']['device_id']} "
                        "because depth is missing"
                    )
                    continue
                # Host timestamp taken right before persisting this capture.
                save_timestamp_ns = time.time_ns()
                device_timestamp_usec = getattr(cap, out["camera"]["spec"]["timestamp_attr"])
                out["ts_writer"].writerow([out["frame_idx"], save_timestamp_ns, device_timestamp_usec])
                out["record"].write_capture(cap)
                out["frame_idx"] += 1
    except KeyboardInterrupt:
        print("Stopping subordinate PC recording")
    finally:
        for out in outputs:
            try:
                if out["record"].captures_count > 0:
                    out["record"].flush()
            finally:
                out["record"].close()
                out["ts_fh"].flush()
                out["ts_fh"].close()
                out["camera"]["device"].stop()
            print(
                f"saved {out['mkv_path']} frames={out['record'].captures_count} "
                f"timestamps={out['ts_csv_path']}"
            )


if __name__ == "__main__":
    main()
