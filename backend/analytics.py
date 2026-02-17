import os
import json
import math
import subprocess
from typing import Dict, Any, List, Tuple

import cv2
import numpy as np

import matplotlib
matplotlib.use("Agg")  # server-safe backend for threads
import matplotlib.pyplot as plt

from ultralytics import YOLO


def _set_job(
    jobs: Dict[str, Any],
    job_id: str,
    *,
    status=None,
    progress=None,
    message=None,
    error=None,
    outputs=None,
):
    job = jobs[job_id]
    if status is not None:
        job["status"] = status
    if progress is not None:
        job["progress"] = int(progress)
    if message is not None:
        job["message"] = message
    if error is not None:
        job["error"] = str(error)
        job["status"] = "error"
    if outputs is not None:
        job["outputs"] = outputs


def _color_for_id(track_id: int) -> Tuple[int, int, int]:
    np.random.seed(track_id * 9973)
    c = np.random.randint(60, 255, size=3).tolist()
    return int(c[0]), int(c[1]), int(c[2])


def _ffmpeg_exists() -> bool:
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return True
    except Exception:
        return False


def _to_h264_mp4(src_avi: str, dst_mp4: str):
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            src_avi,
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            dst_mp4,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


def process_video_job(job_id: str, jobs: Dict[str, Any]):
    """
    Outputs:
      - trajectory_overlay.mp4
      - heatmap_overlay.mp4
      - heatmap.png
      - speed_chart.png
      - congestion_overlay.png
      - congestion_instant.png
      - summary.json
    """
    try:
        job_dir = jobs[job_id]["job_dir"]
        input_path = os.path.join(job_dir, "input.mp4")

        out_traj_avi = os.path.join(job_dir, "trajectory_overlay.avi")
        out_traj_mp4 = os.path.join(job_dir, "trajectory_overlay.mp4")

        out_heat_vid_avi = os.path.join(job_dir, "heatmap_overlay.avi")
        out_heat_vid_mp4 = os.path.join(job_dir, "heatmap_overlay.mp4")

        out_heatmap = os.path.join(job_dir, "heatmap.png")
        out_speed = os.path.join(job_dir, "speed_chart.png")

        out_congestion = os.path.join(job_dir, "congestion_overlay.png")
        out_congestion_instant = os.path.join(job_dir, "congestion_instant.png")

        out_summary = os.path.join(job_dir, "summary.json")

        _set_job(jobs, job_id, status="processing", progress=10, message="Loading model...")

        model = YOLO("yolov8n.pt")
        try:
            model.fuse()
        except Exception:
            pass

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError("Could not open video.")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        # Performance (CPU)
        frame_stride = 6
        infer_max_width = 640
        imgsz = 416
        conf = 0.45

        # Congestion (density-first)
        grid_size = 120
        density_threshold = 2
        window_seconds = 1.0
        slow_ratio = 0.65  # only for severity label

        # Heatmap overlay video (adaptive memory)
        heat_alpha = 0.45
        base_heat_decay = 0.90
        crowd_heat_decay = 0.97
        heat_impulse = 3.0
        heat_blur_sigma = 9

        # Drawing
        bbox_thickness = 2
        trail_thickness = 5
        trail_len = 18
        font_scale = 0.6
        font_thickness = 2

        # Alerts
        alert_cells_threshold = 1
        alert_cooldown_sec = 2.5

        scale = 1.0
        w, h = w0, h0
        if w0 > infer_max_width:
            scale = infer_max_width / float(w0)
            w = int(w0 * scale)
            h = int(h0 * scale)

        window_frames = max(1, int(window_seconds * fps / frame_stride))
        grid_h = (h + grid_size - 1) // grid_size
        grid_w = (w + grid_size - 1) // grid_size

        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        traj_writer = cv2.VideoWriter(out_traj_avi, fourcc, float(fps / frame_stride), (w, h))
        heat_writer = cv2.VideoWriter(out_heat_vid_avi, fourcc, float(fps / frame_stride), (w, h))
        if not traj_writer.isOpened() or not heat_writer.isOpened():
            raise RuntimeError("Failed to open video writers. Try a different codec.")

        trajectories: Dict[int, List[Tuple[int, float, float]]] = {}
        last_pos: Dict[int, Tuple[float, float]] = {}
        last_t: Dict[int, float] = {}

        heat_cumulative = np.zeros((h, w), dtype=np.float32)
        heat_recent = np.zeros((h, w), dtype=np.float32)

        window_speeds: List[float] = []
        current_window_speeds: List[float] = []

        congestion_counts = np.zeros((grid_h, grid_w), dtype=np.int32)
        last_congestion_mask = np.zeros((grid_h, grid_w), dtype=np.uint8)
        last_congestion_severity = np.zeros((grid_h, grid_w), dtype=np.float32)
        window_idx = 0

        alerts: List[Dict[str, Any]] = []
        last_alert_time = -1e9
        congestion_active_last_window = False

        _set_job(jobs, job_id, progress=25, message="Detecting & tracking...")

        processed = 0
        frame_i = 0

        while True:
            ok, frame0 = cap.read()
            if not ok:
                break

            frame_i += 1
            if frame_i % frame_stride != 0:
                continue

            processed += 1
            t_sec = frame_i / fps

            if total_frames > 0:
                frac = frame_i / total_frames
                prog = 30 + 30 * frac
                _set_job(jobs, job_id, progress=prog, message="Tracking + analytics...")

            if scale != 1.0:
                frame = cv2.resize(frame0, (w, h), interpolation=cv2.INTER_AREA)
            else:
                frame = frame0

            overlay = frame.copy()

            decay = crowd_heat_decay if congestion_active_last_window else base_heat_decay
            heat_recent *= decay

            cell_ids = [[set() for _ in range(grid_w)] for __ in range(grid_h)]
            cell_speeds = [[[] for _ in range(grid_w)] for __ in range(grid_h)]

            results = model.track(
                frame,
                persist=True,
                classes=[0],
                verbose=False,
                imgsz=imgsz,
                conf=conf,
            )
            r0 = results[0]

            if r0.boxes is not None and r0.boxes.id is not None:
                ids = r0.boxes.id.cpu().numpy().astype(int)
                boxes = r0.boxes.xyxy.cpu().numpy()

                for track_id, (x1, y1, x2, y2) in zip(ids, boxes):
                    cx = float((x1 + x2) / 2.0)
                    cy = float((y1 + y2) / 2.0)

                    trajectories.setdefault(track_id, []).append((frame_i, cx, cy))

                    ix = int(np.clip(cx, 0, w - 1))
                    iy = int(np.clip(cy, 0, h - 1))
                    heat_cumulative[iy, ix] += 1.0
                    heat_recent[iy, ix] += heat_impulse

                    v = None
                    if track_id in last_pos:
                        px, py = last_pos[track_id]
                        dt = t_sec - last_t.get(track_id, t_sec)
                        if dt > 1e-6:
                            v = math.hypot(cx - px, cy - py) / dt
                            current_window_speeds.append(v)

                    last_pos[track_id] = (cx, cy)
                    last_t[track_id] = t_sec

                    gi = min(grid_h - 1, max(0, int(cy // grid_size)))
                    gj = min(grid_w - 1, max(0, int(cx // grid_size)))
                    cell_ids[gi][gj].add(track_id)
                    if v is not None:
                        cell_speeds[gi][gj].append(v)

                    color = _color_for_id(track_id)
                    cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), color, bbox_thickness)
                    cv2.putText(
                        overlay,
                        f"ID {track_id}",
                        (int(x1), max(18, int(y1) - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        color,
                        font_thickness,
                    )

                    pts = trajectories[track_id][-trail_len:]
                    for k in range(2, len(pts), 2):
                        _, x_prev, y_prev = pts[k - 2]
                        _, x_cur, y_cur = pts[k]
                        cv2.line(overlay, (int(x_prev), int(y_prev)), (int(x_cur), int(y_cur)), color, trail_thickness)

            if processed % window_frames == 0:
                if current_window_speeds:
                    window_speeds.append(float(np.mean(current_window_speeds)))
                else:
                    window_speeds.append(0.0)
                current_window_speeds = []

                global_avg_speed = float(np.mean(window_speeds)) if window_speeds else 30.0
                slow_thresh = slow_ratio * global_avg_speed

                last_congestion_mask[:] = 0
                last_congestion_severity[:] = 0.0

                congested_cells = 0
                max_density = 0
                max_severity = 0.0

                for i in range(grid_h):
                    for j in range(grid_w):
                        density = len(cell_ids[i][j])
                        if density <= 0:
                            continue
                        max_density = max(max_density, density)

                        if density >= density_threshold:
                            last_congestion_mask[i, j] = 1
                            congestion_counts[i, j] += 1
                            congested_cells += 1

                            avg_v = float(np.mean(cell_speeds[i][j])) if cell_speeds[i][j] else global_avg_speed
                            slow_bonus = 1.0 if avg_v <= slow_thresh else 0.0
                            dens_norm = min(1.0, density / 6.0)
                            sev = 0.75 * dens_norm + 0.25 * slow_bonus
                            last_congestion_severity[i, j] = float(np.clip(sev, 0.0, 1.0))
                            max_severity = max(max_severity, last_congestion_severity[i, j])

                congestion_active_last_window = congested_cells > 0

                if congested_cells >= alert_cells_threshold and (t_sec - last_alert_time) >= alert_cooldown_sec:
                    alerts.append(
                        {
                            "time_sec": round(t_sec, 2),
                            "window_sec": float(window_seconds),
                            "congested_cells": int(congested_cells),
                            "max_cell_density": int(max_density),
                            "max_severity_0_1": float(round(max_severity, 3)),
                            "message": "Crowd congestion detected (density-based).",
                        }
                    )
                    last_alert_time = t_sec

                window_idx += 1

            heat_blur = cv2.GaussianBlur(
                heat_recent,
                (0, 0),
                sigmaX=heat_blur_sigma,
                sigmaY=heat_blur_sigma,
            )
            mx = float(heat_blur.max())
            if mx > 1e-6:
                heat_norm = (heat_blur / mx * 255).astype(np.uint8)
            else:
                heat_norm = np.zeros_like(heat_blur, dtype=np.uint8)

            heat_color = cv2.applyColorMap(heat_norm, cv2.COLORMAP_JET)
            heat_overlay = cv2.addWeighted(frame, 1 - heat_alpha, heat_color, heat_alpha, 0)

            alert_active = bool(alerts) and abs(alerts[-1]["time_sec"] - t_sec) <= window_seconds
            if alert_active:
                cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 255), -1)
                cv2.putText(
                    overlay,
                    "⚠ CROWD CONGESTION",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 255, 255),
                    3,
                )

                cv2.rectangle(heat_overlay, (0, 0), (w, 60), (0, 0, 255), -1)
                cv2.putText(
                    heat_overlay,
                    "⚠ CROWD CONGESTION",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 255, 255),
                    3,
                )

            cv2.putText(
                heat_overlay,
                f"Realtime Heatmap (decay={decay:.2f})",
                (20, h - 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            traj_writer.write(overlay)
            heat_writer.write(heat_overlay)

        cap.release()
        traj_writer.release()
        heat_writer.release()

        _set_job(jobs, job_id, progress=65, message="Rendering outputs...")

        if _ffmpeg_exists():
            _to_h264_mp4(out_traj_avi, out_traj_mp4)
            _to_h264_mp4(out_heat_vid_avi, out_heat_vid_mp4)
        else:
            out_traj_mp4 = out_traj_avi
            out_heat_vid_mp4 = out_heat_vid_avi

        heat_blur_full = cv2.GaussianBlur(heat_cumulative, (0, 0), sigmaX=12, sigmaY=12)
        heat_norm_full = heat_blur_full / (heat_blur_full.max() + 1e-6)

        plt.figure()
        plt.imshow(heat_norm_full, cmap="hot")
        plt.colorbar(label="Movement Intensity")
        plt.title("Movement Heatmap (Cumulative)")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_heatmap, dpi=200)
        plt.close()

        if window_speeds:
            smin, smax = float(min(window_speeds)), float(max(window_speeds))
            if smax - smin > 1e-6:
                pace = [100.0 * (s - smin) / (smax - smin) for s in window_speeds]
            else:
                pace = [0.0 for _ in window_speeds]
        else:
            pace = []

        plt.figure()
        plt.plot(pace)
        plt.title("Movement Pace Over Time (Normalized)")
        plt.xlabel("Window Index")
        plt.ylabel("Relative Pace (0–100)")
        plt.tight_layout()
        plt.savefig(out_speed, dpi=200)
        plt.close()

        _set_job(jobs, job_id, progress=80, message="Computing congestion overlays...")

        congestion_score = congestion_counts.astype(np.float32)
        if congestion_score.max() > 0:
            congestion_score /= congestion_score.max()

        overlay_img = np.zeros((h, w, 3), dtype=np.uint8)
        for i in range(grid_h):
            for j in range(grid_w):
                s = float(congestion_score[i, j])
                if s <= 0:
                    continue
                x1, y1 = j * grid_size, i * grid_size
                x2, y2 = min(w - 1, (j + 1) * grid_size), min(h - 1, (i + 1) * grid_size)
                intensity = int(255 * (0.35 + 0.65 * s))
                cv2.rectangle(overlay_img, (x1, y1), (x2, y2), (0, 0, intensity), -1)

        cv2.putText(
            overlay_img,
            "Congestion Recurrence (red = repeated high-density zones)",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
        )
        cv2.imwrite(out_congestion, overlay_img)

        instant_img = np.zeros((h, w, 3), dtype=np.uint8)
        for i in range(grid_h):
            for j in range(grid_w):
                if last_congestion_mask[i, j] == 0:
                    continue
                x1, y1 = j * grid_size, i * grid_size
                x2, y2 = min(w - 1, (j + 1) * grid_size), min(h - 1, (i + 1) * grid_size)
                sev = float(last_congestion_severity[i, j])
                intensity = int(255 * (0.4 + 0.6 * sev))
                cv2.rectangle(instant_img, (x1, y1), (x2, y2), (0, 0, intensity), -1)

        cv2.putText(
            instant_img,
            "Instant Congestion (last window)",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        cv2.imwrite(out_congestion_instant, instant_img)

        flat = [(int(congestion_counts[i, j]), i, j) for i in range(grid_h) for j in range(grid_w)]
        flat.sort(reverse=True, key=lambda x: x[0])

        bottlenecks = []
        for rank, (count, i, j) in enumerate(flat[:5], start=1):
            if count <= 0:
                continue
            x = int(j * grid_size + grid_size / 2)
            y = int(i * grid_size + grid_size / 2)
            bottlenecks.append(
                {
                    "rank": rank,
                    "grid_cell": {"row": i, "col": j},
                    "approx_location_px": {"x": x, "y": y},
                    "crowded_windows": count,
                    "note": "Repeated high-density zone (workflow bottleneck candidate).",
                }
            )

        track_stats = []
        for tid, pts in trajectories.items():
            if len(pts) < 2:
                continue
            dist = 0.0
            for k in range(1, len(pts)):
                _, x0, y0 = pts[k - 1]
                _, x1, y1 = pts[k]
                dist += math.hypot(x1 - x0, y1 - y0)
            duration_sec = (pts[-1][0] - pts[0][0]) / fps
            avg_speed = dist / max(duration_sec, 1e-6)
            track_stats.append(
                {
                    "track_id": int(tid),
                    "points": int(len(pts)),
                    "duration_sec": float(duration_sec),
                    "avg_speed_px_per_sec": float(avg_speed),
                }
            )

        summary = {
            "video": {
                "fps": float(fps),
                "input_width": int(w0),
                "input_height": int(h0),
                "output_width": int(w),
                "output_height": int(h),
                "frame_stride": int(frame_stride),
                "processed_frames": int(processed),
                "window_seconds": float(window_seconds),
                "grid_size_px": int(grid_size),
                "heat_decay_base": float(base_heat_decay),
                "heat_decay_crowd": float(crowd_heat_decay),
            },
            "definition": {
                "congestion_primary": "High-density grid cells per time window (density >= threshold)",
                "severity": "Density + optional slowdown signal",
                "density_threshold": int(density_threshold),
            },
            "counts": {
                "unique_tracks": int(len(trajectories)),
                "windows": int(window_idx),
                "alerts": int(len(alerts)),
            },
            "speed": {
                "window_avg_speed_px_per_sec": window_speeds,
                "window_relative_pace_0_100": pace,
                "overall_avg_speed_px_per_sec": float(np.mean(window_speeds)) if window_speeds else 0.0,
            },
            "alerts": alerts,
            "bottlenecks_top5": bottlenecks,
            "track_stats": track_stats[:100],
        }

        with open(out_summary, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        outputs = {
            "trajectory_overlay.mp4": f"/api/download/{job_id}/trajectory_overlay.mp4",
            "heatmap_overlay.mp4": f"/api/download/{job_id}/heatmap_overlay.mp4",
            "heatmap.png": f"/api/download/{job_id}/heatmap.png",
            "speed_chart.png": f"/api/download/{job_id}/speed_chart.png",
            "congestion_overlay.png": f"/api/download/{job_id}/congestion_overlay.png",
            "congestion_instant.png": f"/api/download/{job_id}/congestion_instant.png",
            "summary.json": f"/api/download/{job_id}/summary.json",
        }

        _set_job(jobs, job_id, status="complete", progress=100, message="Complete", outputs=outputs)

    except Exception as e:
        _set_job(jobs, job_id, error=e, progress=100, message="Failed")
