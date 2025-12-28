import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import base64
import io
import cv2
import numpy as np
from PIL import Image
import gradio as gr
import torch
import base64
from pycocotools import mask as mask_util
import copy
import re
import imageio
import torchvision

try:
    from segment_anything import sam_model_registry, SamPredictor
except ImportError as e:
    raise RuntimeError("segment-anything not found: pip install git+https://github.com/facebookresearch/segment-anything.git") from e

# Paths
SAM_CHECKPOINT = "./checkpoints/sam_vit_h_4b8939.pth"
SAM_TYPE = "vit_h"

# Video generation constants
T = 81
H = 480
W = 832
if_color = 0
if_mask = 1
if_vr = 0
if_special_corr = 0
controls = ['gaussian_channel', 'vae_channel']
vae_channel = 'point'

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {device}")

# Load SAM model
sam = sam_model_registry[SAM_TYPE](checkpoint=SAM_CHECKPOINT)
sam.to(device=device)
predictor = SamPredictor(sam)

# WorldCanvas pipeline (lazy loading)
pipe = None

def load_worldcanvas_pipeline():
    global pipe
    if pipe is not None:
        return pipe

    from diffsynth import save_video
    from diffsynth.pipelines.WorldCanvas import WorldCanvasPipeline, ModelConfig

    print("[INFO] Loading WorldCanvas pipeline...")
    pipe = WorldCanvasPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(path="./checkpoints/Wan2.2-T2V-A14B/models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
            ModelConfig(path="./checkpoints/WorldCanvas_dit/WorldCanvas/high_model.safetensors", offload_device="cpu"),
            ModelConfig(path="./checkpoints/WorldCanvas_dit/WorldCanvas/low_model.safetensors", offload_device="cpu"),
            ModelConfig(path="./checkpoints/Wan2.2-T2V-A14B/Wan2.1_VAE.pth", offload_device="cpu"),
        ],
    )
    pipe.enable_vram_management()
    pipe.to(device)
    print("[INFO] WorldCanvas pipeline loaded.")
    return pipe

def load_and_resize_image(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image path not found: {path}")
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise RuntimeError(f"Could not read image: {path}")
    h, w = img_bgr.shape[:2]
    if w > h:
        new_h, new_w = 480, 832
    else:
        new_h, new_w = 832, 480
    img_bgr_resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    img_rgb_resized = cv2.cvtColor(img_bgr_resized, cv2.COLOR_BGR2RGB)
    return img_rgb_resized

def set_predictor_image(img_rgb):
    predictor.set_image(img_rgb)

def compute_sam_mask(img_rgb, pos_points, neg_points):
    pts = []
    labels = []
    for (x, y) in pos_points:
        pts.append([x, y])
        labels.append(1)
    for (x, y) in neg_points:
        pts.append([x, y])
        labels.append(0)
    if len(pts) == 0:
        return None, None
    point_coords = np.array(pts, dtype=np.float32)
    point_labels = np.array(labels, dtype=np.int32)
    masks, scores, logits = predictor.predict(point_coords=point_coords, point_labels=point_labels, multimask_output=True)
    best_idx = int(np.argmax(scores))
    best_mask = masks[best_idx]
    return best_mask.astype(np.uint8), float(scores[best_idx])

def encode_mask_to_base64_png(mask_cluster):
    mask_cluster = mask_util.encode(np.asfortranarray(mask_cluster.astype(np.uint8)))
    mask_cluster['counts'] = base64.b64encode(mask_cluster['counts']).decode('ascii')
    return mask_cluster

def decode_png_base64_to_mask(b64str):
    tmp = copy.deepcopy(b64str)
    tmp['counts'] = base64.b64decode(tmp['counts'].encode('ascii'))
    mask_cluster = mask_util.decode(tmp).astype(bool).astype(np.uint8)
    return mask_cluster

def draw_overlay(
    base_img_rgb,
    pos_points,
    neg_points,
    mask_uint8,
    draw_points,
    track81,
    vis81,
    erase_buffer,
    prev_results
):
    if base_img_rgb is None:
        return None
    canvas_bgr = cv2.cvtColor(base_img_rgb.copy(), cv2.COLOR_RGB2BGR)

    if prev_results:
        for item in prev_results:
            try:
                b64 = item.get("mask_cluster", None)
                if not b64:
                    continue
                prev_mask = decode_png_base64_to_mask(b64)
                if prev_mask.shape[:2] != canvas_bgr.shape[:2]:
                    continue
                color_prev_mask = (255, 0, 0)
                alpha_prev = 0.25
                mask_bool = prev_mask.astype(bool)
                overlay = canvas_bgr.copy()
                overlay[mask_bool] = (overlay[mask_bool] * (1 - alpha_prev) + np.array(color_prev_mask) * alpha_prev).astype(np.uint8)
                canvas_bgr = overlay
                contours, _ = cv2.findContours(prev_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(canvas_bgr, contours, -1, (255, 0, 0), 2)
            except Exception:
                continue

    if prev_results:
        for item in prev_results:
            trk = item.get("tracking", None)
            vis = item.get("tracking_vis_value", None)
            if not trk or not vis or len(trk) != 81 or len(vis) != 81:
                continue
            for i in range(80):
                x1, y1 = trk[i]
                x2, y2 = trk[i + 1]
                if x1 >= 0 and y1 >= 0 and x2 >= 0 and y2 >= 0:
                    color = (255, 255, 0) if (vis[i] == 1 and vis[i + 1] == 1) else (180, 180, 180)
                    cv2.line(canvas_bgr, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            for i in range(81):
                x, y = trk[i]
                if x >= 0 and y >= 0:
                    color = (255, 255, 0) if vis[i] == 1 else (180, 180, 180)
                    cv2.circle(canvas_bgr, (int(x), int(y)), 3, color, -1)

    for (x, y) in pos_points:
        cv2.circle(canvas_bgr, (int(x), int(y)), 6, (0, 255, 0), -1)
        cv2.circle(canvas_bgr, (int(x), int(y)), 9, (0, 100, 0), 2)
    for (x, y) in neg_points:
        cv2.circle(canvas_bgr, (int(x), int(y)), 6, (0, 0, 255), -1)
        cv2.circle(canvas_bgr, (int(x), int(y)), 9, (0, 0, 100), 2)

    if mask_uint8 is not None:
        color = (0, 255, 0)
        alpha = 0.35
        mask_bool = mask_uint8.astype(bool)
        overlay = canvas_bgr.copy()
        overlay[mask_bool] = (overlay[mask_bool] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
        canvas_bgr = overlay
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas_bgr, contours, -1, (0, 200, 0), 2)

    if draw_points and len(draw_points) > 0:
        for i, (x, y) in enumerate(draw_points):
            cv2.circle(canvas_bgr, (int(x), int(y)), 4, (255, 255, 0), -1)
            if i > 0:
                x0, y0 = draw_points[i - 1]
                cv2.line(canvas_bgr, (int(x0), int(y0)), (int(x), int(y)), (255, 255, 0), 2)

    if track81 is not None and vis81 is not None:
        for i in range(80):
            x1, y1 = track81[i]
            x2, y2 = track81[i + 1]
            if x1 >= 0 and y1 >= 0 and x2 >= 0 and y2 >= 0:
                color = (0, 220, 0) if (vis81[i] == 1 and vis81[i + 1] == 1) else (0, 0, 220)
                cv2.line(canvas_bgr, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        for i in range(81):
            x, y = track81[i]
            if x >= 0 and y >= 0:
                color = (0, 220, 0) if vis81[i] == 1 else (0, 0, 220)
                cv2.circle(canvas_bgr, (int(x), int(y)), 3, color, -1)

    if erase_buffer and len(erase_buffer) > 0:
        for (x, y) in erase_buffer:
            cv2.circle(canvas_bgr, (int(x), int(y)), 6, (0, 255, 255), 2)

    return cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB)

def interpolate_polyline(points, n_samples, endpoint=False):
    if n_samples <= 0:
        return []
    if not points or len(points) == 0:
        return []
    if len(points) == 1:
        return [tuple(float(x) for x in points[0]) for _ in range(n_samples)]

    num_segments = len(points) - 1
    total_points = n_samples
    points_per_segment = []
    remaining = total_points
    used = 0

    for i in range(num_segments):
        if i == num_segments - 1:
            pts_in_seg = remaining
        else:
            pts_in_seg = max(1, (total_points - used + num_segments - i - 1) // (num_segments - i))
        points_per_segment.append(pts_in_seg)
        used += pts_in_seg
        remaining -= pts_in_seg

    out = []
    pts = np.array(points, dtype=np.float32)

    for seg_idx in range(num_segments):
        start = pts[seg_idx]
        end = pts[seg_idx + 1]
        count = int(points_per_segment[seg_idx])

        if count == 1:
            out.append((float(start[0]), float(start[1])))
        else:
            ts = np.linspace(0, 1, count)
            seg_points = (1 - ts[:, None]) * start + ts[:, None] * end
            seg_tuples = [(float(p[0]), float(p[1])) for p in seg_points]
            if seg_idx == 0:
                out.extend(seg_tuples)
            else:
                out.extend(seg_tuples[1:])

    if len(out) > n_samples:
        out = out[:n_samples]
    elif len(out) < n_samples:
        last_point = out[-1] if out else tuple(float(x) for x in points[-1])
        while len(out) < n_samples:
            out.append(last_point)

    return out

def find_nearest_track_index(click_xy, track81, valid_range=None):
    x_click, y_click = click_xy
    best_idx = None
    best_dist2 = None
    lo, hi = (0, 80) if valid_range is None else valid_range
    lo = max(0, lo)
    hi = min(80, hi)
    for i in range(lo, hi + 1):
        x, y = track81[i]
        if x >= 0 and y >= 0:
            dx = x - x_click
            dy = y - y_click
            d2 = dx * dx + dy * dy
            if (best_dist2 is None) or (d2 < best_dist2):
                best_dist2 = d2
                best_idx = i
    return best_idx

def on_load_image(path, state):
    if path is None:
        return (gr.update(), "Please upload an image first.", state)
    try:
        img_rgb = load_and_resize_image(path)
        set_predictor_image(img_rgb)
        state = {
            "base_img": img_rgb,
            "pos_points": [],
            "neg_points": [],
            "mask_uint8": None,
            "mask_confirmed": False,
            "draw_points": [],
            "track81": None,
            "vis81": None,
            "st": 0,
            "et": 81,
            "stage": "mask",
            "erase_mode": False,
            "erase_buffer": [],
            "results": state.get("results", []),
            "image_path": path
        }
        overlay = draw_overlay(state["base_img"], [], [], None, [], None, None, [], state["results"])
        return (
            gr.update(value=overlay, visible=True),
            "Image loaded and resized.",
            state
        )
    except Exception as e:
        return (gr.update(value=None, visible=True), f"Load failed: {e}", state)

def on_click_image(evt: gr.SelectData, point_type, state):
    if state is None or state.get("base_img") is None:
        return (None, "Please load an image first.", state)

    if evt is None or ((getattr(evt, "x", None) is None or getattr(evt, "y", None) is None) and (getattr(evt, "index", None) is None)):
        overlay = draw_overlay(state["base_img"], state["pos_points"], state["neg_points"],
                               state["mask_uint8"], state["draw_points"],
                               state["track81"], state["vis81"], state["erase_buffer"],
                               state["results"])
        return (overlay, "Click event parsing failed.", state)

    if getattr(evt, "x", None) is not None and getattr(evt, "y", None) is not None:
        x, y = float(evt.x), float(evt.y)
    else:
        x, y = float(evt.index[0]), float(evt.index[1])

    H0, W0 = state["base_img"].shape[:2]
    x = float(np.clip(x, 0, W0 - 1))
    y = float(np.clip(y, 0, H0 - 1))

    stage = state.get("stage", "mask")
    msg = ""

    if stage == "mask":
        if point_type == "Positive Point":
            state["pos_points"].append((x, y))
        else:
            state["neg_points"].append((x, y))
        if not state["mask_confirmed"]:
            mask, score = compute_sam_mask(state["base_img"], state["pos_points"], state["neg_points"])
            state["mask_uint8"] = mask
            msg = f"Points: Pos {len(state['pos_points'])}, Neg {len(state['neg_points'])}; Mask score approx {score:.3f}" if mask is not None else "Please add at least one point"

    elif stage == "traj" and not state.get("erase_mode", False):
        state["draw_points"].append((x, y))
        msg = f"Trajectory click points: {len(state['draw_points'])}"

    elif state.get("erase_mode", False):
        state["erase_buffer"].append((x, y))
        if len(state["erase_buffer"]) >= 2:
            if state["track81"] is not None and state["vis81"] is not None:
                st = state.get("st", 0)
                et = state.get("et", 81)
                idx1 = find_nearest_track_index(state["erase_buffer"][0], state["track81"], (st, et - 1))
                idx2 = find_nearest_track_index(state["erase_buffer"][1], state["track81"], (st, et - 1))
                if idx1 is not None and idx2 is not None:
                    lo = min(idx1, idx2)
                    hi = max(idx1, idx2)
                    for k in range(lo, hi + 1):
                        state["vis81"][k] = 0
                    msg = f"Visibility segment [{lo}, {hi}] erased."
                else:
                    msg = "Cannot locate track index."
            else:
                msg = "Trajectory not generated."
            state["erase_buffer"] = []
        else:
            msg = "Erase start point recorded. Click end point."

    overlay = draw_overlay(state["base_img"], state["pos_points"], state["neg_points"],
                           state["mask_uint8"], state["draw_points"],
                           state["track81"], state["vis81"], state["erase_buffer"],
                           state["results"])
    return (overlay, msg, state)

def undo_last_mask_point(state):
    if state.get("stage") != "mask":
        overlay = draw_overlay(state["base_img"], state["pos_points"], state["neg_points"],
                               state["mask_uint8"], state["draw_points"],
                               state["track81"], state["vis81"], state["erase_buffer"],
                               state["results"])
        return (overlay, "Not in mask annotation stage.", state)
    if len(state["neg_points"]) > 0:
        state["neg_points"].pop()
    elif len(state["pos_points"]) > 0:
        state["pos_points"].pop()
    if not state["mask_confirmed"]:
        if len(state["pos_points"]) + len(state["neg_points"]) > 0:
            mask, score = compute_sam_mask(state["base_img"], state["pos_points"], state["neg_points"])
            state["mask_uint8"] = mask
            msg = f"Undo successful; mask score approx {score:.3f}"
        else:
            state["mask_uint8"] = None
            msg = "No points left, mask cleared."
    else:
        msg = "Mask confirmed, cannot auto-update."
    overlay = draw_overlay(state["base_img"], state["pos_points"], state["neg_points"],
                           state["mask_uint8"], state["draw_points"],
                           state["track81"], state["vis81"], state["erase_buffer"],
                           state["results"])
    return (overlay, msg, state)

def clear_mask_points(state):
    if state.get("stage") != "mask":
        overlay = draw_overlay(state["base_img"], state["pos_points"], state["neg_points"],
                               state["mask_uint8"], state["draw_points"],
                               state["track81"], state["vis81"], state["erase_buffer"],
                               state["results"])
        return (overlay, "Not in mask annotation stage.", state)
    state["pos_points"] = []
    state["neg_points"] = []
    if not state["mask_confirmed"]:
        state["mask_uint8"] = None
        msg = "SAM points and mask cleared."
    else:
        msg = "Mask confirmed, retaining the confirmed mask."
    overlay = draw_overlay(state["base_img"], state["pos_points"], state["neg_points"],
                           state["mask_uint8"], state["draw_points"],
                           state["track81"], state["vis81"], state["erase_buffer"],
                           state["results"])
    return (overlay, msg, state)

def confirm_mask(state):
    if state.get("stage") != "mask":
        return (None, "Not in mask annotation stage.", state)
    if state.get("mask_uint8") is None:
        return (None, "Please add points and generate a mask first.", state)
    state["mask_confirmed"] = True
    state["stage"] = "traj"
    msg = "Mask confirmed. Now draw trajectory points."
    overlay = draw_overlay(state["base_img"], state["pos_points"], state["neg_points"],
                           state["mask_uint8"], state["draw_points"],
                           state["track81"], state["vis81"], state["erase_buffer"],
                           state["results"])
    return (overlay, msg, state)

def build_trajectory(st, et, state):
    if state.get("stage") != "traj":
        return (None, "Not in trajectory drawing stage.", state)
    st = int(max(0, min(81, st)))
    et = int(max(0, min(81, et)))
    if et <= st:
        return (None, "et must be greater than st.", state)
    if len(state.get("draw_points", [])) == 0:
        return (None, "Please click points on the image first.", state)

    n = et - st
    samples = interpolate_polyline(state["draw_points"], n_samples=n, endpoint=False)
    track81 = [[-1.0, -1.0] for _ in range(81)]
    vis81 = [0 for _ in range(81)]
    for i in range(n):
        idx = st + i
        if 0 <= idx < 81:
            x, y = samples[i]
            track81[idx] = [float(x), float(y)]
            vis81[idx] = 1
    state["track81"] = track81
    state["vis81"] = vis81
    state["st"] = st
    state["et"] = et

    overlay = draw_overlay(state["base_img"], state["pos_points"], state["neg_points"],
                           state["mask_uint8"], state["draw_points"],
                           state["track81"], state["vis81"], state["erase_buffer"],
                           state["results"])
    msg = f"Trajectory generated: [{st}, {et})"
    return (overlay, msg, state)

def undo_last_traj_point(state):
    if state.get("stage") != "traj":
        return (None, "Not in trajectory drawing stage.", state)
    if len(state["draw_points"]) > 0:
        state["draw_points"].pop()
    overlay = draw_overlay(state["base_img"], state["pos_points"], state["neg_points"],
                           state["mask_uint8"], state["draw_points"],
                           state["track81"], state["vis81"], state["erase_buffer"],
                           state["results"])
    msg = f"Trajectory click points: {len(state['draw_points'])}"
    return (overlay, msg, state)

def clear_traj_points(state):
    if state.get("stage") != "traj":
        return (None, "Not in trajectory drawing stage.", state)
    state["draw_points"] = []
    state["track81"] = None
    state["vis81"] = None
    overlay = draw_overlay(state["base_img"], state["pos_points"], state["neg_points"],
                           state["mask_uint8"], state["draw_points"],
                           state["track81"], state["vis81"], state["erase_buffer"],
                           state["results"])
    return (overlay, "Trajectory cleared.", state)

def toggle_erase_mode(erase_mode, state):
    state["erase_mode"] = bool(erase_mode)
    state["stage"] = "traj" if not state["erase_mode"] else "erase"
    state["erase_buffer"] = []
    overlay = draw_overlay(state["base_img"], state["pos_points"], state["neg_points"],
                           state["mask_uint8"], state["draw_points"],
                           state["track81"], state["vis81"], state["erase_buffer"],
                           state["results"])
    msg = "Erase Mode ON" if state["erase_mode"] else "Erase Mode OFF"
    return (overlay, msg, state)

def confirm_current_object(obj_id, obj_text, state):
    if not state.get("mask_confirmed", False) or state.get("mask_uint8") is None:
        return (None, "Please confirm mask first.", state)
    if state.get("track81") is None or state.get("vis81") is None:
        return (None, "Please generate trajectory first.", state)

    id_str = (obj_id or "").strip()
    text_str = (obj_text or "").strip()
    if id_str == "" or text_str == "":
        overlay = draw_overlay(state["base_img"], state["pos_points"], state["neg_points"],
                               state["mask_uint8"], state["draw_points"],
                               state["track81"], state["vis81"], state["erase_buffer"],
                               state["results"])
        return (overlay, "Please fill Object ID and Text Description.", state)

    mask_b64 = encode_mask_to_base64_png(state["mask_uint8"])
    r = int(max(state["mask_uint8"].sum() * 320 // 399360, 5))
    result_item = {
        "tracking": state["track81"],
        "tracking_vis_value": state["vis81"],
        "r": r,
        "id": int(id_str),
        "mask_cluster": mask_b64,
        "text": text_str,
    }
    results = state.get("results", [])
    results.append(result_item)
    state["results"] = results

    state["pos_points"] = []
    state["neg_points"] = []
    state["mask_uint8"] = None
    state["mask_confirmed"] = False
    state["draw_points"] = []
    state["track81"] = None
    state["vis81"] = None
    state["erase_mode"] = False
    state["erase_buffer"] = []
    state["stage"] = "mask"

    overlay = draw_overlay(state["base_img"], state["pos_points"], state["neg_points"],
                           state["mask_uint8"], state["draw_points"],
                           state["track81"], state["vis81"], state["erase_buffer"],
                           state["results"])
    msg = f"Trajectory added (Total: {len(results)}). Ready for next object."
    return (overlay, msg, state)

def prepare_json_data(results):
    """Convert results to format expected by WorldCanvas pipeline"""
    re_ = {}

    id_caption_map = {}
    for tp in results:
        if tp.get("text") and tp["text"] != 'None':
            id_caption_map[str(tp["id"])] = tp["text"]

    if len(id_caption_map) == 0:
        return None

    text_prompt = ''
    id_caption_order = {}
    color_idx = 0
    appears_ids = []

    re_['tracking_points'] = []
    re_['vis'] = []
    re_['rs'] = []
    re_['ids'] = []
    re_['point_masks'] = []

    for tp in results:
        if "tracking" in tp:
            re_['tracking_points'].append(tp['tracking'])
            if (tp['tracking'][0][0] < 0 or tp['tracking'][0][0] >= W) or (tp['tracking'][0][1] < 0 or tp['tracking'][0][1] >= H):
                appears_ids.append(str(tp['id']))
            re_['vis'].append(tp['tracking_vis_value'])
            re_['rs'].append(tp['r'])
            re_['ids'].append(tp['id'])
            re_['point_masks'].append(tp["mask_cluster"])

    for tid, ca in id_caption_map.items():
        if tid not in appears_ids:
            text_prompt += f'Object {color_idx+1}: '
        else:
            text_prompt += f'Object {color_idx+1} appears: '
        text_prompt += re.sub(r'\s+', ' ', ca).strip()
        text_prompt += os.linesep
        id_caption_order[tid] = color_idx
        color_idx += 1

    re_['text_prompt'] = text_prompt[:-1] if text_prompt else ''
    re_['id_caption_order'] = id_caption_order

    return re_

def crop_and_resize(image, target_height, target_width):
    width, height = image.size
    scale = max(target_width / width, target_height / height)
    image = torchvision.transforms.functional.resize(
        image,
        (round(height*scale), round(width*scale)),
        interpolation=torchvision.transforms.InterpolationMode.BILINEAR
    )
    image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
    return image

def generate_video(seed, num_steps, state, progress=gr.Progress()):
    """Generate video from current annotations"""
    from diffsynth import save_video

    results = state.get("results", [])
    if not results:
        return None, "No trajectories defined. Please add at least one trajectory."

    image_path = state.get("image_path", "")
    if not image_path or not os.path.exists(image_path):
        return None, "Image path not found. Please reload the image."

    progress(0.1, desc="Preparing data...")

    # Deep copy results to avoid modifying state
    results_copy = copy.deepcopy(results)

    # Prepare JSON data
    sample_json = prepare_json_data(results_copy)
    if sample_json is None:
        return None, "No valid captions found. Please add text descriptions."

    progress(0.2, desc="Loading WorldCanvas model (first time takes a while)...")

    # Load pipeline
    try:
        pipe = load_worldcanvas_pipeline()
    except Exception as e:
        return None, f"Failed to load model: {e}"

    progress(0.3, desc="Loading input image...")

    # Load input image
    input_image = Image.open(image_path).convert('RGB')
    input_image = crop_and_resize(input_image, H, W)

    current_prompt = sample_json['text_prompt']

    progress(0.4, desc=f"Generating video with seed {seed}, {int(num_steps)} steps...")

    try:
        video = pipe(
            prompt=current_prompt,
            negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
            seed=int(seed),
            json_data=sample_json,
            tiled=True,
            height=H,
            width=W,
            num_frames=T,
            input_image=input_image,
            switch_DiT_boundary=0.9,
            num_inference_steps=int(num_steps),
            if_color=if_color,
            if_mask=if_mask,
            if_special_corr=if_special_corr,
            if_vr=if_vr,
            controls=controls,
            vae_channel=vae_channel,
        )
    except Exception as e:
        torch.cuda.empty_cache()
        return None, f"Video generation failed: {e}"

    progress(0.9, desc="Saving video...")

    # Save video
    os.makedirs("outputs", exist_ok=True)
    output_path = f"outputs/video_seed{seed}.mp4"

    pil_frames = [Image.fromarray(frame) for frame in np.stack(video, axis=0)]
    save_video(pil_frames, output_path, fps=16, quality=5)

    # Clean up GPU memory for next generation
    del video
    del pil_frames
    torch.cuda.empty_cache()

    progress(1.0, desc="Done!")

    return output_path, f"Video saved to {output_path}"

# Build Gradio UI
with gr.Blocks(
    title="WorldCanvas - Trajectory to Video",
    css="""
    #canvas_wrap { overflow: auto; }
    #canvas_img img { max-width: none !important; width: auto !important; height: auto !important; }
    """
) as demo:
    gr.Markdown("# WorldCanvas - Trajectory Annotation & Video Generation")

    state = gr.State({
        "base_img": None,
        "pos_points": [],
        "neg_points": [],
        "mask_uint8": None,
        "mask_confirmed": False,
        "draw_points": [],
        "track81": None,
        "vis81": None,
        "st": 0,
        "et": 81,
        "stage": "mask",
        "erase_mode": False,
        "erase_buffer": [],
        "results": [],
        "image_path": ""
    })

    with gr.Row():
        img_upload = gr.Image(label="Upload Image", type="filepath", sources=["upload", "clipboard"], scale=4)
        btn_load = gr.Button("Load Image", scale=1)

    with gr.Row():
        with gr.Column(scale=2):
            with gr.Row(elem_id="canvas_wrap"):
                canvas = gr.Image(
                    label="Annotation Canvas",
                    interactive=True,
                    elem_id="canvas_img"
                )
            status = gr.Textbox(label="Status", interactive=False)

        with gr.Column(scale=1):
            gr.Markdown("### Stage 1: SAM Mask")
            point_type = gr.Radio(choices=["Positive Point", "Negative Point"], value="Positive Point", label="Point Type")
            with gr.Row():
                btn_undo_mask_pt = gr.Button("Undo")
                btn_clear_mask_pts = gr.Button("Clear")
                btn_confirm_mask = gr.Button("Confirm Mask", variant="primary")

            gr.Markdown("### Stage 2: Trajectory")
            with gr.Row():
                st_slider = gr.Slider(0, 80, value=0, step=1, label="Start Frame")
                et_slider = gr.Slider(1, 81, value=81, step=1, label="End Frame")
            with gr.Row():
                btn_undo_traj_pt = gr.Button("Undo")
                btn_clear_traj_pts = gr.Button("Clear")
                btn_build_traj = gr.Button("Build Trajectory", variant="primary")
            erase_mode = gr.Checkbox(value=False, label="Erase Mode")

            gr.Markdown("### Stage 3: Confirm Object")
            with gr.Row():
                obj_id = gr.Textbox(label="Object ID", placeholder="1", scale=1)
                obj_text = gr.Textbox(label="Description", placeholder="a red car moving", scale=2)
            btn_confirm_object = gr.Button("Add Object", variant="primary")

            gr.Markdown("---")
            gr.Markdown("### Stage 4: Generate Video")
            with gr.Row():
                seed_input = gr.Number(label="Seed", value=0, precision=0)
                steps_slider = gr.Slider(label="Inference Steps", minimum=10, maximum=50, value=50, step=5)
            btn_generate = gr.Button("Generate Video", variant="primary", size="lg")

    with gr.Row():
        video_output = gr.Video(label="Generated Video")
        gen_status = gr.Textbox(label="Generation Status", interactive=False)

    # Event handlers
    btn_load.click(fn=on_load_image, inputs=[img_upload, state], outputs=[canvas, status, state])
    img_upload.change(fn=on_load_image, inputs=[img_upload, state], outputs=[canvas, status, state])
    canvas.select(fn=on_click_image, inputs=[point_type, state], outputs=[canvas, status, state])

    btn_undo_mask_pt.click(undo_last_mask_point, inputs=[state], outputs=[canvas, status, state])
    btn_clear_mask_pts.click(clear_mask_points, inputs=[state], outputs=[canvas, status, state])
    btn_confirm_mask.click(confirm_mask, inputs=[state], outputs=[canvas, status, state])

    btn_undo_traj_pt.click(undo_last_traj_point, inputs=[state], outputs=[canvas, status, state])
    btn_clear_traj_pts.click(clear_traj_points, inputs=[state], outputs=[canvas, status, state])
    btn_build_traj.click(build_trajectory, inputs=[st_slider, et_slider, state], outputs=[canvas, status, state])

    erase_mode.change(toggle_erase_mode, inputs=[erase_mode, state], outputs=[canvas, status, state])

    btn_confirm_object.click(confirm_current_object, inputs=[obj_id, obj_text, state], outputs=[canvas, status, state])

    btn_generate.click(generate_video, inputs=[seed_input, steps_slider, state], outputs=[video_output, gen_status])

if __name__ == "__main__":
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=8050,
        share=False
    )
