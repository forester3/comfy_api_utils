COMFYUI_URL = "http://127.0.0.1:8188"
CSS_PATH = "/content/comfy_api_utils/uistyle.css"
RESO_PATH = "/content/comfy_api_utils/resolutions.json"
CKPT_DIR = "/content/ComfyUI/models/checkpoints"
LOG_FILE = "/content/comfyui_debug_log.txt"

import os
import json
import random
from datetime import datetime
import pytz
import gradio as gr
from functools import partial

import comfy_api as capi
import comfy_task_manager as ctm
import comfy_log as clg
import importlib
importlib.reload(capi)
importlib.reload(ctm)
importlib.reload(clg)

def get_ckpt_choices(dir):
    try:
        files = [f for f in os.listdir(dir) if f.endswith(".safetensors") or f.endswith(".ckpt")]
        return files
    except Exception as e:
        # UIに返す前にログに出力
        print(f"Error reading checkpoint dir '{dir}': {e}")
        return []

def refresh_ckpt_list():
    choices = get_ckpt_choices(CKPT_DIR)
    if not choices:
        print("No checkpoints found in:", CKPT_DIR)
        return gr.update(choices=[], value=None)
    return gr.update(choices=choices, value=choices[0])

def update_seed_enable(mode):
    return gr.update(interactive=(mode == "Fix"))

def seed_initial_load(seed_mode):
    seed_upd = gr.update(interactive=(seed_mode == "Fix"))
    choices = get_ckpt_choices(CKPT_DIR)
    dropdown_upd = gr.update(choices=choices, value=choices[0] if choices else None)
    return seed_upd, dropdown_upd

def input_values( workflow_sta, node_ids_sta, presets_sta,
                  positive_prompt, negative_prompt, selected_ckpt, height, width, preset_name,
                  seed_mode, seed_input, steps, cfg, denoise, stop_at_clip_layer):

    if not ctm.is_last_saved():
        inputs = {"message" :"image not saved yet!" }
        return json.dumps(inputs, indent=4)

    preset = presets_sta.get(preset_name, {})
    sampler = preset.get("sampler", "unknown")
    scheduler = preset.get("scheduler", "unknown")
    seed = seed_input if seed_mode == "Fix" else random.randint(0, 2**32 - 1)

    prompt_id = capi.generate_image_with_api(COMFYUI_URL, workflow_sta, node_ids_sta,
                                        positive_prompt, negative_prompt,
                                        seed, width, height, steps, cfg,
                                        sampler, scheduler, denoise,
                                        selected_ckpt, stop_at_clip_layer,
                                        filename_prefix=datetime.now(pytz.timezone('Asia/Tokyo')).strftime("%y%m%d"))

    inputs = {  "prompt id"       : prompt_id,
                "positive prompt" : positive_prompt,
                "negative prompt" : negative_prompt,
                "checkpoint"      : selected_ckpt,
                "width"           : width,
                "height"          : height,
                "sampler"         : sampler,
                "scheduler"       : scheduler,
                "seed"            : seed,
                "steps"           : steps,
                "cfg"             : cfg,
                "denoise"         : denoise,
                "stop_at_clip"    : stop_at_clip_layer }

    ctm.init_task_status(prompt_id, COMFYUI_URL)
    return json.dumps(inputs, indent=4)

def size_on_select(resolution_str):
    w, h = map(int, resolution_str.split("x"))
    return w, h, f"{w} x {h}"

def size_on_adjust(w, h):
    return f"{w} x {h}"


# Gradio UI
def build_gradio_ui(workflow, node_ids, presets):

    ctm.clear_task_status()             # task_status全削除

    with open(RESO_PATH, "r") as f:
        resolutions = json.load(f)

    resolution_options = [f"{r['width']}x{r['height']}" for r in resolutions]

    def_size_idx = 5
    def_size_str = resolution_options[def_size_idx]
    def_w = resolutions[def_size_idx]["width"]
    def_h = resolutions[def_size_idx]["height"]

    with open(CSS_PATH, "r") as f:
        css = f.read()

    with gr.Blocks(css=css) as demo:
        workflow_sta = gr.State(value=workflow)
        node_ids_sta = gr.State(value=node_ids)
        presets_sta = gr.State(value=presets)
        with gr.Row():
            with gr.Tabs():
                with gr.Tab("Load ckpt"):
                    ckpt_dropdown = gr.Dropdown(choices=[], label="Checkpoints")
                    refresh_btn = gr.Button("🔄 Refresh", elem_id="ref_button")
                    selected_ckpt = gr.Textbox(label="Selected Checkpoint", interactive=False)

                with gr.Tab("Pos-P"):
                    positive_prompt = gr.Textbox(show_label=False, value="beautiful girl, soft lighting, 8k", lines=8)

                with gr.Tab("Neg-P"):
                    negative_prompt = gr.Textbox(show_label=False, value="blurry, deformed, extra limbs", lines=8)

                with gr.Tab("Size"):
                    size_dropdown = gr.Dropdown(label="Preset(w/h)", choices=resolution_options, 
                                                value=def_size_str, interactive=True)
                    with gr.Row():
                        with gr.Column():
                            width = gr.Slider(512, 1600, value=def_w, step=16, label="width", interactive=True)
                        with gr.Column():
                            height = gr.Slider(512, 1408, value=def_h, step=16, label="height", interactive=True)
                    size_output = gr.Textbox(label="Output(w/h) ", value=f"{def_w} x {def_h}")

                    size_dropdown.change(fn=size_on_select, inputs=size_dropdown, outputs=[width, height, size_output])
                    width.change(fn=size_on_adjust, inputs=[width, height], outputs=size_output)
                    height.change(fn=size_on_adjust, inputs=[width, height], outputs=size_output)

                with gr.Tab("KSamp"):
                    with gr.Row():
                        with gr.Row():
                            seed_mode = gr.Radio(["Random", "Fix"], label="Seed Mode", value="Random")
                            seed_input = gr.Number(value=1234, label="Seed Fix", interactive=False)
                        denoise = gr.Slider(0.0, 1.0, value=1.0, step=0.01, label="Denoise", interactive=True)
                    with gr.Row():
                        with gr.Column():
                            preset_dropdown = gr.Dropdown(label="Sampler/Scheduler", choices=list(presets.keys()),
                                                value=list(presets.keys())[0], type="value" )
                        with gr.Column():
                            steps = gr.Slider(10, 50, value=20, step=1, label="Steps")
                        with gr.Column():
                            cfg = gr.Slider(1.0, 15.0, value=7.5, step=0.1, label="CFG Scale")

                with gr.Tab("Clp-Stp"):
                    with gr.Row():
                        stop_at_clip_layer = gr.Slider(-11, -1, value=-1, step=1, label="CLIP Stop", interactive=True)
                        gr.Markdown("")
                        gr.Markdown("")
            with gr.Column():
                run_button = gr.Button("Generate Image", elem_id="run_button")
                with gr.Tabs():
                    with gr.Tab("Inputs"):
                        inputs_txt = gr.Textbox(show_label=False, lines=20, interactive=False)
                    with gr.Tab("ComfyUI Log"):
                        log_box = gr.Textbox(show_label=False, lines=20, interactive=False)
                    with gr.Tab("WebSocket Log"):
                        ws_box = gr.Textbox(show_label=False, lines=20, interactive=False)

        timer_log = gr.Timer(2)
        timer_log.tick(fn=partial(clg.get_log_text, LOG_FILE), outputs=log_box)
        timer_ws_log = gr.Timer(1)
        timer_ws_log.tick(fn=capi.get_ws_log_text, outputs=ws_box)

        refresh_btn.click(refresh_ckpt_list, inputs=None, outputs=ckpt_dropdown)
        ckpt_dropdown.change(lambda x: x, inputs=ckpt_dropdown, outputs=selected_ckpt)
        seed_mode.change(update_seed_enable, inputs=seed_mode, outputs=seed_input)
        demo.load(seed_initial_load, inputs=seed_mode, outputs=[seed_input, ckpt_dropdown])

        run_button.click(input_values,
                  inputs=[  workflow_sta, node_ids_sta, presets_sta,
                            positive_prompt, negative_prompt, selected_ckpt, height, width, preset_dropdown,
                            seed_mode, seed_input, steps, cfg, denoise, stop_at_clip_layer],
                  outputs=[inputs_txt, ])
    return demo
