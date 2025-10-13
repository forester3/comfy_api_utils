import urllib.request
import urllib.parse
import websocket
import requests
import json
import time
import threading
import traceback
import os
import comfy_task_manager as ctm

OUTPUT_DIR = "/content/ComfyUI/output"

def make_ws_url(url):
    host_port = url.replace("http://", "")
    return f"ws://{host_port}/ws"

def queue_prompt(base_url, prompt_workflow):
    payload = {"prompt": prompt_workflow}
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    req = urllib.request.Request(f"{base_url}/prompt", data=data, headers=headers)
    try:
        response = urllib.request.urlopen(req)
        return json.loads(response.read().decode("utf-8"))
    except Exception as e:
        print(f"Error queueing prompt: {e}")
        return None

def generate_image_with_api(
    base_url, workflow_json, node_ids, positive_prompt, negative_prompt,
    seed, width, height, steps, cfg, sampler_name, scheduler, denoise,
    model_name, stop_at_clip_layer, filename_prefix="ComfyUI_API"  ):

    prompt = workflow_json.copy()
    prompt[node_ids["KSampler"]]["inputs"].update({
        "seed": seed, "steps": steps, "cfg": cfg,
        "sampler_name": sampler_name, "scheduler": scheduler, "denoise": denoise })
    prompt[node_ids["EmptyLatentImage"]]["inputs"].update({"width": width, "height": height})

    # --- SDXL対応: text_l, text_g 両方に設定 ---
    positive_node = prompt[node_ids["PositivePrompt_TextEncode"]]["inputs"]
    negative_node = prompt[node_ids["NegativePrompt_TextEncode"]]["inputs"]

    positive_node["text"] = positive_prompt
    positive_node["text_g"] = positive_prompt
    positive_node["text_l"] = positive_prompt

    negative_node["text"] = negative_prompt
    negative_node["text_g"] = negative_prompt
    negative_node["text_l"] = negative_prompt
    
    prompt[node_ids["SaveImage"]]["inputs"]["filename_prefix"] = filename_prefix
    prompt[node_ids["CheckpointLoaderSimple"]]["inputs"]["ckpt_name"] = model_name
    prompt[node_ids["CLIPSetLastLayer"]]["inputs"]["stop_at_clip_layer"] = stop_at_clip_layer

    prompt_info = queue_prompt(base_url, prompt)        # Queue the prompt
    print("API response:", prompt_info)
    if not prompt_info or "prompt_id" not in prompt_info:
        print("❌ Failed to queue prompt.")
        return None

    prompt_id = prompt_info["prompt_id"]
#    print(f"⏳ Prompt queued. ID: {prompt_id}")

    return prompt_id


ws_log_lock = threading.Lock()
ws_logs = []

def get_ws_log_text():
    with ws_log_lock:
        return "\n".join(ws_logs[-100:])

def websocket_receiver(prompt_id, url):
    ws_url = make_ws_url(url)
#    print("🟡 WebSocket receiver started...")
    while True:  # Keep trying to connect if connection is lost
        try:
            ws = websocket.WebSocket()
            ws.connect(ws_url)
            with ws_log_lock:
                ws_logs.append("🔌 WebSocket connected.")
#            print("🟢 WebSocket connected.")

            while True:  # Listen for messages
                msg_str = ws.recv()
                if not msg_str:
                    continue  # Handle empty message

                try:
                    msg = json.loads(msg_str)
                    msg_type = msg.get("type", "")
                    data = msg.get("data", {})

                    # progress_state で全ノード finished 判定
                    if msg_type == "progress_state" and prompt_id is not None:
                        nodes = data.get("nodes", {})
                        states = [  node.get("state") == "finished"
                                    for node in nodes.values()
                                    if node.get("prompt_id") == prompt_id   ]
                        all_finished = all(states)

                    # 進捗ログ
                        for node in nodes.values():
                            if node.get("prompt_id") != prompt_id:
                                continue
                            value = node.get("value")
                            max_val = node.get("max")
                            if value is not None and max_val:
                                percent = (value / max_val) * 100
                                with ws_log_lock:
                                    ws_logs.append(f"Node {node.get('node_id')} Progress: {percent:.1f}%")

                        if all_finished:
                            ctm.finish_generation(prompt_id, url)
                            break

                except json.JSONDecodeError:
                    err = f"⚠️ Failed to decode JSON: {msg_str[:200]}..."
                    with ws_log_lock:
                        ws_logs.append(err)
                    print(err)
                except Exception as e:
                    tb = traceback.format_exc()
                    err = f"⚠️ Error processing message: {e}\nTraceback:\n{tb}\n - Msg: {msg_str[:200]}..."
                    with ws_log_lock:
                        ws_logs.append(err)
                    print(err)

        except websocket.WebSocketConnectionClosedException:
            err = "🔌 WebSocket connection closed. Attempting to reconnect..."
            with ws_log_lock:
                ws_logs.append(err)
            print(err)
            time.sleep(5)
        except Exception as e:
            err = f"⚠️ WebSocket general error: {e}. Attempting to reconnect..."
            with ws_log_lock:
                ws_logs.append(err)
            print(err)
            time.sleep(3)

        if ctm.generation_is_finished(prompt_id):
#            print(f"Monitored prompt {prompt_id} finished, exiting receiver thread.")
            break

    # Ensure WebSocket is closed on exiting the outer loop
    try:
        ws.close()
        msg = "🔌 WebSocket connection explicitly closed."
        with ws_log_lock:
            ws_logs.append(msg)
#        print(msg)
    except Exception:
        pass


def get_image_paths(prompt_id, url):
    while True:
        time.sleep(1)
        try:
            ep_url = f"{url}/history/{prompt_id}"
            resp = requests.get(ep_url)
            resp.raise_for_status()
            history = resp.json()
        except Exception as e:
            print(f"❌ 履歴取得エラー: {e}")
            continue

        if not history:
#            print(f"{prompt_id} is not in /history yet!")
            continue

        outputs = history[prompt_id].get("outputs", {})
        image_paths = []
        for node_id, node_output in outputs.items():
            for img in node_output.get("images", []):
                filename = img.get("filename")
                subfolder = img.get("subfolder", "")
                if filename:
                    full_path = os.path.join(OUTPUT_DIR, subfolder, filename) if subfolder else os.path.join(OUTPUT_DIR, filename)
#                    print(f"DEBUG: appending full_path={full_path}")
                    image_paths.append(full_path)

        if image_paths:
            ctm.save_image_paths(prompt_id, image_paths)

            saved_thread = threading.Thread(target=ctm.image_saved, args=(prompt_id,))
            saved_thread.daemon = True
            saved_thread.start()
            break
        else:
            print("❌ image_pathsがありません")
            continue
    return
