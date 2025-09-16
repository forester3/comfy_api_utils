import os
import time
import threading
import comfy_api as capi

GENERATED = "generated"
SAVED = "saved"
PATHS = "paths"

task_status_lock = threading.Lock()
task_status = {}

def clear_task_status():
    with task_status_lock:
        task_status.clear()

def init_task_status(prompt_id, url):
    with task_status_lock:
        task_status[prompt_id] = {GENERATED:False, SAVED:False, PATHS:[]}

    websocket_thread = threading.Thread(target=capi.websocket_receiver, args=(prompt_id, url,))
    websocket_thread.daemon = True
    websocket_thread.start()
    return

def finish_generation(prompt_id, url):
    with task_status_lock:
        task_status[prompt_id][GENERATED] = True
    paths_thread = threading.Thread(target=capi.get_image_paths, args=(prompt_id, url,))
    paths_thread.daemon = True
    paths_thread.start()
    return

def generation_is_finished(prompt_id):
    with task_status_lock:
        if task_status[prompt_id][GENERATED]:
            return True
    return False

def is_file_finished(path, wait_time=0.5, max_attempts=10):
    for _ in range(max_attempts):
        if not os.path.exists(path):
            time.sleep(wait_time)
            continue
        size1 = os.path.getsize(path)
        time.sleep(wait_time)
        size2 = os.path.getsize(path)
        if size1 == size2:
            return True
    return False

sem_view = threading.Semaphore(0)

def image_saved(prompt_id):
    with task_status_lock:
        paths = task_status[prompt_id][PATHS]

    for path in paths:
        attempts = 0
        while not is_file_finished(path) and attempts < 20:
            time.sleep(0.2)
            attempts += 1
        if is_file_finished(path):
            print(f"image saved: {path}")
        else:
            print(f"⚠️ timeout waiting for file: {path}")

    with task_status_lock:
        task_status[prompt_id][SAVED] = True
    sem_view.release()      #表示用Semaphoreリリース
    return

def is_last_saved():
    with task_status_lock:
        if task_status:
            last_prompt_id = list(task_status.keys())[-1]
            if task_status[last_prompt_id][SAVED] == False:
                return False
            else:
                return True
        else:
            return True

def get_latest_path():
    with task_status_lock:
        prompt_id = list(task_status.keys())[-1]
        path = task_status[prompt_id][PATHS][0]
    return path

def save_image_paths(prompt_id, image_paths):
    with task_status_lock:
        task_status[prompt_id][PATHS] = image_paths
