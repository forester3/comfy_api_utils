import threading

log_lock = threading.Lock()       # ファイル書き込み用のロック

# ログをリアルタイムで読み取り、ファイルに書き込む関数
def read_logs_and_write_file(pipe, log_file_path, prefix):
    with open(log_file_path, "a", encoding="utf-8") as f:
        for line in iter(pipe.readline, b''):
            decoded_line = line.decode('utf-8', errors='ignore').strip()
            tagged_line = f"[{prefix}] {decoded_line}"
            # print(tagged_line)    #debug時、即時コメント解除

            with log_lock:
                f.write(tagged_line + "\n")
                f.flush() # ファイルバッファをフラッシュ
    pipe.close()

def get_log_text(path: str):
    try:
        with log_lock:
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        return "".join(lines[-100:])  # 最新100行だけ
    except Exception as e:
        return f"⚠️ ログの読み取りに失敗しました: {e}"