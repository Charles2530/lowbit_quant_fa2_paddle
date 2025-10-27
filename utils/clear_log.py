import os
import time


def clear_log(log_file):
    if (
        str(log_file).startswith("eval_")
        or os.path.getmtime(log_file) > time.time() - 3600 * 24
    ):
        print("Skip clear log file:", log_file)
        return
    os.remove(log_file)


if __name__ == "__main__":
    os.chdir("/home/lin/codes/yf/codes/charles/SageAttention/logs")
    for log_file in os.listdir():
        if log_file.endswith(".log"):
            clear_log(log_file)
