from datetime import datetime
import time


def get_current_time() -> str:
    dt = datetime.fromtimestamp(time.time())
    ms = dt.microsecond // 1000
    return f"{dt.year}/{dt.month:02d}/{dt.day:02d}-{dt.hour:02d}:{dt.minute:02d}:{dt.second:02d}:{ms:03d}"


time_val = 1
last_time = time.time()
while True:
    current_time = time.time()
    if current_time - last_time >= time_val:
        print(f"[info] current time is {get_current_time()}")
        last_time = current_time
