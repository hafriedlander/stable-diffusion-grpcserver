import threading
import time

import psutil
import pynvml


def mb(v):
    return f"{v / 1024 / 1024 :.2f}MB"


class RamMonitor(threading.Thread):
    stop_flag = False
    ram_current = 0
    ram_max_usage = 0
    vram_current = 0
    vram_max_usage = 0

    total = -1

    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        ps = psutil.Process()

        self.loop_lock = threading.Lock()

        self.vram = False
        try:
            pynvml.nvmlInit()
            self.vram = True
        except:
            print("Unable to initialize NVIDIA management. No VRAM stats. \n")
            return

        print("Recording max memory usage...")

        self.ram_total = psutil.virtual_memory().total

        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.vram_total = pynvml.nvmlDeviceGetMemoryInfo(handle).total

        print(f"Total available RAM: {mb(self.ram_total)}, VRAM: {mb(self.vram_total)}")

        while not self.stop_flag:
            self.ram_current = ps.memory_info().rss
            self.ram_max_usage = max(self.ram_max_usage, self.ram_current)

            self.vram_current = pynvml.nvmlDeviceGetMemoryInfo(handle).used
            self.vram_max_usage = max(self.vram_max_usage, self.vram_current)

            if self.loop_lock.locked():
                self.loop_lock.release()

            time.sleep(0.1)

        print("Stopped recording.")
        pynvml.nvmlShutdown()

    def print(self):
        # Wait for the update loop to run at least once
        self.loop_lock.acquire(timeout=0.5)
        print(
            f"Current RAM: {mb(self.ram_current)}, VRAM: {mb(self.vram_current)} | "
            f"Peak RAM: {mb(self.ram_max_usage)}, VRAM: {mb(self.vram_max_usage)}"
        )

    def read(self):
        return dict(
            ram_max=self.ram_max_usage,
            ram_total=self.ram_total,
            vram_max=self.vram_max_usage,
            vram_total=self.vram_total,
        )

    def read_and_reset(self):
        result = self.read()
        self.vram_current = self.ram_current = 0
        self.vram_max_usage = self.ram_max_usage = 0
        return result

    def stop(self):
        self.stop_flag = True

    def read_and_stop(self):
        self.stop()
        return self.read()
