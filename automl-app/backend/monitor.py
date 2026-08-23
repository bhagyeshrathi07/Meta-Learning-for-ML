import psutil
import threading
import time

class ResourceMonitor:
    def __init__(self, interval=0.1):
        self.interval = interval
        self.keep_running = False
        self.max_cpu = 0
        self.max_ram = 0
        self.peak_gpu_mem = 0
        self.process = None
        self.cpu_count = psutil.cpu_count()

    def _monitor(self):
        """Background thread that continuously monitors CPU and RAM"""
        # Get the current process (this Python script)
        self.process = psutil.Process()

        # Initialize baseline measurement
        self.process.cpu_percent(interval=None)
        time.sleep(0.15)
        
        while self.keep_running:
            try:
                # CPU: Get process-specific percentage and normalize by core count
                raw_cpu = self.process.cpu_percent(interval=None)

                # Normalize to 0-100% range (divide by number of cores)
                normalized_cpu = raw_cpu / self.cpu_count

                # Only update if we got a valid reading
                if normalized_cpu > 0:
                    self.max_cpu = max(self.max_cpu, normalized_cpu)

                # RAM
                ram_mb = self.process.memory_info().rss / (1024 ** 2)
                self.max_ram = max(self.max_ram, ram_mb)

            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break

            time.sleep(self.interval)

    def __enter__(self):
        """ Monitor context manager entry point """
        self.keep_running = True
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()
        time.sleep(0.2)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """ Monitor context manager exit point """
        self.keep_running = False
        if hasattr(self, 'thread'):
            self.process = None