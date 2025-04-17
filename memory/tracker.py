import gc
import torch


class MemoryTracker:
    units_dict = {
        "KB": 1024,
        "MB": 1024 ** 2,
        "GB": 1024 ** 3
    }

    def __init__(self, units):
        # Ensure CUDA libraries are loaded ahead of time:
        torch.cuda.current_blas_handle()
        self.unit = self.units_dict[units.upper()]

    def clear_cache(self):
        torch.cuda.empty_cache()
        gc.collect()

    def track(self):
        torch.cuda.synchronize()  # not sure we need this!!

        stats = torch.cuda.memory_stats()
        current_allocated = stats['allocated_bytes.all.current'] / self.unit
        peak_allocated = stats['allocated_bytes.all.peak'] / self.unit
        current_reserved = stats['reserved_bytes.all.current'] / self.unit
        peak_reserved = stats['reserved_bytes.all.peak'] / self.unit

        self.reset_peak_memory()

        return current_allocated, peak_allocated, current_reserved, peak_reserved

    def reset_peak_memory(self):
        torch.cuda.reset_peak_memory_stats()
