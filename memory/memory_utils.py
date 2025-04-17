import gc
import torch
import wandb


def obj_to_bytes(obj):
    return obj.untyped_storage().nbytes()

def measure_memory(verbose=False):
    # gc.collect()
    # torch.cuda.empty_cache()  # Clean up any cached GPU memory

    accounted_for = []  # Pytorch sometimes uses different views of the same tensor - we don't want to count these.
    mem = 0

    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                    # Account for the main tensor
                    ptr = obj.untyped_storage().data_ptr()
                    if ptr not in accounted_for:
                        accounted_for.append(ptr)
                        mem += obj_to_bytes(obj) / (1024 ** 2)

                    # Check if the tensor has gradients and account for them
                    if obj.requires_grad and obj.grad is not None:
                        grad_ptr = obj.grad.untyped_storage().data_ptr()
                        if grad_ptr not in accounted_for:
                            accounted_for.append(grad_ptr)
                            mem += obj_to_bytes(obj.grad) / (1024 ** 2)
                    if obj.requires_grad and obj.small_grad is not None:
                        grad_ptr = obj.small_grad.untyped_storage().data_ptr()
                        if grad_ptr not in accounted_for:
                            accounted_for.append(grad_ptr)
                            mem += obj_to_bytes(obj.small_grad) / (1024 ** 2)


        except Exception as e:
            if verbose:
                print('An exception occured: {}'.format(e))
    return mem


def print_tensors(verbose=False):
    accounted_for = []  # Pytorch sometimes uses different views of the same tensor - we don't want to count these.
    all = dict()
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                ptr = obj.untyped_storage().data_ptr()
                if ptr in accounted_for:
                    continue
                else:
                    accounted_for.append(ptr)
                if str((obj.size(), obj.dtype)) not in all:
                    all[str((obj.size(), obj.dtype))] = []
                all[str((obj.size(), obj.dtype))].append(id(obj))

        except Exception as e:
            if verbose:
                print('An exception occured: {}'.format(e))
    for k, v in all.items():
        print(f'{len(v)}× {k}: {v}')


def track(tracker, global_step, name):
    if (tracker is not None):
        current_allocated, peak_allocated, current_reserved, peak_reserved = tracker.track()
        wandb.log({
            f"{name}._allocated": current_allocated,
            f"{name}.allocated_peak": peak_allocated,
            f"{name}._reserved": current_reserved,
            f"{name}.reserved_peak": peak_reserved,
            f"{name}.tensors": measure_memory(),
        },
            step=global_step,
        )
        # print(f"{name} {current_allocated}")


def print_memory_stats(units='Mb'):
    stats = torch.cuda.memory_stats()
    scale = 1024 if units == 'Kb' else (1024 ** 2 if units == 'Mb' else 1024 ** 3)
    mx = 20
    print(f"{mx * ' '}      current     allocated      freed        peak")
    print(f"{'-' * 75}")
    for key in ["allocated_bytes.all.", "reserved_bytes.all.", "active_bytes.all."]:
        k = key.split('.')[0]
        print(f"{k + (mx - len(k)) * ' '}", end="")
        for state in ["current", "allocated", "freed", "peak"]:
            print(f"{stats[key + state] / scale :10.1f} {units}", end='')
        print()
    print(f"\n{'=' * 75}\n")


####
def print_current_peak(units='Mb'):
    stats = torch.cuda.memory_stats()
    scale = 1024 if units == 'Kb' else (1024 ** 2 if units == 'Mb' else 1024 ** 3)
    print(
        f"current {stats['allocated_bytes.all.current'] / scale :5.1f} peak {stats['allocated_bytes.all.peak'] / scale :5.1f}")
    torch.cuda.reset_peak_memory_stats()
