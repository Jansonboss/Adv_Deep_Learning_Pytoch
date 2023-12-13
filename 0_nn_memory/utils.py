import torch

def get_gpu_memory(synchronize=True, empty_cache=True):
    return torch.cuda.memory_allocated(), torch.cuda.memory_cached()


def create_memory_hook(handle_ref, mem, idx, hook_type, exp):
    def hook(self, *args):
        if len(mem) == 0 or mem[-1]["exp"] != exp:
            call_idx = 0
        else:
            call_idx = mem[-1]["call_idx"] + 1

        # synchronize to ensure all GPU operations are completed
        # https://discuss.pytorch.org/t/how-does-torch-cuda-synchronize-behave/147049/4
        # https://www.speechmatics.com/company/articles-and-news/timing-operations-in-pytorch
        torch.cuda.synchronize()
        # fetch memory consumption
        memory_total, memory_cached = get_gpu_memory()

        mem.append({
            "layer_idx": idx,
            "call_idx": call_idx,
            "layer_type": type(self).__name__,
            "exp": exp,
            "hook_type": hook_type,
            "mem_total": memory_total,
            "mem_cached": memory_cached
        })

    return hook


def add_memory_hooks(idx, module, mem_log, exp, hr):
    h = module.register_forward_pre_hook(create_memory_hook(hr, mem_log, idx, "pre-forward", exp))
    hr.append(h)

    h = module.register_forward_hook(create_memory_hook(hr, mem_log, idx, "forward", exp))
    hr.append(h)

    h = module.register_backward_hook(create_memory_hook(hr, mem_log, idx, "backward", exp))
    hr.append(h)


def track_memory_usage(model, x, mem_log=None, exp_name=None):
    mem_log = mem_log or []
    exp_name = exp_name or f"exp_{len(mem_log)}"
    hr = []
    # add memory hooks
    for i, module in enumerate(model.modules()):
        add_memory_hooks(i, module, mem_log, exp_name, hr)

    try:
        out = model(x)
        loss = out.sum()
        loss.backward()
    finally:
        for h in hr:
            h.remove()

    return mem_log
