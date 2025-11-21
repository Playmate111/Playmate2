import functools
import gc
import os
import time
from dataclasses import dataclass

import torch
from diffusers.pipelines import DiffusionPipeline
from torchao.dtypes.affine_quantized_tensor import AffineQuantizedTensor


@dataclass
class OffloadConfig:
    # high_cpu_memory: Whether to use pinned memory for offload optimization. This can effectively prevent increased model offload latency caused by memory swapping.
    high_cpu_memory: bool = True
    # parameters_level: Whether to enable parameter-level offload. This further reduces VRAM requirements but may result in increased latency.
    parameters_level: bool = False
    # compiler_transformer: Whether to enable compilation optimization for the transformer.
    compiler_transformer: bool = False
    compiler_cache: str = "/tmp/compile_cache"


class HfHook:
    def __init__(self):
        device_id = os.environ.get("LOCAL_RANK", 0)
        self.execution_device = f"cuda:{device_id}"

    def detach_hook(self, module):
        pass


class Offload:
    def __init__(self) -> None:
        self.active_models = []
        self.active_models_ids = []
        self.active_subcaches = {}
        self.models = {}
        self.verboseLevel = 0
        self.models_to_quantize = []
        self.pinned_modules_data = {}
        self.blocks_of_modules = {}
        self.blocks_of_modules_sizes = {}
        self.compile = False
        self.device_mem_capacity = torch.cuda.get_device_properties(0).total_memory
        self.last_reserved_mem_check = 0
        self.loaded_blocks = {}
        self.prev_blocks_names = {}
        self.next_blocks_names = {}
        device_id = os.environ.get("LOCAL_RANK", 0)
        self.device_id = f"cuda:{device_id}"
        self.default_stream = torch.cuda.default_stream(self.device_id)  # torch.cuda.current_stream()
        self.transfer_stream = torch.cuda.Stream()
        self.async_transfers = False
        self.last_run_model = None

    @classmethod
    def offload(cls, pipeline: DiffusionPipeline, config: OffloadConfig = OffloadConfig()):
        """
        Enable offloading for multiple models in the pipeline, supporting video generation inference on user-level GPUs.
        pipe: the pipeline object
        config: offload strategy configuration
        """
        self = cls()
        self.pinned_modules_data = {}
        if config.parameters_level:
            model_budgets = {
                "transformer": 600 * 1024 * 1024,
                "text_encoder": 3 * 1024 * 1024 * 1024,
                "text_encoder_2": 3 * 1024 * 1024 * 1024,
            }
            self.async_transfers = True
        else:
            model_budgets = {}

        device_id = os.getenv("LOCAL_RANK", 0)
        torch.set_default_device(f"cuda:{device_id}")
        pipeline.hf_device_map = torch.device(f"cuda:{device_id}")
        pipe_or_dict_of_modules = pipeline.components
        if config.compiler_transformer:
            pipeline.transformer.to("cuda")
        models = {
            k: v
            for k, v in pipe_or_dict_of_modules.items()
            if isinstance(v, torch.nn.Module) and not (config.compiler_transformer and k == "transformer")
        }
        print_info = {k: type(v) for k, v in models.items()}
        print(f"offload models: {print_info}")
        if config.compiler_transformer:
            pipeline.text_encoder.to("cpu")
            pipeline.text_encoder_2.to("cpu")
            torch.cuda.empty_cache()
            pipeline.transformer.to("cuda")
            pipeline.vae.to("cuda")

            def move_text_encoder_to_gpu(pipe):
                torch.cuda.empty_cache()
                pipe.text_encoder.to("cuda")
                pipe.text_encoder_2.to("cuda")

            def move_text_encoder_to_cpu(pipe):
                pipe.text_encoder.to("cpu")
                pipe.text_encoder_2.to("cpu")
                torch.cuda.empty_cache()

            setattr(pipeline, "text_encoder_to_cpu", functools.partial(move_text_encoder_to_cpu, pipeline))
            setattr(pipeline, "text_encoder_to_gpu", functools.partial(move_text_encoder_to_gpu, pipeline))

            for k, module in pipe_or_dict_of_modules.items():
                if isinstance(module, torch.nn.Module):
                    for submodule_name, submodule in module.named_modules():
                        if not hasattr(submodule, "_hf_hook"):
                            setattr(submodule, "_hf_hook", HfHook())
            return self

        sizeofbfloat16 = torch.bfloat16.itemsize
        modelPinned = config.high_cpu_memory
        # Pin in RAM models
        # Calculate the VRAM requirements of the computational modules to determine whether parameters-level offload is necessary.
        for model_name, curr_model in models.items():
            curr_model.to("cpu").eval()
            pinned_parameters_data = {}
            current_model_size = 0
            print(f"{model_name} move to pinned memory:{modelPinned}")
            for p in curr_model.parameters():
                if isinstance(p, AffineQuantizedTensor):
                    if not modelPinned and p.tensor_impl.scale.dtype == torch.float32:
                        p.tensor_impl.scale = p.tensor_impl.scale.to(torch.bfloat16)
                    current_model_size += torch.numel(p.tensor_impl.scale) * sizeofbfloat16
                    current_model_size += torch.numel(p.tensor_impl.float8_data) * sizeofbfloat16 / 2
                    if modelPinned:
                        p.tensor_impl.float8_data = p.tensor_impl.float8_data.pin_memory()
                        p.tensor_impl.scale = p.tensor_impl.scale.pin_memory()
                        pinned_parameters_data[p] = [p.tensor_impl.float8_data, p.tensor_impl.scale]
                else:
                    p.data = p.data.to(torch.bfloat16) if p.data.dtype == torch.float32 else p.data.to(p.data.dtype)
                    current_model_size += torch.numel(p.data) * p.data.element_size()
                    if modelPinned:
                        p.data = p.data.pin_memory()
                        pinned_parameters_data[p] = p.data

            for buffer in curr_model.buffers():
                buffer.data = (
                    buffer.data.to(torch.bfloat16)
                    if buffer.data.dtype == torch.float32
                    else buffer.data.to(buffer.data.dtype)
                )
                current_model_size += torch.numel(buffer.data) * buffer.data.element_size()
                if modelPinned:
                    buffer.data = buffer.data.pin_memory()

            if model_name not in self.models:
                self.models[model_name] = curr_model

            curr_model_budget = model_budgets.get(model_name, 0)
            if curr_model_budget > 0 and curr_model_budget > current_model_size:
                model_budgets[model_name] = 0

            if modelPinned:
                pinned_buffers_data = {b: b.data for b in curr_model.buffers()}
                pinned_parameters_data.update(pinned_buffers_data)
                self.pinned_modules_data[model_name] = pinned_parameters_data
            gc.collect()
            torch.cuda.empty_cache()

        # if config.compiler_transformer:
        #    module = pipeline.transformer
        #    print("wrap transformer forward")
        #    # gpu model wrap
        #    for submodule_name, submodule in module.named_modules():
        #        if not hasattr(submodule, "_hf_hook"):
        #            setattr(submodule, "_hf_hook", HfHook())
        #
        #    forward_method = getattr(module, "forward")
        #
        #    def wrap_unload_all(*args, **kwargs):
        #        self.unload_all("transformer")
        #        return forward_method(*args, **kwargs)
        #
        #    setattr(module, "forward", functools.update_wrapper(wrap_unload_all, forward_method))

        # wrap forward methods
        for model_name, curr_model in models.items():
            current_budget = model_budgets.get(model_name, 0)
            current_size = 0
            self.loaded_blocks[model_name] = None
            cur_blocks_prefix, prev_blocks_name, cur_blocks_name, cur_blocks_seq = None, None, None, -1

            for submodule_name, submodule in curr_model.named_modules():
                # create a fake accelerate parameter so that the _execution_device property returns always "cuda"
                if not hasattr(submodule, "_hf_hook"):
                    setattr(submodule, "_hf_hook", HfHook())

                if not submodule_name:
                    continue

                # usr parameters-level offload
                if current_budget > 0:
                    if isinstance(submodule, (torch.nn.ModuleList, torch.nn.Sequential)):
                        if cur_blocks_prefix == None:
                            cur_blocks_prefix = submodule_name + "."
                        else:
                            if not submodule_name.startswith(cur_blocks_prefix):
                                cur_blocks_prefix = submodule_name + "."
                                cur_blocks_name, cur_blocks_seq = None, -1
                    else:
                        if cur_blocks_prefix is not None:
                            if submodule_name.startswith(cur_blocks_prefix):
                                num = int(submodule_name[len(cur_blocks_prefix) :].split(".")[0])
                                if num != cur_blocks_seq and (cur_blocks_name == None or current_size > current_budget):
                                    prev_blocks_name = cur_blocks_name
                                    cur_blocks_name = cur_blocks_prefix + str(num)
                                cur_blocks_seq = num
                            else:
                                cur_blocks_prefix = None
                                prev_blocks_name = None
                                cur_blocks_name = None
                                cur_blocks_seq = -1

                if hasattr(submodule, "forward"):
                    submodule_forward = getattr(submodule, "forward")
                    if not callable(submodule_forward):
                        print("***")
                        continue
                    if len(submodule_name.split(".")) == 1:
                        self.hook_me(submodule, curr_model, model_name, submodule_name, submodule_forward)
                    else:
                        self.hook_me_light(
                            submodule, model_name, cur_blocks_name, submodule_forward, context=submodule_name
                        )
                    current_size = self.add_module_to_blocks(model_name, cur_blocks_name, submodule, prev_blocks_name)

        gc.collect()
        torch.cuda.empty_cache()
        return self

    def add_module_to_blocks(self, model_name, blocks_name, submodule, prev_block_name):

        entry_name = model_name if blocks_name is None else model_name + "/" + blocks_name
        if entry_name in self.blocks_of_modules:
            blocks_params = self.blocks_of_modules[entry_name]
            blocks_params_size = self.blocks_of_modules_sizes[entry_name]
        else:
            blocks_params = []
            self.blocks_of_modules[entry_name] = blocks_params
            blocks_params_size = 0
            if blocks_name != None:
                prev_entry_name = None if prev_block_name == None else model_name + "/" + prev_block_name
                self.prev_blocks_names[entry_name] = prev_entry_name
                if not prev_block_name == None:
                    self.next_blocks_names[prev_entry_name] = entry_name

        for p in submodule.parameters(recurse=False):
            blocks_params.append(p)
            if isinstance(p, AffineQuantizedTensor):
                blocks_params_size += p.tensor_impl.float8_data.nbytes
                blocks_params_size += p.tensor_impl.scale.nbytes
            else:
                blocks_params_size += p.data.nbytes

        for p in submodule.buffers(recurse=False):
            blocks_params.append(p)
            blocks_params_size += p.data.nbytes

        self.blocks_of_modules_sizes[entry_name] = blocks_params_size

        return blocks_params_size

    def can_model_be_cotenant(self, model_name):
        cotenants_map = {
            "text_encoder": ["vae", "text_encoder_2"],
            "text_encoder_2": ["vae", "text_encoder"],
        }
        potential_cotenants = cotenants_map.get(model_name, None)
        if potential_cotenants is None:
            return False
        for existing_cotenant in self.active_models_ids:
            if existing_cotenant not in potential_cotenants:
                return False
        return True

    @torch.compiler.disable()
    def gpu_load_blocks(self, model_name, blocks_name, async_load=False):
        if blocks_name != None:
            self.loaded_blocks[model_name] = blocks_name

        def cpu_to_gpu(stream_to_use, blocks_params, record_for_stream=None):
            with torch.cuda.stream(stream_to_use):
                for p in blocks_params:
                    if isinstance(p, AffineQuantizedTensor):
                        p.tensor_impl.float8_data = p.tensor_impl.float8_data.cuda(
                            non_blocking=True, device=self.device_id
                        )
                        p.tensor_impl.scale = p.tensor_impl.scale.cuda(non_blocking=True, device=self.device_id)
                    else:
                        p.data = p.data.cuda(non_blocking=True, device=self.device_id)

                    if record_for_stream != None:
                        if isinstance(p, AffineQuantizedTensor):
                            p.tensor_impl.float8_data.record_stream(record_for_stream)
                            p.tensor_impl.scale.record_stream(record_for_stream)
                        else:
                            p.data.record_stream(record_for_stream)

        entry_name = model_name if blocks_name is None else model_name + "/" + blocks_name
        if self.verboseLevel >= 2:
            model = self.models[model_name]
            model_name = model._get_name()
            print(f"Loading model {entry_name} ({model_name}) in GPU")

        if self.async_transfers and blocks_name != None:
            first = self.prev_blocks_names[entry_name] == None
            next_blocks_entry = self.next_blocks_names[entry_name] if entry_name in self.next_blocks_names else None
            if first:
                cpu_to_gpu(torch.cuda.current_stream(), self.blocks_of_modules[entry_name])
            torch.cuda.synchronize()

            if next_blocks_entry != None:
                cpu_to_gpu(self.transfer_stream, self.blocks_of_modules[next_blocks_entry])

        else:
            cpu_to_gpu(self.default_stream, self.blocks_of_modules[entry_name])
            torch.cuda.synchronize()

    @torch.compiler.disable()
    def gpu_unload_blocks(self, model_name, blocks_name):
        if blocks_name != None:
            self.loaded_blocks[model_name] = None

        blocks_name = model_name if blocks_name is None else model_name + "/" + blocks_name

        if self.verboseLevel >= 2:
            model = self.models[model_name]
            model_name = model._get_name()
            print(f"Unloading model {blocks_name} ({model_name}) from GPU")

        blocks_params = self.blocks_of_modules[blocks_name]

        if model_name in self.pinned_modules_data:
            pinned_parameters_data = self.pinned_modules_data[model_name]
            for p in blocks_params:
                if isinstance(p, AffineQuantizedTensor):
                    data = pinned_parameters_data[p]
                    p.tensor_impl.float8_data = data[0]
                    p.tensor_impl.scale = data[1]
                else:
                    p.data = pinned_parameters_data[p]
        else:
            for p in blocks_params:
                if isinstance(p, AffineQuantizedTensor):
                    p.tensor_impl.float8_data = p.tensor_impl.float8_data.cpu()
                    p.tensor_impl.scale = p.tensor_impl.scale.cpu()
                else:
                    p.data = p.data.cpu()

    @torch.compiler.disable()
    def gpu_load(self, model_name):
        model = self.models[model_name]
        self.active_models.append(model)
        self.active_models_ids.append(model_name)

        self.gpu_load_blocks(model_name, None)

        # torch.cuda.current_stream().synchronize()

    @torch.compiler.disable()
    def unload_all(self, model_name: str):
        if len(self.active_models_ids) == 0 and self.last_run_model == model_name:
            self.last_run_model = model_name
            return
        for model_name in self.active_models_ids:
            self.gpu_unload_blocks(model_name, None)
            loaded_block = self.loaded_blocks[model_name]
            if loaded_block != None:
                self.gpu_unload_blocks(model_name, loaded_block)
                self.loaded_blocks[model_name] = None

        self.active_models = []
        self.active_models_ids = []
        self.active_subcaches = []
        torch.cuda.empty_cache()
        gc.collect()
        self.last_reserved_mem_check = time.time()
        self.last_run_model = model_name

    def move_args_to_gpu(self, *args, **kwargs):
        new_args = []
        new_kwargs = {}
        for arg in args:
            if torch.is_tensor(arg):
                if arg.dtype == torch.float32:
                    arg = arg.to(torch.bfloat16).cuda(non_blocking=True, device=self.device_id)
                else:
                    arg = arg.cuda(non_blocking=True, device=self.device_id)
            new_args.append(arg)

        for k in kwargs:
            arg = kwargs[k]
            if torch.is_tensor(arg):
                if arg.dtype == torch.float32:
                    arg = arg.to(torch.bfloat16).cuda(non_blocking=True, device=self.device_id)
                else:
                    arg = arg.cuda(non_blocking=True, device=self.device_id)
            new_kwargs[k] = arg

        return new_args, new_kwargs

    def ready_to_check_mem(self):
        if self.compile:
            return
        cur_clock = time.time()
        # can't check at each call if we can empty the cuda cache as quering the reserved memory value is a time consuming operation
        if (cur_clock - self.last_reserved_mem_check) < 0.200:
            return False
        self.last_reserved_mem_check = cur_clock
        return True

    def empty_cache_if_needed(self):
        mem_reserved = torch.cuda.memory_reserved()
        mem_threshold = 0.9 * self.device_mem_capacity
        if mem_reserved >= mem_threshold:
            mem_allocated = torch.cuda.memory_allocated()
            if mem_allocated <= 0.70 * mem_reserved:
                torch.cuda.empty_cache()
                tm = time.time()
                if self.verboseLevel >= 2:
                    print(f"Empty Cuda cache at {tm}")

    def any_param_or_buffer(self, target_module: torch.nn.Module):

        for _ in target_module.parameters(recurse=False):
            return True

        for _ in target_module.buffers(recurse=False):
            return True

        return False

    def hook_me_light(self, target_module, model_name, blocks_name, previous_method, context):

        anyParam = self.any_param_or_buffer(target_module)

        def check_empty_cuda_cache(module, *args, **kwargs):
            if self.ready_to_check_mem():
                self.empty_cache_if_needed()
            return previous_method(*args, **kwargs)

        def load_module_blocks(module, *args, **kwargs):
            if blocks_name == None:
                if self.ready_to_check_mem():
                    self.empty_cache_if_needed()
            else:
                loaded_block = self.loaded_blocks[model_name]
                if loaded_block == None or loaded_block != blocks_name:
                    if loaded_block != None:
                        self.gpu_unload_blocks(model_name, loaded_block)
                        if self.ready_to_check_mem():
                            self.empty_cache_if_needed()
                    self.loaded_blocks[model_name] = blocks_name
                    self.gpu_load_blocks(model_name, blocks_name)
            return previous_method(*args, **kwargs)

        if hasattr(target_module, "_mm_id"):
            orig_model_name = getattr(target_module, "_mm_id")
            if self.verboseLevel >= 2:
                print(
                    f"Model '{model_name}' shares module '{target_module._get_name()}' with module '{orig_model_name}' "
                )
            assert not anyParam
            return
        setattr(target_module, "_mm_id", model_name)

        if blocks_name != None and anyParam:
            setattr(
                target_module,
                "forward",
                functools.update_wrapper(functools.partial(load_module_blocks, target_module), previous_method),
            )
            # print(f"new cache:{blocks_name}")
        else:
            setattr(
                target_module,
                "forward",
                functools.update_wrapper(functools.partial(check_empty_cuda_cache, target_module), previous_method),
            )

    def hook_me(self, target_module, model, model_name, module_id, previous_method):
        def check_change_module(module, *args, **kwargs):
            performEmptyCacheTest = False
            if not model_name in self.active_models_ids:
                new_model_name = getattr(module, "_mm_id")
                if not self.can_model_be_cotenant(new_model_name):
                    self.unload_all(model_name)
                    performEmptyCacheTest = False
                self.gpu_load(new_model_name)
            args, kwargs = self.move_args_to_gpu(*args, **kwargs)
            if performEmptyCacheTest:
                self.empty_cache_if_needed()
            return previous_method(*args, **kwargs)

        if hasattr(target_module, "_mm_id"):
            return
        setattr(target_module, "_mm_id", model_name)

        setattr(
            target_module,
            "forward",
            functools.update_wrapper(functools.partial(check_change_module, target_module), previous_method),
        )

        if not self.verboseLevel >= 1:
            return

        if module_id == None or module_id == "":
            model_name = model._get_name()
            print(f"Hooked in model '{model_name}' ({model_name})")
