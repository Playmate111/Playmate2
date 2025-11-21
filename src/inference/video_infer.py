import logging
import os
import threading
import time
from datetime import timedelta
from typing import Any
from typing import Dict

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torchao.quantization import float8_weight_only
from torchao.quantization import quantize_

from .offload import Offload
from .offload import OffloadConfig
from .pipeline import VideoPipeline

from src.playmate2.schedulers import FlowMatchScheduler
from src.playmate2.models.transformer import WanModel
from src.playmate2.models.clip import CLIPModel
from src.playmate2.models.t5 import T5EncoderModel
from src.playmate2.models.vae import WanVAE
from wan_configs import WAN_CONFIGS

logger = logging.getLogger("VideoInfer")
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    f"%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d - %(funcName)s] - %(message)s"
)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

make_abs_path = lambda fn: os.path.join(os.path.dirname(os.path.realpath(__file__)), fn)


class VideoSingleGpuInfer:
    def _load_model(
        self,
        model_path: str,
        quant_model: bool = True,
        gpu_device: str = "cuda:0",
    ) -> VideoPipeline:
        logger.info(f"load finetune model:{model_path} gpu_device:{gpu_device}")

        transformer = WanModel.from_pretrained(
            model_path=model_path,
            additional_kwargs={
                'use_audio_module': True,
            }
        ).to(dtype=torch.bfloat16)

        scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)

        text_encoder = T5EncoderModel(
            text_len=WAN_CONFIGS.text_len,
            checkpoint_path=f'{WAN_CONFIGS.base_dir}/{WAN_CONFIGS.t5_checkpoint}',
            tokenizer_path=f'{WAN_CONFIGS.base_dir}/{WAN_CONFIGS.t5_tokenizer}',
            shard_fn=None,
        ).eval()
        
        vae = WanVAE(
            vae_pth=f'{WAN_CONFIGS.base_dir}/{WAN_CONFIGS.vae_checkpoint}',
        ).eval()
        
        clip = CLIPModel(
            checkpoint_path=f'{WAN_CONFIGS.base_dir}/{WAN_CONFIGS.clip_checkpoint}',
            tokenizer_path=f'{WAN_CONFIGS.base_dir}/{WAN_CONFIGS.clip_tokenizer}'
        ).eval()

        if quant_model:
            quantize_(text_encoder, float8_weight_only(), device=gpu_device)
            text_encoder.to("cpu")
            torch.cuda.empty_cache()
            quantize_(transformer, float8_weight_only(), device=gpu_device)
            transformer.to("cpu")
            torch.cuda.empty_cache()
        
        pipe = VideoPipeline(
            text_encoder=text_encoder,
            vae=vae,
            clip=clip,
            transformer=transformer,
            scheduler=scheduler,
        )
        
        return pipe

    def __init__(
        self,
        model_path: str,
        quant_model: bool = True,
        local_rank: int = 0,
        world_size: int = 1,
        is_offload: bool = True,
        offload_config: OffloadConfig = OffloadConfig(),
        add_port: str = '33459',
    ):
        self.gpu_rank = local_rank
        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://127.0.0.1:{add_port}",
            timeout=timedelta(seconds=600),
            world_size=world_size,
            rank=local_rank,
        )
        os.environ["LOCAL_RANK"] = str(local_rank)
        logger.info(f"rank:{local_rank} Distributed backend: {dist.get_backend()}")
        torch.cuda.set_device(dist.get_rank())
        torch.backends.cuda.enable_cudnn_sdp(False)
        gpu_device = f"cuda:{dist.get_rank()}"

        self.pipe: VideoPipeline = self._load_model(
            model_path=model_path,
            quant_model=quant_model,
            gpu_device=gpu_device,
        )

        from para_attn.context_parallel import init_context_parallel_mesh
        from src.context_parallel.diffusers_adapters import parallelize_pipe

        mesh = init_context_parallel_mesh(
            'cpu',
            max_ring_dim_size=1,
            max_batch_dim_size=world_size,
        )
        parallelize_pipe(self.pipe, mesh=mesh)

        if is_offload:
            Offload.offload(
                pipeline=self.pipe,
                config=offload_config,
            )
        else:
            self.pipe.to(gpu_device)

    def damon_inference(self, request_queue: mp.Queue, response_queue: mp.Queue):
        response_queue.put(f"rank:{self.gpu_rank} ready")
        logger.info(f"rank:{self.gpu_rank} finish init pipe")
        while True:
            logger.info(f"rank:{self.gpu_rank} waiting for request")
            kwargs = request_queue.get()
            start_time = time.time()
            out = self.pipe(**kwargs)
            logger.info(f"rank:{dist.get_rank()} inference time: {time.time() - start_time}")
            if dist.get_rank() == 0:
                response_queue.put(out)


def single_gpu_run(
    rank,
    model_path: str,
    request_queue: mp.Queue,
    response_queue: mp.Queue,
    quant_model: bool = True,
    world_size: int = 1,
    is_offload: bool = True,
    offload_config: OffloadConfig = OffloadConfig(),
    add_port: str = '33459',
):
    pipe = VideoSingleGpuInfer(
        model_path=model_path,
        quant_model=quant_model,
        local_rank=rank,
        world_size=world_size,
        is_offload=is_offload,
        offload_config=offload_config,
        add_port=add_port,
    )
    pipe.damon_inference(request_queue, response_queue)


class VideoInfer:
    def __init__(
        self,
        model_path: str,
        quant_model: bool = True,
        world_size: int = 1,
        is_offload: bool = True,
        offload_config: OffloadConfig = OffloadConfig(),
        add_port: str = '33459',
    ):
        self.world_size = world_size
        smp = mp.get_context("spawn")
        self.REQ_QUEUES: mp.Queue = smp.Queue()
        self.RESP_QUEUE: mp.Queue = smp.Queue()
        assert self.world_size > 0, "gpu_num must be greater than 0"
        spawn_thread = threading.Thread(
            target=self.launch_single_gpu_infer,
            args=(model_path, quant_model, world_size, is_offload, offload_config, add_port),
            daemon=True,
        )
        spawn_thread.start()
        logger.info(f"Started multi-GPU thread with GPU_NUM: {world_size}")
        print(f"Started multi-GPU thread with GPU_NUM: {world_size}")
        for _ in range(world_size):
            msg = self.RESP_QUEUE.get()
            logger.info(f"launch_multi_gpu get init msg: {msg}")
            print(f"launch_multi_gpu get init msg: {msg}")

    def launch_single_gpu_infer(
        self,
        model_path: str,
        quant_model: bool = True,
        world_size: int = 1,
        is_offload: bool = True,
        offload_config: OffloadConfig = OffloadConfig(),
        add_port: str = '33459',
    ):
        mp.spawn(
            single_gpu_run,
            nprocs=world_size,
            join=True,
            daemon=True,
            args=(
                model_path,
                self.REQ_QUEUES,
                self.RESP_QUEUE,
                quant_model,
                world_size,
                is_offload,
                offload_config,
                add_port,
            ),
        )
        logger.info(f"finish launch multi gpu infer, world_size:{world_size}")

    def inference(self, kwargs: Dict[str, Any]):
        for _ in range(self.world_size):
            self.REQ_QUEUES.put(kwargs)
        return self.RESP_QUEUE.get()
