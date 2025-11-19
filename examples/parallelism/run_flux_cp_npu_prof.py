import os
import sys

sys.path.append("..")

import time
import torch
import torch_npu
from diffusers import (
    FluxPipeline,
    FluxTransformer2DModel,
    PipelineQuantizationConfig,
)
from utils import (
    get_args,
    strify,
    cachify,
    maybe_init_distributed,
    maybe_destroy_distributed,
)
import cache_dit
from torch_npu.contrib import transfer_to_npu


args = get_args()
print(args)

# profiling配置
experimental_config = torch_npu.profiler._ExperimentalConfig(
    export_type=[
        torch_npu.profiler.ExportType.Text,
        torch_npu.profiler.ExportType.Db
        ],
    profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
    msprof_tx=False,
    aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
    l2_cache=False,
    op_attr=False,
    data_simplification=False,
    record_op_args=False,
    gc_detect_threshold=None,
    host_sys=[
        torch_npu.profiler.HostSystem.CPU,
        torch_npu.profiler.HostSystem.MEM],
    sys_io=False,
    sys_interconnection=False
)

# profiling实例化
prof = torch_npu.profiler.profile(
    activities=[
        torch_npu.profiler.ProfilerActivity.CPU,
        torch_npu.profiler.ProfilerActivity.NPU
        ],
    schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1, skip_first=0),
    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./profiling_flux"),  # profiling落盘位置
    record_shapes=False,
    profile_memory=False,
    with_stack=False,
    with_modules=False,
    with_flops=False,
    experimental_config=experimental_config)

rank, device = maybe_init_distributed(args)

pipe: FluxPipeline = FluxPipeline.from_pretrained(
    os.environ.get(
        "FLUX_DIR",
        "black-forest-labs/FLUX.1-dev",
    ),
    torch_dtype=torch.bfloat16,
    quantization_config=(
        PipelineQuantizationConfig(
            quant_backend="bitsandbytes_4bit",
            quant_kwargs={
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": torch.bfloat16,
            },
            components_to_quantize=["text_encoder_2"],
        )
        if args.quantize
        else None
    ),
).to("npu")

def transpose_to_nz(model):
    if not hasattr(model, "named_modules"):
        return
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if module.weight.data.device.type == "cpu":
                module.weight.data = module.weight.data.to("npu")
            try:
                weight = torch_npu.npu_format_cast(module.weight.data.contiguous(), 29)
                module.weight.data = weight
            except Exception as e:
                print(f"Failed to transpose {name} to NZ, skipping: {e}")
              
transpose_to_nz(pipe.vae)
transpose_to_nz(pipe.text_encoder)
transpose_to_nz(pipe.text_encoder_2)
transpose_to_nz(pipe.tokenizer)
transpose_to_nz(pipe.tokenizer_2)
transpose_to_nz(pipe.transformer)
transpose_to_nz(pipe.scheduler)

if args.cache or args.parallel_type is not None:
    cachify(args, pipe)

assert isinstance(pipe.transformer, FluxTransformer2DModel)

pipe.set_progress_bar_config(disable=rank != 0)


def run_pipe(pipe: FluxPipeline):
    image = pipe(
        "A cat holding a sign that says hello world",
        height=1024 if args.height is None else args.height,
        width=1024 if args.width is None else args.width,
        num_inference_steps=28 if args.steps is None else args.steps,
        generator=torch.Generator("cpu").manual_seed(0),
    ).images[0]
    return image


if args.compile:
    cache_dit.set_compile_configs()
    pipe.transformer = torch.compile(pipe.transformer)

# warmup
_ = run_pipe(pipe)

prof.start() # 开始profiling
start = time.time()
image = run_pipe(pipe)
prof.step()  # 记录一个轮次
end = time.time()
prof.stop() # 结束profiling

if rank == 0:
    cache_dit.summary(pipe)

    time_cost = end - start
    save_path = f"flux.{strify(args, pipe)}.png"
    print(f"Time cost: {time_cost:.2f}s")
    print(f"Saving image to {save_path}")
    image.save(save_path)

maybe_destroy_distributed()
