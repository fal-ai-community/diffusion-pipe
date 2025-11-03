# Test deepspeed.initialize()
# Run from the top-level directory of diffusion-pipe using this command:
#   deepspeed --num_gpus=2 --module test.debug_deepspeed_init --deepspeed

import argparse
import os
import json

import toml
import deepspeed
from deepspeed import comm as dist
import torch

from utils.pipeline import ManualPipelineModule
from utils.common import DTYPE_MAP


parser = argparse.ArgumentParser()
parser.add_argument('--config', default='./test/config.toml', help='Path to TOML configuration file.')
parser.add_argument('--local_rank', type=int, default=-1,
                    help='local rank passed from distributed launcher')
parser.add_argument('--master_port', type=int, default=29500, help='Master port for distributed training')
parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()


def distributed_init(args):
    """Initialize distributed training environment."""
    world_size = int(os.getenv('WORLD_SIZE', '1'))
    rank = int(os.getenv('RANK', '0'))
    local_rank = args.local_rank

    # Set environment variables for distributed training
    os.environ['MASTER_ADDR'] = os.getenv('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = str(args.master_port)

    return world_size, rank, local_rank


if __name__ == '__main__':
    with open(args.config) as f:
        # Inline TOML tables are not pickleable, which messes up the multiprocessing dataset stuff. This is a workaround.
        config = json.loads(json.dumps(toml.load(f)))

    model_config = config['model']
    model_dtype_str = model_config['dtype']
    model_config['dtype'] = DTYPE_MAP[model_dtype_str]
    if transformer_dtype := model_config.get('transformer_dtype', None):
        model_config['transformer_dtype'] = DTYPE_MAP.get(transformer_dtype, transformer_dtype)

    # Initialize distributed environment before deepspeed
    world_size, rank, local_rank = distributed_init(args)

    # Now initialize deepspeed
    deepspeed.init_distributed()

    # needed for broadcasting Queue in dataset.py
    torch.cuda.set_device(dist.get_rank())

    model_type = config['model']['type']

    if model_type == 'flux':
        from models import flux
        model = flux.FluxPipeline(config)
    elif model_type == 'ltx-video':
        from models import ltx_video
        model = ltx_video.LTXVideoPipeline(config)
    elif model_type == 'hunyuan-video':
        from models import hunyuan_video
        model = hunyuan_video.HunyuanVideoPipeline(config)
    elif model_type == 'sdxl':
        from models import sdxl
        model = sdxl.SDXLPipeline(config)
    elif model_type == 'cosmos':
        from models import cosmos
        model = cosmos.CosmosPipeline(config)
    elif model_type == 'lumina_2':
        from models import lumina_2
        model = lumina_2.Lumina2Pipeline(config)
    elif model_type == 'wan':
        from models.wan import wan
        model = wan.WanPipeline(config)
    elif model_type == 'chroma':
        from models import chroma
        model = chroma.ChromaPipeline(config)
    elif model_type == 'hidream':
        from models import hidream
        model = hidream.HiDreamPipeline(config)
    elif model_type == 'sd3':
        from models import sd3
        model = sd3.SD3Pipeline(config)
    elif model_type == 'cosmos_predict2':
        from models import cosmos_predict2
        model = cosmos_predict2.CosmosPredict2Pipeline(config)
    elif model_type == 'omnigen2':
        from models import omnigen2
        model = omnigen2.OmniGen2Pipeline(config)
    else:
        raise NotImplementedError(f'Model type {model_type} is not implemented')

    model.load_diffusion_model()
    layers = model.to_layers()

    num_stages = config.get('pipeline_stages', 1)
    pipeline_model = ManualPipelineModule(
        layers=layers,
        num_stages=num_stages,
        partition_method='parameters',
        loss_fn=model.get_loss_fn(),
    )
    parameters_to_train = [p for p in pipeline_model.parameters() if p.requires_grad]

    ds_config = {
        'train_micro_batch_size_per_gpu': config.get('micro_batch_size_per_gpu', 1),
        'gradient_accumulation_steps': config.get('gradient_accumulation_steps', 1),
        'steps_per_print': config.get('steps_per_print', 1),
    }

    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=pipeline_model,
        model_parameters=parameters_to_train,
        optimizer=torch.optim.SGD,
        config=ds_config,
    )
