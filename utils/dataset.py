from pathlib import Path
import os.path
import random
from collections import defaultdict

import torch
from torchvision import transforms
from deepspeed.utils.logging import logger
from deepspeed import comm as dist
import datasets
from PIL import Image, ImageOps
from datasets.fingerprint import Hasher

from utils.common import zero_first, empty_cuda_cache, is_main_process


DEBUG = False


def shuffle_with_seed(l, seed=None):
    rng_state = random.getstate()
    random.seed(seed)
    random.shuffle(l)
    random.setstate(rng_state)


def crop_and_resize(pil_img, size_bucket):
    if pil_img.mode not in ['RGB', 'RGBA'] and 'transparency' in pil_img.info:
        pil_img = pil_img.convert('RGBA')

    # add white background for transparent images
    if pil_img.mode == 'RGBA':
        canvas = Image.new('RGBA', pil_img.size, (255, 255, 255))
        canvas.alpha_composite(pil_img)
        pil_img = canvas.convert('RGB')
    else:
        pil_img = pil_img.convert('RGB')

    return ImageOps.fit(pil_img, size_bucket)


pil_to_tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
def encode_pil_to_latents(pil_img, vae):
    img = pil_to_tensor(pil_img)
    img = img.unsqueeze(0)
    latents = vae.encode(img.to(vae.device, vae.dtype)).latent_dist.sample()
    if hasattr(vae.config, 'shift_factor') and vae.config.shift_factor is not None:
        latents = latents - vae.config.shift_factor
    latents = latents * vae.config.scaling_factor
    latents = latents.to('cpu')
    return latents


tensor_to_pil = transforms.Compose([transforms.Lambda(lambda x: (x / 2 + 0.5).clamp(0, 1)), transforms.ToPILImage()])
def decode_latents_to_pil(latents, vae):
    latents = latents.to(vae.device)
    latents = latents / vae.config.scaling_factor
    if hasattr(vae.config, 'shift_factor'):
        latents = latents + vae.config.shift_factor
    img = vae.decode(latents.to(vae.dtype), return_dict=False)[0].to(torch.float32)
    img = img.squeeze(0)
    return tensor_to_pil(img)


def process_caption_fn(shuffle_tags=False, caption_prefix=''):
    def fn(example):
        with open(example['caption_file']) as f:
            caption = f.read().strip()
        if shuffle_tags:
            tags = [tag.strip() for tag in caption.split(',')]
            random.shuffle(tags)
            caption = ', '.join(tags)
        caption = caption_prefix + caption

        example['caption'] = caption
        return example
    return fn


def process_image_fn(vae, size_bucket):
    def fn(example):
        image_file = example['image_file']
        try:
            pil_img = Image.open(image_file)
        except Exception:
            logger.warning(f'Image file {image_file} could not be opened. Skipping.')
            return None
        pil_img = crop_and_resize(pil_img, size_bucket)
        latents = encode_pil_to_latents(pil_img, vae)

        example['latents'] = latents.squeeze(0)
        return example
    return fn


# Dataset that does caching, batching, and dividing batches across data parallel ranks.
# Logically represents a single folder of images and captions on disk.
class SizeBucketDataset:
    def __init__(self, filepaths, dataset_config, size_bucket, model):
        logger.info(f'size_bucket: {size_bucket}, num_images: {len(filepaths)}')
        self.filepaths = filepaths
        self.config = dataset_config
        self.config.setdefault('shuffle_tags', False)
        self.config.setdefault('caption_prefix', '')
        self.size_bucket = size_bucket
        self.model = model
        self.path = Path(self.config['path'])
        self.cache_dir = self.path / 'cache' / f'cache_{size_bucket[0]}x{size_bucket[1]}'
        self.text_embedding_datasets = []

        os.makedirs(self.cache_dir, exist_ok=True)
        image_and_caption_files = self.filepaths
        # This is the one place we shuffle the data. Use a fixed seed, so the dataset is identical on all processes.
        # Processes other than rank 0 will then load it from cache.
        shuffle_with_seed(image_and_caption_files, seed=0)
        image_files, caption_files = zip(*image_and_caption_files)
        ds = datasets.Dataset.from_dict({'image_file': image_files, 'caption_file': caption_files})
        self.image_file_and_caption_dataset = ds.map(process_caption_fn(shuffle_tags=self.config['shuffle_tags'], caption_prefix=self.config['caption_prefix']), remove_columns='caption_file', keep_in_memory=True)
        self.image_file_and_caption_dataset.set_format('torch')

    def _map_and_cache(self, dataset, map_fn, cache_file_prefix='', new_fingerprint_args=[]):
        # Do the fingerprinting ourselves, because otherwise map() does it by serializing the map function.
        # That goes poorly when the function is capturing huge models (slow, OOMs, etc).
        new_fingerprint_args.append(dataset._fingerprint)
        new_fingerprint = Hasher.hash(new_fingerprint_args)
        cache_file = self.cache_dir / f'{cache_file_prefix}{new_fingerprint}.arrow'
        if cache_file.exists():
            logger.info('Dataset fingerprint matched cache, loading from cache')
        else:
            logger.info('Dataset fingerprint changed, removing existing cache file and regenerating')
            for existing_cache_file in self.cache_dir.glob(f'{cache_file_prefix}*.arrow'):
                existing_cache_file.unlink()
        # lower writer_batch_size from the default of 1000 or we get a weird pyarrow overflow error
        dataset = dataset.map(map_fn, cache_file_name=str(cache_file), writer_batch_size=100, new_fingerprint=new_fingerprint)
        return dataset

    def _get_text_embedding_map_fn(self, i):
        def fn(example):
            example[f'text_embedding'] = self.model.get_text_embedding(i, example['caption']).to('cpu').squeeze(0)
            return example
        return fn

    # TODO: just like the model is responsible for implementing a method to get the text embedding for a caption, it should probably also
    # have a method to get the latents from an image, rather than having the code for that in this file.
    def _cache_latents(self, vae):
        with torch.no_grad():
            self.latent_dataset = self._map_and_cache(self.image_file_and_caption_dataset, process_image_fn(vae, self.size_bucket), cache_file_prefix='latent_')

    def _cache_text_embedding(self, i):
        with torch.no_grad():
            self.text_embedding_datasets.append(self._map_and_cache(self.image_file_and_caption_dataset, self._get_text_embedding_map_fn(i), cache_file_prefix=f'text_embedding_{i+1}_', new_fingerprint_args=[i]))

    def __getitem__(self, i):
        if DEBUG:
            print(Path(self.image_file_and_caption_dataset[i]['image_file']).stem)
        latents = self.latent_dataset[i]['latents']
        text_embeddings = [te_dataset[i]['text_embedding'] for te_dataset in self.text_embedding_datasets]
        return latents, *text_embeddings

    def __len__(self):
        return len(self.image_file_and_caption_dataset)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


class ConcatenatedDataset:
    def __init__(self, datasets):
        self.datasets = datasets
        iteration_order = []
        for i, ds in enumerate(self.datasets):
            iteration_order.extend([i]*len(ds))
        shuffle_with_seed(iteration_order, 0)
        self.iteration_order = iteration_order

    def __len__(self):
        return len(self.iteration_order)

    def __iter__(self):
        iterators = [iter(ds) for ds in self.datasets]
        for i in self.iteration_order:
            yield next(iterators[i])

    def _make_divisible_by(self, n):
        new_length = (len(self) // n) * n
        self.iteration_order = self.iteration_order[:new_length]
        if new_length == 0 and is_main_process():
            logger.warning(f"size bucket {self.datasets[0].size_bucket} is being completely dropped because it doesn't have enough images")


class Dataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset_config, model):
        super().__init__()
        self.model = model
        self.post_init_called = False
        res = dataset_config['resolution']

        if dataset_config.get('enable_bucket', False):
            min_bucket_reso = dataset_config.get('min_bucket_reso')
            max_bucket_reso = dataset_config.get('max_bucket_reso')
            bucket_reso_steps = dataset_config.get('bucket_reso_steps')
            side1 = res
            side2 = res
            size_buckets = set()
            while side1 <= max_bucket_reso and side2 >= min_bucket_reso:
                size_buckets.add((side1, side2))
                size_buckets.add((side2, side1))
                side1 += bucket_reso_steps
                side2 -= bucket_reso_steps
        else:
            size_buckets = {(res, res)}
        size_buckets = list(size_buckets)

        datasets_by_size_bucket = defaultdict(list)
        for directory_config in dataset_config['directory']:
            size_bucket_to_filepaths = self._split_into_size_buckets(directory_config['path'], size_buckets)
            for size_bucket, filepaths in size_bucket_to_filepaths.items():
                datasets_by_size_bucket[size_bucket].append(SizeBucketDataset(filepaths, directory_config, size_bucket, model))

        self.buckets = []
        for datasets in datasets_by_size_bucket.values():
            self.buckets.append(ConcatenatedDataset(datasets))

    def post_init(self, data_parallel_rank, data_parallel_world_size, per_device_batch_size, gradient_accumulation_steps):
        self.data_parallel_rank = data_parallel_rank
        self.data_parallel_world_size = data_parallel_world_size
        self.batch_size = per_device_batch_size * gradient_accumulation_steps
        self.global_batch_size = self.data_parallel_world_size * self.batch_size
        self._make_divisible_by(self.global_batch_size)
        self.post_init_called = True

    def _make_divisible_by(self, n):
        for ds in self.buckets:
            ds._make_divisible_by(n)
        iteration_order = []
        for i, bucket in enumerate(self.buckets):
            iteration_order.extend([i]*(len(bucket) // n))
        shuffle_with_seed(iteration_order, 0)
        self.iteration_order = iteration_order
        if DEBUG:
            print(f'Dataset iteration_order: {self.iteration_order}')

    def __len__(self):
        assert self.post_init_called
        return len(self.iteration_order)

    def __iter__(self):
        assert self.post_init_called
        iterators = [iter(bucket) for bucket in self.buckets]
        for i in self.iteration_order:
            iterator = iterators[i]
            examples = [next(iterator) for _ in range(self.global_batch_size)]
            batch = self._collate(examples)
            start_idx = self.data_parallel_rank*self.batch_size
            selector = slice(start_idx, start_idx + self.batch_size)
            if DEBUG:
                print(selector)
            batch_for_this_dp_rank = tuple(x[selector] for x in batch)
            yield self.model.prepare_inputs(batch_for_this_dp_rank)

    def _collate(self, examples):
        return tuple(torch.stack(tensors, dim=0) for tensors in zip(*examples))

    def _split_into_size_buckets(self, path, size_buckets):
        size_bucket_to_filepaths = defaultdict(list)
        path = Path(path)
        if not path.exists() or not path.is_dir():
            raise RuntimeError(f'Invalid path: {path}')
        files = list(Path(path).glob('*'))
        # deterministic order
        files.sort()
        for file in files:
            if not file.is_file() or file.suffix == '.txt':
                continue
            image_file = file
            caption_file = image_file.with_suffix('.txt')
            if not os.path.exists(caption_file):
                logger.warning(f'Image file {image_file} does not have corresponding caption file. Skipping.')
                continue
            size_bucket = self._find_closest_size_bucket(image_file, size_buckets)
            if size_bucket:
                size_bucket_to_filepaths[size_bucket].append((str(image_file), str(caption_file)))
        return size_bucket_to_filepaths

    def _find_closest_size_bucket(self, image_file, size_buckets):
        try:
            pil_img = Image.open(image_file)
        except Exception:
            logger.warning(f'Image file {image_file} could not be opened. Skipping.')
            return None
        width, height = pil_img.size
        ar = width / height
        best_size_bucket = None
        best_ar_diff = float('inf')
        for size_bucket in size_buckets:
            bucket_ar = size_bucket[0] / size_bucket[1]
            ar_diff = abs(bucket_ar - ar)
            if ar_diff < best_ar_diff:
                best_ar_diff = ar_diff
                best_size_bucket = size_bucket
        return best_size_bucket


# Helper class to make caching multiple datasets more efficient by moving
# models to GPU as few times as needed.
class DatasetManager:
    def __init__(self, model):
        self.model = model
        self.datasets = []

    def register(self, dataset):
        for bucket in dataset.buckets:
            for ds in bucket.datasets:
                self.datasets.append(ds)

    def cache(self):
        with zero_first():
            self._cache()

    @torch.no_grad()
    def _cache(self):
        vae = self.model.get_vae()
        if is_main_process():
            vae.to('cuda')
        for ds in self.datasets:
            ds._cache_latents(vae)
        vae.to('cpu')
        empty_cuda_cache()

        for i, text_encoder in enumerate(self.model.get_text_encoders()):
            if is_main_process():
                text_encoder.to('cuda')
            for ds in self.datasets:
                ds._cache_text_embedding(i)
            text_encoder.to('cpu')
            empty_cuda_cache()


def split_batch(batch, pieces):
    example_tuple = batch
    split_size = example_tuple[0].size(0) // pieces
    split_examples = zip(*(torch.split(tensor, split_size) for tensor in example_tuple))
    # Deepspeed works with a tuple of (features, labels), even if we don't provide a loss_fn to PipelineEngine,
    # and instead compute the loss ourselves in the model. It's okay to just return None for the labels here.
    return [(ex, None) for ex in split_examples]


# DataLoader that divides batches into microbatches for gradient accumulation steps when doing
# pipeline parallel training. Iterates indefinitely (deepspeed requirement). Keeps track of epoch.
class PipelineDataLoader:
    def __init__(self, dataset, gradient_accumulation_steps):
        self.dataset = dataset
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.reset()

    def reset(self):
        self.epoch = 1
        self.num_batches_pulled = 0
        self._create_dataloader()

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.dataset) * self.gradient_accumulation_steps

    def __next__(self):
        try:
            micro_batch = next(self.data)
        except StopIteration:
            self._create_dataloader()
            micro_batch = next(self.data)
            self.epoch += 1
        return micro_batch

    def _create_dataloader(self):
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            pin_memory=True,
            batch_size=None
        )
        self.data = self._pull_batches_from_dataloader()
        self.num_batches_pulled = 0

    def _pull_batches_from_dataloader(self):
        for batch in self.dataloader:
            self.num_batches_pulled += 1
            for micro_batch in split_batch(batch, self.gradient_accumulation_steps):
                yield micro_batch

    # Only the first and last stages in the pipeline pull from the dataloader. Parts of the code need
    # to know the epoch, so we synchronize the epoch so the processes that don't use the dataloader
    # know the current epoch.
    def sync_epoch(self):
        process_group = dist.get_world_group()
        result = [None] * dist.get_world_size(process_group)
        torch.distributed.all_gather_object(result, self.epoch, group=process_group)
        max_epoch = -1
        for epoch in result:
            max_epoch = max(epoch, max_epoch)
        self.epoch = max_epoch


if __name__ == '__main__':
    from utils import common
    common.is_main_process = lambda: True
    from contextlib import contextmanager
    @contextmanager
    def _zero_first():
        yield
    common.zero_first = _zero_first

    from utils import dataset as dataset_util
    dataset_util.DEBUG = True

    from models import flux
    model = flux.CustomFluxPipeline.from_pretrained('/data2/imagegen_models/FLUX.1-dev', torch_dtype=torch.bfloat16)
    model.model_config = {'guidance': 1.0, 'dtype': torch.bfloat16}

    import toml
    dataset_manager = dataset_util.DatasetManager(model)
    with open('/home/anon/code/diffusion-pipe-configs/datasets/tiny1.toml') as f:
        dataset_config = toml.load(f)
    train_data = dataset_util.Dataset(dataset_config, model)
    dataset_manager.register(train_data)
    dataset_manager.cache()

    train_data.post_init(data_parallel_rank=2, data_parallel_world_size=4, per_device_batch_size=1, gradient_accumulation_steps=1)
    print(f'Dataset length: {len(train_data)}')

    for item in train_data:
        pass