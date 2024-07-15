import argparse
import time

import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint

from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor

import sys, os
sys.path.append(os.getcwd())

from backbone import eva2
# from backbone.vim import VisionMambaSeg
from model import benchmark_model


def num_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    n_params = sum([torch.prod(torch.tensor(p.size())) for p in model_parameters])
    return n_params.item()

def parse_args():
    parser = argparse.ArgumentParser(description='MMSeg benchmark a model')
    parser.add_argument('config', help='test config file path')
    # parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--log-interval', type=int, default=5, help='interval of logging')
    parser.add_argument(
        '--if-benchmark-model', action='store_true', help='if benchmark model')
    parser.add_argument(
        '--input-img-size', type=int, default=512, help='input image size')
    parser.add_argument(
        '--batch-size', type=int, default=1, help='batch size')
    parser.add_argument(
        '--if-plain-backbone', action='store_true', help='if plain backbone')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark
    torch.backends.cudnn.benchmark = False
    cfg.model.pretrained = None    
    cfg.data.test.test_mode = True
    
    # set input image size
    TARGET_SIZE = args.input_img_size
    if not args.if_benchmark_model:
        cfg.model.backbone.img_size = TARGET_SIZE
        cfg.crop_size = (TARGET_SIZE, TARGET_SIZE)
        cfg.model.test_cfg.crop_size = cfg.crop_size
        cfg.data.test.pipeline[1]['img_scale'] = (TARGET_SIZE*4, TARGET_SIZE)
    else:
        cfg.model.backbone.img_size = TARGET_SIZE
        cfg.crop_size = (TARGET_SIZE, TARGET_SIZE)
        cfg.data.benchmark.pipeline[1]['img_scale'] = (TARGET_SIZE*4, TARGET_SIZE)
        cfg.data.benchmark.pipeline[2]['crop_size'] = (TARGET_SIZE, TARGET_SIZE)
        # cfg.benchmark_pipeline[1]['img_scale'] = (TARGET_SIZE*4, TARGET_SIZE)
        # cfg.benchmark_pipeline[2]['crop_size'] = (TARGET_SIZE, TARGET_SIZE)


    # set benchmark model
    if args.if_benchmark_model:
        cfg.model.type = 'BenchmarkEncoderDecoder'

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    if not args.if_benchmark_model:
        dataset = build_dataset(cfg.data.test)
    else:
        dataset = build_dataset(cfg.data.benchmark)
    
    BATCH_SIZE = args.batch_size if args.if_benchmark_model else 1
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=BATCH_SIZE,
        # workers_per_gpu=cfg.data.workers_per_gpu,
        workers_per_gpu=1,
        dist=False,
        shuffle=False)

    # build the model and load checkpoint
    cfg.model.train_cfg = None

    if not args.if_benchmark_model:
        model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    else:
        model = build_segmentor(cfg.model, test_cfg=cfg.get('benchmark_cfg'))
    # load_checkpoint(model, args.checkpoint, map_location='cpu')

    model = MMDataParallel(
        model,#.to(torch.float16), 
        device_ids=[0])

    print(f"Number of parameters: {num_params(model)}")
    # import sys; sys.exit(0)

    model.eval()

    print(model)
    

    # the first several iterations may be very slow so skip them
    num_warmup = 5
    pure_inf_time = 0
    total_iters = len(data_loader)

    # benchmark with 200 image and take the average
    for i, data in enumerate(data_loader):

        torch.cuda.synchronize()
        start_time = time.perf_counter()

        with torch.no_grad():
            if not args.if_benchmark_model:
                model(return_loss=False, rescale=True, **data)
            else:
                data['img'] = data['img']#.to(torch.float16)
                model(benchmark=True, benchmark_plain_backbone=args.if_plain_backbone, **data)


        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time

        if i >= num_warmup:
            pure_inf_time += elapsed
            if (i + 1) % args.log_interval == 0:
                fps = (i + 1 - num_warmup) / pure_inf_time * BATCH_SIZE
                print(f'Done image [{i + 1:<3}/ {total_iters}], '
                      f'fps: {fps:.2f} img / s')
                # print(len(data_loader))

        if (i + 1) == total_iters:
            fps = (i + 1 - num_warmup) / pure_inf_time * BATCH_SIZE
            print(f'Overall fps: {fps:.2f} img / s')
            break


if __name__ == '__main__':
    main()