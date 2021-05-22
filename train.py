# Author:LiPu
import argparse
from models import *
from utils.datasets import *
import torch.optim as optim
from utils.config import coco
from utils.config import ship, voc
import torch.optim.lr_scheduler as lr_scheduler
import test

from utils.torch_utils import ModelEMA, select_device  # DDP import
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

wdir = 'weights' + os.sep  # weights dir
last = wdir + 'last.pt'
best = wdir + 'best.pt'
results_file = 'results.txt'

# Hyperparameters
hyp = {'giou': 3.54,  # giou loss gain
       'cls': 37.4,  # cls loss gain
       'cls_pw': 1.0,  # cls BCELoss positive_weight
       'obj': 64.3,  # obj loss gain (*=img_size/320 if img_size != 320)
       'obj_pw': 1.0,  # obj BCELoss positive_weight
       'iou_t': 0.20,  # iou training threshold
       'lr0': 0.001,  # initial learning rate (SGD=5E-3, Adam=5E-4)
       'lrf': 0.0005,  # final learning rate (with cos scheduler)
       'momentum': 0.9,  # SGD momentum
       'weight_decay': 5e-4,  # optimizer weight decay
       'fl_gamma': 0.0,  # focal loss gamma (efficientDet default is gamma=1.5)
       'hsv_h': 0.0138,  # image HSV-Hue augmentation (fraction)
       'hsv_s': 0.678,  # image HSV-Saturation augmentation (fraction)
       'hsv_v': 0.36,  # image HSV-Value augmentation (fraction)
       'degrees': 1.98 * 0,  # image rotation (+/- deg)
       'translate': 0.05 * 0,  # image translation (+/- fraction)
       'scale': 0.05 * 0,  # image scale (+/- gain)
       'shear': 0.641 * 0,
       'overlap': 0.5,  # overlap threshold
       'neg_pos': 3 # 正负例比例
       }


def train():
    cfg = opt.cfg
    data = opt.data
    epochs = opt.epochs  # 500200 batches at bs 64, 117263 images = 273 epochs
    batch_size = opt.batch_size
    weights = opt.weights
    img_size = opt.img_size

    data_dict = parse_data_cfg(data)
    train_path = data_dict['train']
    test_path = data_dict['valid']
    nc = int(data_dict['classes'])

    # DDP  init
    if opt.local_rank != -1:
        if opt.local_rank == 0:
            print("--------------using ddp---------------")
        assert torch.cuda.device_count() > opt.local_rank
        torch.cuda.set_device(opt.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')  # distributed backend

        assert opt.batch_size % opt.world_size == 0, '--batch-size must be multiple of CUDA device count'
        opt.batch_size = opt.batch_size // opt.world_size
    else:
        dist.init_process_group(backend='nccl',  # 'distributed backend'
                                init_method='tcp://127.0.0.1:9999',  # distributed training init method
                                world_size=1,  # number of nodes for distributed training
                                rank=0)  # distributed training node rank

    model = SSD(cfg, nc, ship).to(device)
    criterion = MultiBoxLoss(nc, device, overlap_thresh=hyp['overlap'], neg_pos=hyp['neg_pos'])

    # Optimizer
    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in dict(model.named_parameters()).items():
        if '.bias' in k:
            pg2 += [v]  # biases
        elif 'Conv2d.weight' in k:
            pg1 += [v]  # apply weight_decay
        else:
            pg0 += [v]  # all else

    if opt.adam:
        # hyp['lr0'] *= 0.1  # reduce lr (i.e. SGD=5E-3, Adam=5E-4)
        optimizer = optim.Adam(pg0, lr=hyp['lr0'])
        # optimizer = AdaBound(pg0, lr=hyp['lr0'], final_lr=0.1)
    else:
        optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)
    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    print('Optimizer groups: %g .bias, %g Conv2d.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
    del pg0, pg1, pg2

    start_epoch = 0
    best_fitness = 0
    if weights.endswith('.pt'):  # pytorch format
        chkpt = torch.load(weights, map_location=device)

        # load model
        try:
            chkpt['model'] = {k: v for k, v in chkpt['model'].items() if model.state_dict()[k].numel() == v.numel()}
            model.load_state_dict(chkpt['model'], strict=False)
        except KeyError as e:
            s = "%s is not compatible with %s. Specify --weights '' or specify a --cfg compatible with %s. " \
                "See https://github.com/ultralytics/yolov3/issues/657" % (opt.weights, opt.cfg, opt.weights)
            raise KeyError(s) from e

        # load optimizer
        if chkpt['optimizer'] is not None:
            optimizer.load_state_dict(chkpt['optimizer'])
            if chkpt.get('best_fitness') is not None:
                best_fitness = chkpt['best_fitness']
        # load results
        if chkpt.get('training_results') is not None:
            with open(results_file, 'w') as file:
                file.write(chkpt['training_results'])  # write results.txt
        if chkpt.get('epoch') is not None:
            start_epoch = chkpt['epoch'] + 1
        del chkpt

    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(epochs * 0.6), int(epochs * 0.8)], gamma=0.1,
                                         last_epoch=-1)
    scheduler.last_epoch = start_epoch

    # Initialize distributed training
    if opt.local_rank != -1:
        model = DDP(model, device_ids=[opt.local_rank], output_device=opt.local_rank, find_unused_parameters=False)
    else:
        model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=False)

    # Dataset
    dataset = LoadImagesAndLabels(train_path, img_size, batch_size,
                                  augment=False,
                                  hyp=hyp,  # augmentation hyperparameters
                                  rect=opt.rect,  # rectangular training
                                  cache_images=opt.cache_images)
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)  # ddp sampler

    batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             num_workers=nw,
                                             shuffle=False if (opt.local_rank != -1) else not opt.rect,
                                             pin_memory=True,
                                             collate_fn=dataset.collate_fn,
                                             sampler=train_sampler if (opt.local_rank != -1) else None
                                             )

    # Testloader
    testloader = torch.utils.data.DataLoader(LoadImagesAndLabels(test_path, img_size, batch_size,
                                                                 hyp=hyp,
                                                                 rect=False,
                                                                 cache_images=opt.cache_images),
                                             batch_size=batch_size,
                                             num_workers=nw,
                                             pin_memory=True,
                                             collate_fn=dataset.collate_fn)

    nb = len(dataloader)
    t0 = time.time()
    for epoch in range(start_epoch, epochs):
        if opt.local_rank != -1:
            dataloader.sampler.set_epoch(epoch)  # DDP set seed
        model.train()
        if opt.local_rank == -1 or opt.local_rank == 0:
            print(('\n' + '%10s' * 7) % ('Epoch', 'gpu_mem', 'obj', 'cls', 'total', 'targets', 'img_size'))
        pbar = tqdm(enumerate(dataloader), total=nb)  # progress bar
        mloss = torch.zeros(3).to(device)  # mean losses
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------

            imgs = imgs.to(device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
            targets = targets.to(device)
            output = model(imgs)
            loss_l, loss_c = criterion(output, targets)
            loss = loss_l + loss_c
            loss_items = torch.cat((loss_l.unsqueeze(0), loss_c.unsqueeze(0), loss.unsqueeze(0))).detach()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
            mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
            s = ('%10s' * 2 + '%10.3g' * 5) % ('%g/%g' % (epoch, epochs - 1), mem, *mloss, len(targets), img_size)
            pbar.set_description(s)

        # Update scheduler
        scheduler.step()
        # test
        final_epoch = epoch + 1 == epochs
        if not opt.notest or final_epoch:  # Calculate mAP
            is_coco = any([x in data for x in ['coco.data', 'coco2014.data', 'coco2017.data']]) and model.nc == 80
            results, maps = test.test(cfg,
                                      data,
                                      batch_size=batch_size,
                                      imgsz=img_size,
                                      model=model,
                                      save_json=final_epoch and is_coco,
                                      dataloader=testloader,
                                      quantized=opt.quantized,
                                      a_bit=opt.a_bit,
                                      w_bit=opt.w_bit,
                                      BN_Fold=opt.BN_Fold,
                                      FPGA=opt.FPGA,
                                      rank=opt.local_rank)

        # Write
        if opt.local_rank in [-1, 0]:
            with open(results_file, 'a') as f:
                f.write(s + '%10.3g' * 7 % results + '\n')  # P, R, mAP, F1, test_losses=(GIoU, obj, cls)
            if len(opt.name) and opt.bucket:
                os.system('gsutil cp results.txt gs://%s/results/results%s.txt' % (opt.bucket, opt.name))

        # Update best mAP
        fi = fitness(np.array(results).reshape(1, -1))  # fitness_i = weighted combination of [P, R, mAP, F1]
        if fi > best_fitness:
            best_fitness = fi
        # Save_model
        save = (not opt.nosave) or (final_epoch and not opt.evolve)
        if hasattr(model, 'module'):
            model_temp = model.module.state_dict()
        else:
            model_temp = model.state_dict()
        if save and dist.get_rank() == 0:  # DDP save model only once
            with open(results_file, 'r') as f:  # create checkpoint
                chkpt = {'epoch': epoch,
                         'best_fitness': best_fitness,
                         'training_results': f.read(),
                         'model': model_temp,
                         'optimizer': None if final_epoch else optimizer.state_dict()}

            # Save last, best and delete
            torch.save(chkpt, last)
            if (best_fitness == fi) and not final_epoch:
                torch.save(chkpt, best)
            del chkpt

        # end epoch ----------------------------------------------------------------------------------------------------
    # end training

    n = opt.name
    if len(n):
        n = '_' + n if not n.isnumeric() else n
        fresults, flast, fbest = 'results%s.txt' % n, wdir + 'last%s.pt' % n, wdir + 'best%s.pt' % n
        for f1, f2 in zip([wdir + 'last.pt', wdir + 'best.pt', 'results.txt'], [flast, fbest, fresults]):
            if os.path.exists(f1):
                os.rename(f1, f2)  # rename
                ispt = f2.endswith('.pt')  # is *.pt
                strip_optimizer(f2) if ispt else None  # strip optimizer
                os.system('gsutil cp %s gs://%s/weights' % (f2, opt.bucket)) if opt.bucket and ispt else None  # upload

    if not opt.evolve:
        plot_results()  # save as results.png
    if opt.local_rank in [-1, 0]:
        print('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))
    dist.destroy_process_group() if torch.cuda.device_count() > 1 else None
    torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)  # 500200 batches at bs 16, 117263 COCO images = 273 epochs
    parser.add_argument('--batch-size', type=int, default=16)  # effective bs = batch_size * accumulate = 16 * 4 = 64
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='*.cfg path')
    #parser.add_argument('--dataname', type=str, default='voc', help='name of the dataset')
    parser.add_argument('--t_cfg', type=str, default='', help='teacher model cfg file path for knowledge distillation')
    parser.add_argument('--data', type=str, default='data/coco2017.data', help='*.data path')
    parser.add_argument('--multi-scale', action='store_true', help='adjust (67%% - 150%%) img_size every 10 batches')
    parser.add_argument('--img-size', nargs='+', type=int, default=300, help='[min_train, max-train, test]')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', action='store_true', help='resume training from last.pt')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--weights', type=str, default='weights/yolov3-spp-ultralytics.pt', help='initial weights path')
    parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--adam', action='store_true', help='use adam optimizer')
    parser.add_argument('--quantized', type=int, default=0,
                        help='0:quantization way one Ternarized weight and 8bit activation')
    parser.add_argument('--a-bit', type=int, default=8,
                        help='a-bit')
    parser.add_argument('--w-bit', type=int, default=8,
                        help='w-bit')
    parser.add_argument('--BN_Fold', action='store_true', help='BN_Fold')
    parser.add_argument('--FPGA', action='store_true', help='FPGA')

    # DDP get local-rank
    parser.add_argument('--rank', default=0, help='rank of current process')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')

    opt = parser.parse_args()
    opt.weights = last if opt.resume else opt.weights
    if opt.local_rank in [-1, 0]:
        print(opt)

    # DDP set variables
    opt.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    opt.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1

    # scale hyp['obj'] by img_size (evolved at 320)
    # hyp['obj'] *= opt.img_size[0] / 320.

    # DDP set device
    if opt.local_rank != -1:
        if opt.local_rank == 0:
            device = select_device(opt.device, batch_size=opt.batch_size)
        device = torch.device('cuda', opt.local_rank)
    else:
        device = torch_utils.select_device(opt.device, batch_size=opt.batch_size)

    train()  # train normally
