from itertools import product as product
from math import sqrt as sqrt

from utils.box_utils import decode, nms
from utils.config import coco
from utils.layers import *
from utils.parse_config import *


# SSD
def create_modules(module_defs):
    # Constructs module list of layer blocks from module configuration in module_defs

    _ = module_defs.pop(0)  # cfg training hyperparams (unused)
    output_filters = [3]  # input channels
    module_list = nn.ModuleList()
    features = []
    location = []
    classification = []
    l2_norm_index = 0
    l2_norm = 0

    for i, mdef in enumerate(module_defs):
        modules = nn.Sequential()

        if mdef['type'] == 'convolutional':
            bn = int(mdef['batch_normalize'])
            filters = int(mdef['filters'])
            kernel_size = int(mdef['size'])
            dilation = int(mdef['dilation']) if "dilation" in mdef else 1
            modules.add_module('Conv2d', nn.Conv2d(in_channels=output_filters[-1],
                                                   out_channels=filters,
                                                   kernel_size=kernel_size,
                                                   stride=int(mdef['stride']),
                                                   padding=int(mdef['pad']),
                                                   dilation=dilation,
                                                   groups=mdef['groups'] if 'groups' in mdef else 1,
                                                   bias=not bn))
            if bn:
                modules.add_module('BatchNorm2d', nn.BatchNorm2d(filters, momentum=0.1))

            if mdef['activation'] == 'leaky':
                modules.add_module('activation', nn.LeakyReLU(0.1, inplace=True))
                # modules.add_module('activation', nn.PReLU(num_parameters=1, init=0.10))
                # modules.add_module('activation', Swish())
            if mdef['activation'] == 'relu6':
                modules.add_module('activation', ReLU6())
            if mdef['activation'] == 'h_swish':
                modules.add_module('activation', HardSwish())
            if mdef['activation'] == 'relu':
                modules.add_module('activation', nn.ReLU())
            if mdef['activation'] == 'mish':
                modules.add_module('activation', Mish())

        elif mdef['type'] == 'depthwise':
            bn = int(mdef['batch_normalize'])
            filters = int(mdef['filters'])
            kernel_size = int(mdef['size'])
            pad = (kernel_size - 1) // 2 if int(mdef['pad']) else 0
            modules.add_module('DepthWise2d', nn.Conv2d(in_channels=output_filters[-1],
                                                        out_channels=filters,
                                                        kernel_size=kernel_size,
                                                        stride=int(mdef['stride']),
                                                        padding=pad,
                                                        groups=output_filters[-1],
                                                        bias=not bn), )
            if bn:
                modules.add_module('BatchNorm2d', nn.BatchNorm2d(filters, momentum=0.1))

            if mdef['activation'] == 'leaky':
                modules.add_module('activation', nn.LeakyReLU(0.1, inplace=True))
                # modules.add_module('activation', nn.PReLU(num_parameters=1, init=0.10))
                # modules.add_module('activation', Swish())
            if mdef['activation'] == 'relu6':
                modules.add_module('activation', ReLU6())
            if mdef['activation'] == 'h_swish':
                modules.add_module('activation', HardSwish())
            if mdef['activation'] == 'relu':
                modules.add_module('activation', nn.ReLU())
            if mdef['activation'] == 'mish':
                modules.add_module('activation', Mish())

        elif mdef['type'] == 'maxpool':
            filters = int(mdef['filters'])
            kernel_size = int(mdef['size'])  # kernel size
            ceil_mode = True if mdef['ceil_mode'] == 1 else False
            maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=int(mdef['stride']), padding=int(mdef['pad']),
                                   ceil_mode=ceil_mode)
            modules = maxpool

        elif mdef['type'] == 'se':
            if 'filters' in mdef:
                filters = int(mdef['filters'])
                modules.add_module('se', SE(channel=filters))
            if 'reduction' in mdef:
                modules.add_module('se', SE(output_filters[-1], reduction=int(mdef['reduction'])))

        elif mdef['type'] == 'location':
            filters = int(mdef['filters'])
            kernel_size = int(mdef['size'])
            modules.add_module('Conv2d', nn.Conv2d(in_channels=output_filters[-1],
                                                   out_channels=filters,
                                                   kernel_size=kernel_size,
                                                   stride=int(mdef['stride']),
                                                   padding=int(mdef['pad']),
                                                   bias=True))
            location.append(i)
        elif mdef['type'] == 'classification':
            filters = int(mdef['filters'])
            kernel_size = int(mdef['size'])
            modules.add_module('Conv2d', nn.Conv2d(in_channels=output_filters[-1],
                                                   out_channels=filters,
                                                   kernel_size=kernel_size,
                                                   stride=int(mdef['stride']),
                                                   padding=int(mdef['pad']),
                                                   bias=True))
            classification.append(i)

        else:
            print('Warning: Unrecognized Layer Type: ' + mdef['type'])
        if "feature" in mdef:
            if mdef['feature'] == 'linear':  # 传入预测层
                features.append(i)
            elif mdef['feature'] == 'l2_norm':
                features.append(i)
                l2_norm_index = i
                l2_norm = L2Norm(filters)

        # Register module list and number of output filters
        module_list.append(modules)
        if mdef['type'] != 'classification' and mdef['type'] != 'location':
            output_filters.append(filters)

    return module_list, l2_norm_index, features, l2_norm, classification, location


class SSD(nn.Module):
    # SSD

    def __init__(self, cfg, nc, cfg_dataset=coco, quantized=-1, a_bit=8, w_bit=8, BN_Fold=False, FPGA=False):
        super(SSD, self).__init__()

        if isinstance(cfg, str):
            self.module_defs = parse_model_cfg(cfg)
        elif isinstance(cfg, list):
            self.module_defs = cfg
        self.quantized = quantized
        self.a_bit = a_bit
        self.w_bit = w_bit
        self.BN_Fold = BN_Fold
        self.FPGA = FPGA
        self.nc = nc
        self.module_list, self.l2_norm_index, self.features, self.l2_norm, self.classification, self.location = create_modules(
            self.module_defs)
        self.cfg_dataset = cfg_dataset
        self.PriorBox = PriorBox(self.cfg_dataset).box()
        self.softmax = nn.Softmax(dim=-1)
        self.image_size = self.cfg_dataset['min_dim']
        self.detect = Detect(self.nc, 0, 200, 0.1, 0.45, self.image_size)

    def forward(self, x):
        cla, loc = self.forward_once(x)
        self.PriorBox = self.PriorBox.to(x.device)
        if self.training == True:
            return (loc, cla, self.PriorBox)
        else:
            return self.detect(loc, self.softmax(cla), self.PriorBox), (loc, cla, self.PriorBox)

    def forward_once(self, x):
        feature = None
        classification = torch.empty((x.size(0), 0, self.nc)).to(x.device)
        location = torch.empty((x.size(0), 0, 4)).to(x.device)
        for i, module in enumerate(self.module_list):
            if i in self.classification:
                output_clas = module(feature)
                output_clas = output_clas.view(output_clas.size(0), -1, self.nc)
                classification = torch.cat((classification, output_clas), dim=1)
            elif i in self.location:
                output_loc = module(feature)
                output_loc = output_loc.view(output_loc.size(0), -1, 4)
                location = torch.cat((location, output_loc), dim=1)
            else:
                x = module(x)
                if i in self.features:
                    if i != self.l2_norm_index:
                        feature = x
                    else:  # l2_norm
                        feature = self.l2_norm(x)
        return classification, location


class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    """

    def __init__(self, cfg):
        super(PriorBox, self).__init__()
        self.image_size = cfg['min_dim']
        # number of priors for feature map location (either 4 or 6)
        self.num_priors = len(cfg['aspect_ratios'])
        self.variance = cfg['variance'] or [0.1]
        self.feature_maps = cfg['feature_maps']
        self.min_sizes = cfg['min_sizes']
        self.max_sizes = cfg['max_sizes']
        self.steps = cfg['steps']
        self.aspect_ratios = cfg['aspect_ratios']
        self.clip = cfg['clip']
        self.version = cfg['name']
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def box(self):
        mean = []
        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f), repeat=2):
                f_k = self.image_size / self.steps[k]
                # unit center x,y
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                # aspect_ratio: 1
                # rel size: min_size
                s_k = self.min_sizes[k] / self.image_size
                mean += [cx, cy, s_k, s_k]

                # aspect_ratio: 1
                # rel size: sqrt(s_k * s_(k+1))
                s_k_prime = sqrt(s_k * (self.max_sizes[k] / self.image_size))
                mean += [cx, cy, s_k_prime, s_k_prime]

                # rest of aspect ratios
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k * sqrt(ar), s_k / sqrt(ar)]
                    mean += [cx, cy, s_k / sqrt(ar), s_k * sqrt(ar)]
        # back to torch land
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output.type(torch.float32)


class Detect(nn.Module):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """

    def __init__(self, num_classes, bkg_label, top_k, conf_thresh, nms_thresh, img_size):
        super(Detect, self).__init__()
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        # Parameters used in nms.
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.conf_thresh = conf_thresh
        self.variance = cfg['variance']
        self.img_size = img_size

    def forward(self, loc_data, conf_data, prior_data):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors*4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch*num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1,num_priors,4]
        """
        num = loc_data.size(0)  # batch size
        num_priors = prior_data.size(0)
        conf_preds = conf_data.view(num, num_priors,
                                    self.num_classes).transpose(2, 1)
        batch_list = []
        # Decode predictions into bboxes.
        for i in range(num):
            decoded_boxes = decode(loc_data[i], prior_data, self.variance)
            # nms
            conf_scores = conf_preds[i].clone()
            class_max = torch.argmax(conf_scores, dim=0)
            conf_scores = torch.gather(conf_scores, dim=0, index=class_max.unsqueeze(0)).squeeze()

            c_mask = conf_scores.gt(self.conf_thresh) & (class_max != 0)
            scores = conf_scores[c_mask]
            if scores.size(0) == 0:
                continue
            l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
            boxes = decoded_boxes[l_mask].view(-1, 4) * self.img_size
            class_max = class_max[c_mask]
            ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)

            output = torch.cat(
                (boxes[ids[:count]], scores[ids[:count]].unsqueeze(1), class_max[ids[:count]].unsqueeze(1)), 1)
            batch_list.append(output)
        return batch_list
        #     for cl in range(1, self.num_classes):
        #         c_mask = conf_scores[cl].gt(self.conf_thresh)
        #         scores = conf_scores[cl][c_mask]
        #         if scores.size(0) == 0:
        #             continue
        #         l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
        #         boxes = decoded_boxes[l_mask].view(-1, 4)
        #         # idx of highest scoring and non-overlapping boxes per class
        #         ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)
        #         output[i, cl, :count] = \
        #             torch.cat((scores[ids[:count]].unsqueeze(1),
        #                        boxes[ids[:count]]), 1)
        # flt = output.contiguous().view(num, -1, 5)
        # _, idx = flt[:, :, 0].sort(1, descending=True)
        # _, rank = idx.sort(1)
        # flt[(rank < self.top_k).unsqueeze(-1).expand_as(flt)].fill_(0)


if __name__ == '__main__':
    model = SSD("cfg/vgg_bn-coco.cfg")
    input = torch.ones([1, 3, 300, 300])
    output = model(input)
    print("finish!")
