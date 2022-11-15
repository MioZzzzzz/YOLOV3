import torch
import torch.utils.data
import sys
from torch import nn

from tqdm import tqdm


from pytorch_quantization import nn as quant_nn
from pytorch_quantization import calib
from pytorch_quantization.tensor_quant import QuantDescriptor

from torchvision import models
import sys, os, collections
#sys.path.append(0,"/opt/pytorch/vision/references/classification/")

sys.path.insert(0,"opt/pytorch/vision/references/classification/")
from train import evaluate, train_one_epoch, load_data
from pytorch_quantization import quant_modules
quant_modules.initialize()


quant_desc_input = QuantDescriptor(calib_method='histogram')
quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)

print("MileStone-00")

# model = models.resnet50(pretrained=True)
model = torch.load('result/yolov3.pth',map_location='cpu').to('cpu',non_blocking = True)
model.cpu()

data_path = "ILSVRC2012_img_val/"
batch_size = 512

print("MileStone-01")

#traindir = os.path.join(data_path, 'train')
#valdir = os.path.join(data_path, 'val')
traindir = data_path
valdir = data_path
_args = collections.namedtuple("mock_args", ["model", "distributed", "cache_dataset"])
dataset, dataset_test, train_sampler, test_sampler = load_data(traindir, valdir, 
        _args(model="yolov3", distributed=False, cache_dataset=False) )

print("MileStone-02")

'''
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size,
    sampler=train_sampler, num_workers=4, pin_memory=True)
'''
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size,
    sampler=train_sampler, num_workers=0, pin_memory=True)

print("MileStone-03")
'''
data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=batch_size,
    sampler=test_sampler, num_workers=4, pin_memory=True)
'''
data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=batch_size,
    sampler=test_sampler, num_workers=0, pin_memory=True)

print("MileStone-04")

def collect_stats(model, data_loader, num_batches):
    """Feed data to the network and collect statistic"""

    # Enable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()

    for i, (image, _) in tqdm(enumerate(data_loader), total=num_batches):
        model(image.cpu())
        if i >= num_batches:
            break

    # Disable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.enable_quant()
                module.disable_calib()
            else:
                module.enable()

def compute_amax(model, **kwargs):
    # Load calib result
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                if isinstance(module._calibrator, calib.MaxCalibrator):
                    module.load_calib_amax()
                else:
                    module.load_calib_amax(**kwargs)
            print(F"{name:40}: {module}")
    model.cpu()

print("MileStone-05")

# It is a bit slow since we collect histograms on cpu
with torch.no_grad():
     collect_stats(model, data_loader, num_batches=2)
     compute_amax(model, method="percentile", percentile=99.99)

print("MileStone-06")

#conv1._input_quantizer                  : TensorQuantizer(8bit fake per-tensor amax=2.6400 calibrator=MaxCalibrator(track_amax=False) quant)
#conv1._weight_quantizer                 : TensorQuantizer(8bit fake axis=(0) amax=[0.0000, 0.7817](64) calibrator=MaxCalibrator(track_amax=False) quant)
#layer1.0.conv1._input_quantizer         : TensorQuantizer(8bit fake per-tensor amax=6.8645 calibrator=MaxCalibrator(track_amax=False) quant)
#layer1.0.conv1._weight_quantizer        : TensorQuantizer(8bit fake axis=(0) amax=[0.0000, 0.7266](64) calibrator=MaxCalibrator(track_amax=False) quant)

'''
Evaluate the calibrated model
Next we will evaluate the classification accuracy of our post training quantized model on the ImageNet validation set.
'''
# criterion = nn.CrossEntropyLoss()
# with torch.no_grad():
#     evaluate(model, criterion, data_loader_test, device="cpu", print_freq=20)

print("MileStone-07")

# Save the model
torch.save(model.state_dict(), "/tmp/quant_resnet50-calibrated.pth")

print("MileStone-08")

