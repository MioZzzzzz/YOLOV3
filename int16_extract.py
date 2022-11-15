import torch
import numpy as np
import layer_dump
#import darknet
import torch.nn as nn
import torch.nn.functional as F
import torch.quantization
import os
import random
#import LoadModel
from utils import image_preprocess,img_loader,postprocess_boxes,nms,draw_bbox
#from yolov3 import YOLOV3
import cv2
from pytorch_quantization import quant_modules
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import calib
from pytorch_quantization.tensor_quant import QuantDescriptor



input_size = 416
num_class = 80
iou_threshold = 0.45
score_threshold = 0.3
rectangle_colors = (255, 0, 0)
batch_size = 1
img_folder = os.listdir("img/")


model = torch.load('result/quant_yolov3.pth')
model.eval()

# load image
# image_path = './IMAGES/ccc.jpg'
#image_path = './IMAGES/ocean.jpg'
# image_path = './IMAGES/kite.jpg'
#original_image = cv2.imread(image_path)
# print(original_image.shape)

def image_process(image):
    image_data = image_preprocess(np.copy(image), [input_size, input_size])
    image_data = image_data.transpose((2, 0, 1))
    image_data = image_data[np.newaxis, ...].astype(np.float32)  # (1,3,416,416)

    return image_data


darknet_images = []

sample1 = random.sample(img_folder,batch_size )
#img_list = [os.path.join(nm) for nm in os.listdir(img_folder) if nm[-3:] in ['jpg', 'png', 'gif']]
for i in sample1:
    path = os.path.join("img/", i)
    image = cv2.imread(path)
    image = image_process(image)
    darknet_images.append(image)

batch_array = np.concatenate(darknet_images, axis=0)

image = np.load("conv0-conv_img.npy")
input_tensor = torch.from_numpy(image).float()

#dark0
np.save("npy/conv0-conv_img", input_tensor.cpu().detach().numpy())
out = layer_dump.QuantConv2d_save_i16(model.darknet.dark0.conv, input_tensor,
                                  "npy/conv0-conv_img_q.npy",
                                  "npy/conv0-conv_img_i16.npy",
                                  "npy/conv0-conv_weight.npy",
                                  "npy/conv0-conv_weight_q.npy",
                                  "npy/conv0-conv_weight_i16.npy",
                                  "npy/conv0-conv_out.npy",
                                  "npy/conv0-conv_out_i32.npy",
                                  "npy/conv0-conv_out_i32_dq.npy")


out = layer_dump.BN(model.darknet.dark0.bn, out,
                    "npy/conv0-bn_A.npy",
                    "npy/conv0-bn_B.npy",
                    "npy/conv0-bn_out.npy")


out = layer_dump.Relu(model.darknet.dark0.relu,out,
                      "npy/conv0-relu_out.npy")

#dark1

out = layer_dump.QuantConv2d_save_i16(model.darknet.dark1[0].conv, out,
                                  "npy/conv3-conv_img_q.npy",
                                  "npy/conv3-conv_img_i16.npy",
                                  "npy/conv3-conv_weight.npy",
                                  "npy/conv3-conv_weight_q.npy",
                                  "npy/conv3-conv_weight_i16.npy",
                                  "npy/conv3-conv_out.npy",
                                  "npy/conv3-conv_out_i32.npy",
                                  "npy/conv3-conv_out_i32_dq.npy"
                                  )
out = layer_dump.BN(model.darknet.dark1[0].bn, out,
                    "npy/conv3-bn_A.npy",
                    "npy/conv3-bn_B.npy",
                    "npy/conv3-bn_out.npy")
out1 = layer_dump.Relu(model.darknet.dark1[0].relu,out,
                      "npy/conv3-relu_out.npy")


out = layer_dump.QuantConv2d_save_i16(model.darknet.dark1[1].conv1.conv, out1,
                                  "npy/conv6-conv_img_q.npy",
                                  "npy/conv6-conv_img_i16.npy",
                                  "npy/conv6-conv_weight.npy",
                                  "npy/conv6-conv_weight_q.npy",
                                  "npy/conv6-conv_weight_i16.npy",
                                  "npy/conv6-conv_out.npy",
                                  "npy/conv6-conv_out_i32.npy",
                                  "npy/conv6-conv_out_i32_dq.npy")

out =layer_dump.BN(model.darknet.dark1[1].conv1.bn, out,
                   "npy/conv6-bn_A.npy",
                   "npy/conv6-bn_B.npy",
                   "npy/conv6-bn_out.npy")
out = layer_dump.Relu(model.darknet.dark1[1].conv1.relu,out,
                      "npy/conv6-relu_out.npy")


out = layer_dump.QuantConv2d_save_i16(model.darknet.dark1[1].conv2.conv, out,
                                  "npy/conv9-conv_img_q.npy",
                                  "npy/conv9-conv_img_i16.npy",
                                  "npy/conv9-conv_weight.npy",
                                  "npy/conv9-conv_weight_q.npy",
                                  "npy/conv9-conv_weight_i16.npy",
                                  "npy/conv9-conv_out.npy",
                                  "npy/conv9-conv_out_i32.npy",
                                  "npy/conv9-conv_out_i32_dq.npy"
                                  )
out =layer_dump.BN(model.darknet.dark1[1].conv2.bn, out,
                   "npy/conv9-bn_A.npy",
                   "npy/conv9-bn_B.npy",
                   "npy/conv9-bn_out.npy")
out = layer_dump.Relu(model.darknet.dark1[1].conv2.relu,out,
                      "npy/conv9-relu_out.npy")


out1 = out + out1
np.save("npy/eltwise12-add_out.npy", out1.cpu().detach().numpy())

#dark2
out = layer_dump.QuantConv2d_save_i16(model.darknet.dark2[0].conv,out1,
                                  "npy/conv13-conv_img_q.npy",
                                  "npy/conv13-conv_img_i16.npy",
                                  "npy/conv13-conv_weight.npy",
                                  "npy/conv13-conv_weight_q.npy",
                                  "npy/conv13-conv_weight_i16.npy",
                                  "npy/conv13-conv_out.npy",
                                  "npy/conv13-conv_out_i32.npy",
                                  "npy/conv13-conv_out_i32_dq.npy"
                                  )
out =layer_dump.BN(model.darknet.dark2[0].bn, out,
                   "npy/conv13-bn_A.npy",
                   "npy/conv13-bn_B.npy",
                   "npy/conv13-bn_out.npy")
out1 = layer_dump.Relu(model.darknet.dark2[0].relu,out,
                      "npy/conv13-relu_out.npy")

#block
out = layer_dump.QuantConv2d_save_i16(model.darknet.dark2[1].conv1.conv,out1,
                             "npy/conv16-conv_img_q.npy",
                             "npy/conv16-conv_img_i16.npy",
                             "npy/conv16-conv_weight.npy",
                             "npy/conv16-conv_weight_q.npy",
                             "npy/conv16-conv_weight_i16.npy",
                             "npy/conv16-conv_out.npy",
                             "npy/conv16-conv_out_i32.npy",
                             "npy/conv16-conv_out_i32_dq.npy"
                             )
out =layer_dump.BN(model.darknet.dark2[1].conv1.bn, out,
                   "npy/conv16-bn_A.npy",
                   "npy/conv16-bn_B.npy",
                   "npy/conv16-bn_out.npy")
out = layer_dump.Relu(model.darknet.dark2[1].conv1.relu,out,
                      "npy/conv16-relu_out.npy")



out = layer_dump.QuantConv2d_save_i16(model.darknet.dark2[1].conv2.conv,out,
                                  "npy/conv19-conv_img_q.npy",
                                  "npy/conv19-conv_img_i16.npy",
                                  "npy/conv19-conv_weight.npy",
                                  "npy/conv19-conv_weight_q.npy",
                                  "npy/conv19-conv_weight_i16.npy",
                                  "npy/conv19-conv_out.npy",
                                  "npy/conv19-conv_out_i32.npy",
                                  "npy/conv19-conv_out_i32_dq.npy"
                                  )
out =layer_dump.BN(model.darknet.dark2[1].conv2.bn, out,
                   "npy/conv19-bn_A.npy",
                   "npy/conv19-bn_B.npy",
                   "npy/conv19-bn_out.npy")
out = layer_dump.Relu(model.darknet.dark2[1].conv2.relu,out,
                      "npy/conv19-relu_out.npy")

out1 = out + out1
np.save("npy/eltwise22-add_out.npy", out1.cpu().detach().numpy())

#block
out = layer_dump.QuantConv2d_save_i16(model.darknet.dark2[2].conv1.conv,out1,
                             "npy/conv23-conv_img_q.npy",
                             "npy/conv23-conv_img_i16.npy",
                             "npy/conv23-conv_weight.npy",
                             "npy/conv23-conv_weight_q.npy",
                             "npy/conv23-conv_weight_i16.npy",
                             "npy/conv23-conv_out.npy",
                             "npy/conv23-conv_out_i32.npy",
                             "npy/conv23-conv_out_i32_dq.npy"
                             )
out =layer_dump.BN(model.darknet.dark2[2].conv1.bn, out,
                   "npy/conv23-bn_A.npy",
                   "npy/conv23-bn_B.npy",
                   "npy/conv23-bn_out.npy")
out = layer_dump.Relu(model.darknet.dark2[2].conv1.relu,out,
                      "npy/conv23-relu_out.npy")


out = layer_dump.QuantConv2d_save_i16(model.darknet.dark2[2].conv2.conv,out,
                             "npy/conv26-conv_img_q.npy",
                             "npy/conv26-conv_img_i16.npy",
                             "npy/conv26-conv_weight.npy",
                             "npy/conv26-conv_weight_q.npy",
                             "npy/conv26-conv_weight_i16.npy",
                             "npy/conv26-conv_out.npy",
                             "npy/conv26-conv_out_i32.npy",
                             "npy/conv26-conv_out_i32_dq.npy"
                             )
out =layer_dump.BN(model.darknet.dark2[2].conv2.bn, out,
                   "npy/conv26-bn_A.npy",
                   "npy/conv26-bn_B.npy",
                   "npy/conv26-bn_out.npy")
out = layer_dump.Relu(model.darknet.dark2[2].conv2.relu,out,
                      "npy/conv26-relu_out.npy")

out1 = out + out1
np.save("npy/eltwise29-add_out.npy", out1.cpu().detach().numpy())

#print(out1.shape)
#dark3
out = layer_dump.QuantConv2d_save_i16(model.darknet.dark3[0].conv,out1,
                                  "npy/conv30-conv_img_q.npy",
                                  "npy/conv30-conv_img_i16.npy",
                                  "npy/conv30-conv_weight.npy",
                                  "npy/conv30-conv_weight_q.npy",
                                  "npy/conv30-conv_weight_i16.npy",
                                  "npy/conv30-conv_out.npy",
                                  "npy/conv30-conv_out_i32.npy",
                                  "npy/conv30-conv_out_i32_dq.npy")

out = layer_dump.BN(model.darknet.dark3[0].bn, out,
                   "npy/conv30-bn_A.npy",
                   "npy/conv30-bn_B.npy",
                   "npy/conv30-bn_out.npy")
out1 = layer_dump.Relu(model.darknet.dark3[0].relu,out,
                      "npy/conv30-relu_out.npy")

#block1
out = layer_dump.QuantConv2d_save_i16(model.darknet.dark3[1].conv1.conv,out1,
                                  "npy/conv33-conv_img_q.npy",
                                  "npy/conv33-conv_img_i16.npy",
                                  "npy/conv33-conv_weight.npy",
                                  "npy/conv33-conv_weight_q.npy",
                                  "npy/conv33-conv_weight_i16.npy",
                                  "npy/conv33-conv_out.npy",
                                  "npy/conv33-conv_out_i32.npy",
                                  "npy/conv33-conv_out_i32_dq.npy")

out = layer_dump.BN(model.darknet.dark3[1].conv1.bn, out,
                   "npy/conv33-bn_A.npy",
                   "npy/conv33-bn_B.npy",
                   "npy/conv33-bn_out.npy")
out = layer_dump.Relu(model.darknet.dark3[1].conv1.relu,out,
                      "npy/conv33-relu_out.npy")

out = layer_dump.QuantConv2d_save_i16(model.darknet.dark3[1].conv2.conv,out,
                                  "npy/conv36-conv_img_q.npy",
                                  "npy/conv36-conv_img_i16.npy",
                                  "npy/conv36-conv_weight.npy",
                                  "npy/conv36-conv_weight_q.npy",
                                  "npy/conv36-conv_weight_i16.npy",
                                  "npy/conv36-conv_out.npy",
                                  "npy/conv36-conv_out_i32.npy",
                                  "npy/conv36-conv_out_i32_dq.npy")

out = layer_dump.BN(model.darknet.dark3[1].conv2.bn, out,
                   "npy/conv36-bn_A.npy",
                   "npy/conv36-bn_B.npy",
                   "npy/conv36-bn_out.npy")
out = layer_dump.Relu(model.darknet.dark3[1].conv2.relu,out,
                      "npy/conv36-relu_out.npy")


out1 = out + out1
np.save("npy/eltwise39-add_out.npy", out1.cpu().detach().numpy())

#block2
out = layer_dump.QuantConv2d_save_i16(model.darknet.dark3[2].conv1.conv,out1,
                                  "npy/conv40-conv_img_q.npy",
                                  "npy/conv40-conv_img_i16.npy",
                                  "npy/conv40-conv_weight.npy",
                                  "npy/conv40-conv_weight_q.npy",
                                  "npy/conv40-conv_weight_i16.npy",
                                  "npy/conv40-conv_out.npy",
                                  "npy/conv40-conv_out_i32.npy",
                                  "npy/conv40-conv_out_i32_dq.npy")

out = layer_dump.BN(model.darknet.dark3[2].conv1.bn, out,
                   "npy/conv40-bn_A.npy",
                   "npy/conv40-bn_B.npy",
                   "npy/conv40-bn_out.npy")
out = layer_dump.Relu(model.darknet.dark3[2].conv1.relu,out,
                      "npy/conv40-relu_out.npy")

out = layer_dump.QuantConv2d_save_i16(model.darknet.dark3[2].conv2.conv,out,
                                  "npy/conv43-conv_img_q.npy",
                                  "npy/conv43-conv_img_i16.npy",
                                  "npy/conv43-conv_weight.npy",
                                  "npy/conv43-conv_weight_q.npy",
                                  "npy/conv43-conv_weight_i16.npy",
                                  "npy/conv43-conv_out.npy",
                                  "npy/conv43-conv_out_i32.npy",
                                  "npy/conv43-conv_out_i32_dq.npy")

out = layer_dump.BN(model.darknet.dark3[2].conv2.bn, out,
                   "npy/conv43-bn_A.npy",
                   "npy/conv43-bn_B.npy",
                   "npy/conv43-bn_out.npy")
out = layer_dump.Relu(model.darknet.dark3[2].conv2.relu,out,
                      "npy/conv43-relu_out.npy")


out1 = out + out1
np.save("npy/eltwise46-add_out.npy", out1.cpu().detach().numpy())


#block3
out = layer_dump.QuantConv2d_save_i16(model.darknet.dark3[3].conv1.conv,out1,
                                  "npy/conv47-conv_img_q.npy",
                                  "npy/conv47-conv_img_i16.npy",
                                  "npy/conv47-conv_weight.npy",
                                  "npy/conv47-conv_weight_q.npy",
                                  "npy/conv47-conv_weight_i16.npy",
                                  "npy/conv47-conv_out.npy",
                                  "npy/conv47-conv_out_i32.npy",
                                  "npy/conv47-conv_out_i32_dq.npy")

out = layer_dump.BN(model.darknet.dark3[3].conv1.bn, out,
                   "npy/conv47-bn_A.npy",
                   "npy/conv47-bn_B.npy",
                   "npy/conv47-bn_out.npy")
out = layer_dump.Relu(model.darknet.dark3[3].conv1.relu,out,
                      "npy/conv47-relu_out.npy")

out = layer_dump.QuantConv2d_save_i16(model.darknet.dark3[3].conv2.conv,out,
                                  "npy/conv50-conv_img_q.npy",
                                  "npy/conv50-conv_img_i16.npy",
                                  "npy/conv50-conv_weight.npy",
                                  "npy/conv50-conv_weight_q.npy",
                                  "npy/conv50-conv_weight_i16.npy",
                                  "npy/conv50-conv_out.npy",
                                  "npy/conv50-conv_out_i32.npy",
                                  "npy/conv50-conv_out_i32_dq.npy")

out = layer_dump.BN(model.darknet.dark3[3].conv2.bn, out,
                   "npy/conv50-bn_A.npy",
                   "npy/conv50-bn_B.npy",
                   "npy/conv50-bn_out.npy")
out = layer_dump.Relu(model.darknet.dark3[3].conv2.relu,out,
                      "npy/conv50-relu_out.npy")


out1 = out + out1
np.save("npy/eltwise53-add_out.npy", out1.cpu().detach().numpy())

#block4
out = layer_dump.QuantConv2d_save_i16(model.darknet.dark3[4].conv1.conv,out1,
                                  "npy/conv54-conv_img_q.npy",
                                  "npy/conv54-conv_img_i16.npy",
                                  "npy/conv54-conv_weight.npy",
                                  "npy/conv54-conv_weight_q.npy",
                                  "npy/conv54-conv_weight_i16.npy",
                                  "npy/conv54-conv_out.npy",
                                  "npy/conv54-conv_out_i32.npy",
                                  "npy/conv54-conv_out_i32_dq.npy")

out = layer_dump.BN(model.darknet.dark3[4].conv1.bn, out,
                   "npy/conv54-bn_A.npy",
                   "npy/conv54-bn_B.npy",
                   "npy/conv54-bn_out.npy")
out = layer_dump.Relu(model.darknet.dark3[4].conv1.relu,out,
                      "npy/conv54-relu_out.npy")

out = layer_dump.QuantConv2d_save_i16(model.darknet.dark3[4].conv2.conv,out,
                                  "npy/conv57-conv_img_q.npy",
                                  "npy/conv57-conv_img_i16.npy",
                                  "npy/conv57-conv_weight.npy",
                                  "npy/conv57-conv_weight_q.npy",
                                  "npy/conv57-conv_weight_i16.npy",
                                  "npy/conv57-conv_out.npy",
                                  "npy/conv57-conv_out_i32.npy",
                                  "npy/conv57-conv_out_i32_dq.npy")

out = layer_dump.BN(model.darknet.dark3[4].conv2.bn, out,
                   "npy/conv57-bn_A.npy",
                   "npy/conv57-bn_B.npy",
                   "npy/conv57-bn_out.npy")
out = layer_dump.Relu(model.darknet.dark3[4].conv2.relu,out,
                      "npy/conv57-relu_out.npy")


out1 = out + out1
np.save("npy/eltwise60-add_out.npy", out1.cpu().detach().numpy())


#block5
out = layer_dump.QuantConv2d_save_i16(model.darknet.dark3[5].conv1.conv,out1,
                                  "npy/conv61-conv_img_q.npy",
                                  "npy/conv61-conv_img_i16.npy",
                                  "npy/conv61-conv_weight.npy",
                                  "npy/conv61-conv_weight_q.npy",
                                  "npy/conv61-conv_weight_i16.npy",
                                  "npy/conv61-conv_out.npy",
                                  "npy/conv61-conv_out_i32.npy",
                                  "npy/conv61-conv_out_i32_dq.npy")

out = layer_dump.BN(model.darknet.dark3[5].conv1.bn, out,
                   "npy/conv61-bn_A.npy",
                   "npy/conv61-bn_B.npy",
                   "npy/conv61-bn_out.npy")
out = layer_dump.Relu(model.darknet.dark3[5].conv1.relu,out,
                      "npy/conv61-relu_out.npy")

out = layer_dump.QuantConv2d_save_i16(model.darknet.dark3[5].conv2.conv,out,
                                  "npy/conv64-conv_img_q.npy",
                                  "npy/conv64-conv_img_i16.npy",
                                  "npy/conv64-conv_weight.npy",
                                  "npy/conv64-conv_weight_q.npy",
                                  "npy/conv64-conv_weight_i16.npy",
                                  "npy/conv64-conv_out.npy",
                                  "npy/conv64-conv_out_i32.npy",
                                  "npy/conv64-conv_out_i32_dq.npy")

out = layer_dump.BN(model.darknet.dark3[5].conv2.bn, out,
                   "npy/conv64-bn_A.npy",
                   "npy/conv64-bn_B.npy",
                   "npy/conv64-bn_out.npy")
out = layer_dump.Relu(model.darknet.dark3[5].conv2.relu,out,
                      "npy/conv64-relu_out.npy")


out1 = out + out1
np.save("npy/eltwise67-add_out.npy", out1.cpu().detach().numpy())

#block6
out = layer_dump.QuantConv2d_save_i16(model.darknet.dark3[6].conv1.conv,out1,
                                  "npy/conv68-conv_img_q.npy",
                                  "npy/conv68-conv_img_i16.npy",
                                  "npy/conv68-conv_weight.npy",
                                  "npy/conv68-conv_weight_q.npy",
                                  "npy/conv68-conv_weight_i16.npy",
                                  "npy/conv68-conv_out.npy",
                                  "npy/conv68-conv_out_i32.npy",
                                  "npy/conv68-conv_out_i32_dq.npy")

out = layer_dump.BN(model.darknet.dark3[6].conv1.bn, out,
                   "npy/conv68-bn_A.npy",
                   "npy/conv68-bn_B.npy",
                   "npy/conv68-bn_out.npy")
out = layer_dump.Relu(model.darknet.dark3[6].conv1.relu,out,
                      "npy/conv68-relu_out.npy")

out = layer_dump.QuantConv2d_save_i16(model.darknet.dark3[6].conv2.conv,out,
                                  "npy/conv71-conv_img_q.npy",
                                  "npy/conv71-conv_img_i16.npy",
                                  "npy/conv71-conv_weight.npy",
                                  "npy/conv71-conv_weight_q.npy",
                                  "npy/conv71-conv_weight_i16.npy",
                                  "npy/conv71-conv_out.npy",
                                  "npy/conv71-conv_out_i32.npy",
                                  "npy/conv71-conv_out_i32_dq.npy")

out = layer_dump.BN(model.darknet.dark3[6].conv2.bn, out,
                   "npy/conv71-bn_A.npy",
                   "npy/conv71-bn_B.npy",
                   "npy/conv71-bn_out.npy")
out = layer_dump.Relu(model.darknet.dark3[6].conv2.relu,out,
                      "npy/conv71-relu_out.npy")


out1 = out + out1
np.save("npy/eltwise74-add_out.npy", out1.cpu().detach().numpy())




#block7
out = layer_dump.QuantConv2d_save_i16(model.darknet.dark3[7].conv1.conv,out1,
                                  "npy/conv75-conv_img_q.npy",
                                  "npy/conv75-conv_img_i16.npy",
                                  "npy/conv75-conv_weight.npy",
                                  "npy/conv75-conv_weight_q.npy",
                                  "npy/conv75-conv_weight_i16.npy",
                                  "npy/conv75-conv_out.npy",
                                  "npy/conv75-conv_out_i32.npy",
                                  "npy/conv75-conv_out_i32_dq.npy")

out = layer_dump.BN(model.darknet.dark3[7].conv1.bn, out,
                   "npy/conv75-bn_A.npy",
                   "npy/conv75-bn_B.npy",
                   "npy/conv75-bn_out.npy")
out = layer_dump.Relu(model.darknet.dark3[7].conv1.relu,out,
                      "npy/conv75-relu_out.npy")

out = layer_dump.QuantConv2d_save_i16(model.darknet.dark3[7].conv2.conv,out,
                                  "npy/conv78-conv_img_q.npy",
                                  "npy/conv78-conv_img_i16.npy",
                                  "npy/conv78-conv_weight.npy",
                                  "npy/conv78-conv_weight_q.npy",
                                  "npy/conv78-conv_weight_i16.npy",
                                  "npy/conv78-conv_out.npy",
                                  "npy/conv78-conv_out_i32.npy",
                                  "npy/conv78-conv_out_i32_dq.npy")

out = layer_dump.BN(model.darknet.dark3[7].conv2.bn, out,
                   "npy/conv78-bn_A.npy",
                   "npy/conv78-bn_B.npy",
                   "npy/conv78-bn_out.npy")
out = layer_dump.Relu(model.darknet.dark3[7].conv2.relu,out,
                      "npy/conv78-relu_out.npy")


out1 = out + out1
np.save("npy/eltwise81-add_out.npy", out1.cpu().detach().numpy())



#block8
out = layer_dump.QuantConv2d_save_i16(model.darknet.dark3[8].conv1.conv,out1,
                                  "npy/conv82-conv_img_q.npy",
                                  "npy/conv82-conv_img_i16.npy",
                                  "npy/conv82-conv_weight.npy",
                                  "npy/conv82-conv_weight_q.npy",
                                  "npy/conv82-conv_weight_i16.npy",
                                  "npy/conv82-conv_out.npy",
                                  "npy/conv82-conv_out_i32.npy",
                                  "npy/conv82-conv_out_i32_dq.npy")

out = layer_dump.BN(model.darknet.dark3[8].conv1.bn, out,
                   "npy/conv82-bn_A.npy",
                   "npy/conv82-bn_B.npy",
                   "npy/conv82-bn_out.npy")
out = layer_dump.Relu(model.darknet.dark3[8].conv1.relu,out,
                      "npy/conv82-relu_out.npy")

out = layer_dump.QuantConv2d_save_i16(model.darknet.dark3[8].conv2.conv,out,
                                  "npy/conv85-conv_img_q.npy",
                                  "npy/conv85-conv_img_i16.npy",
                                  "npy/conv85-conv_weight.npy",
                                  "npy/conv85-conv_weight_q.npy",
                                  "npy/conv85-conv_weight_i16.npy",
                                  "npy/conv85-conv_out.npy",
                                  "npy/conv85-conv_out_i32.npy",
                                  "npy/conv85-conv_out_i32_dq.npy")

out = layer_dump.BN(model.darknet.dark3[8].conv2.bn, out,
                   "npy/conv85-bn_A.npy",
                   "npy/conv85-bn_B.npy",
                   "npy/conv85-bn_out.npy")
out = layer_dump.Relu(model.darknet.dark3[8].conv2.relu,out,
                      "npy/conv85-relu_out.npy")


out1 = out + out1
np.save("npy/eltwise88-add_out.npy", out1.cpu().detach().numpy())

out3 = out1

#dark4
out = layer_dump.QuantConv2d_save_i16(model.darknet.dark4[0].conv,out1,
                                  "npy/conv89-conv_img_q.npy",
                                  "npy/conv89-conv_img_i16.npy",
                                  "npy/conv89-conv_weight.npy",
                                  "npy/conv89-conv_weight_q.npy",
                                  "npy/conv89-conv_weight_i16.npy",
                                  "npy/conv89-conv_out.npy",
                                  "npy/conv89-conv_out_i32.npy",
                                  "npy/conv89-conv_out_i32_dq.npy")

out = layer_dump.BN(model.darknet.dark4[0].bn, out,
                   "npy/conv89-bn_A.npy",
                   "npy/conv89-bn_B.npy",
                   "npy/conv89-bn_out.npy")
out1 = layer_dump.Relu(model.darknet.dark4[0].relu,out,
                      "npy/conv89-relu_out.npy")


#block1
out = layer_dump.QuantConv2d_save_i16(model.darknet.dark4[1].conv1.conv,out1,
                                  "npy/conv92-conv_img_q.npy",
                                  "npy/conv92-conv_img_i16.npy",
                                  "npy/conv92-conv_weight.npy",
                                  "npy/conv92-conv_weight_q.npy",
                                  "npy/conv92-conv_weight_i16.npy",
                                  "npy/conv92-conv_out.npy",
                                  "npy/conv92-conv_out_i32.npy",
                                  "npy/conv92-conv_out_i32_dq.npy")

out = layer_dump.BN(model.darknet.dark4[1].conv1.bn, out,
                   "npy/conv92-bn_A.npy",
                   "npy/conv92-bn_B.npy",
                   "npy/conv92-bn_out.npy")
out = layer_dump.Relu(model.darknet.dark4[1].conv1.relu,out,
                      "npy/conv92-relu_out.npy")

out = layer_dump.QuantConv2d_save_i16(model.darknet.dark4[1].conv2.conv,out,
                                  "npy/conv95-conv_img_q.npy",
                                  "npy/conv95-conv_img_i16.npy",
                                  "npy/conv95-conv_weight.npy",
                                  "npy/conv95-conv_weight_q.npy",
                                  "npy/conv95-conv_weight_i16.npy",
                                  "npy/conv95-conv_out.npy",
                                  "npy/conv95-conv_out_i32.npy",
                                  "npy/conv95-conv_out_i32_dq.npy")

out = layer_dump.BN(model.darknet.dark4[1].conv2.bn, out,
                   "npy/conv95-bn_A.npy",
                   "npy/conv95-bn_B.npy",
                   "npy/conv95-bn_out.npy")
out = layer_dump.Relu(model.darknet.dark4[1].conv2.relu,out,
                      "npy/conv95-relu_out.npy")


out1 = out + out1
np.save("npy/eltwise98-add_out.npy", out1.cpu().detach().numpy())


#block2
out = layer_dump.QuantConv2d_save_i16(model.darknet.dark4[2].conv1.conv,out1,
                                  "npy/conv99-conv_img_q.npy",
                                  "npy/conv99-conv_img_i16.npy",
                                  "npy/conv99-conv_weight.npy",
                                  "npy/conv99-conv_weight_q.npy",
                                  "npy/conv99-conv_weight_i16.npy",
                                  "npy/conv99-conv_out.npy",
                                  "npy/conv99-conv_out_i32.npy",
                                  "npy/conv99-conv_out_i32_dq.npy")

out = layer_dump.BN(model.darknet.dark4[2].conv1.bn, out,
                   "npy/conv99-bn_A.npy",
                   "npy/conv99-bn_B.npy",
                   "npy/conv99-bn_out.npy")
out = layer_dump.Relu(model.darknet.dark4[2].conv1.relu,out,
                      "npy/conv99-relu_out.npy")

out = layer_dump.QuantConv2d_save_i16(model.darknet.dark4[2].conv2.conv,out,
                                  "npy/conv102-conv_img_q.npy",
                                  "npy/conv102-conv_img_i16.npy",
                                  "npy/conv102-conv_weight.npy",
                                  "npy/conv102-conv_weight_q.npy",
                                  "npy/conv102-conv_weight_i16.npy",
                                  "npy/conv102-conv_out.npy",
                                  "npy/conv102-conv_out_i32.npy",
                                  "npy/conv102-conv_out_i32_dq.npy")

out = layer_dump.BN(model.darknet.dark4[2].conv2.bn, out,
                   "npy/conv102-bn_A.npy",
                   "npy/conv102-bn_B.npy",
                   "npy/conv102-bn_out.npy")
out = layer_dump.Relu(model.darknet.dark4[2].conv2.relu,out,
                      "npy/conv102-relu_out.npy")


out1 = out + out1
np.save("npy/eltwise105-add_out.npy", out1.cpu().detach().numpy())


#block3
out = layer_dump.QuantConv2d_save_i16(model.darknet.dark4[3].conv1.conv,out1,
                                  "npy/conv106-conv_img_q.npy",
                                  "npy/conv106-conv_img_i16.npy",
                                  "npy/conv106-conv_weight.npy",
                                  "npy/conv106-conv_weight_q.npy",
                                  "npy/conv106-conv_weight_i16.npy",
                                  "npy/conv106-conv_out.npy",
                                  "npy/conv106-conv_out_i32.npy",
                                  "npy/conv106-conv_out_i32_dq.npy")

out = layer_dump.BN(model.darknet.dark4[3].conv1.bn, out,
                   "npy/conv106-bn_A.npy",
                   "npy/conv106-bn_B.npy",
                   "npy/conv106-bn_out.npy")
out = layer_dump.Relu(model.darknet.dark4[3].conv1.relu,out,
                      "npy/conv106-relu_out.npy")

out = layer_dump.QuantConv2d_save_i16(model.darknet.dark4[3].conv2.conv,out,
                                  "npy/conv109-conv_img_q.npy",
                                  "npy/conv109-conv_img_i16.npy",
                                  "npy/conv109-conv_weight.npy",
                                  "npy/conv109-conv_weight_q.npy",
                                  "npy/conv109-conv_weight_i16.npy",
                                  "npy/conv109-conv_out.npy",
                                  "npy/conv109-conv_out_i32.npy",
                                  "npy/conv109-conv_out_i32_dq.npy")

out = layer_dump.BN(model.darknet.dark4[3].conv2.bn, out,
                   "npy/conv109-bn_A.npy",
                   "npy/conv109-bn_B.npy",
                   "npy/conv109-bn_out.npy")
out = layer_dump.Relu(model.darknet.dark4[3].conv2.relu,out,
                      "npy/conv109-relu_out.npy")


out1 = out + out1
np.save("npy/eltwise112-add_out.npy", out1.cpu().detach().numpy())

#block4
out = layer_dump.QuantConv2d_save_i16(model.darknet.dark4[4].conv1.conv,out1,
                                  "npy/conv113-conv_img_q.npy",
                                  "npy/conv113-conv_img_i16.npy",
                                  "npy/conv113-conv_weight.npy",
                                  "npy/conv113-conv_weight_q.npy",
                                  "npy/conv113-conv_weight_i16.npy",
                                  "npy/conv113-conv_out.npy",
                                  "npy/conv113-conv_out_i32.npy",
                                  "npy/conv113-conv_out_i32_dq.npy")

out = layer_dump.BN(model.darknet.dark4[4].conv1.bn, out,
                   "npy/conv113-bn_A.npy",
                   "npy/conv113-bn_B.npy",
                   "npy/conv113-bn_out.npy")
out = layer_dump.Relu(model.darknet.dark4[4].conv1.relu,out,
                      "npy/conv113-relu_out.npy")

out = layer_dump.QuantConv2d_save_i16(model.darknet.dark4[4].conv2.conv,out,
                                  "npy/conv116-conv_img_q.npy",
                                  "npy/conv116-conv_img_i16.npy",
                                  "npy/conv116-conv_weight.npy",
                                  "npy/conv116-conv_weight_q.npy",
                                  "npy/conv116-conv_weight_i16.npy",
                                  "npy/conv116-conv_out.npy",
                                  "npy/conv116-conv_out_i32.npy",
                                  "npy/conv116-conv_out_i32_dq.npy")

out = layer_dump.BN(model.darknet.dark4[4].conv2.bn, out,
                   "npy/conv116-bn_A.npy",
                   "npy/conv116-bn_B.npy",
                   "npy/conv116-bn_out.npy")
out = layer_dump.Relu(model.darknet.dark4[4].conv2.relu,out,
                      "npy/conv116-relu_out.npy")


out1 = out + out1
np.save("npy/eltwise119-add_out.npy", out1.cpu().detach().numpy())


#block5
out = layer_dump.QuantConv2d_save_i16(model.darknet.dark4[5].conv1.conv,out1,
                                  "npy/conv120-conv_img_q.npy",
                                  "npy/conv120-conv_img_i16.npy",
                                  "npy/conv120-conv_weight.npy",
                                  "npy/conv120-conv_weight_q.npy",
                                  "npy/conv120-conv_weight_i16.npy",
                                  "npy/conv120-conv_out.npy",
                                  "npy/conv120-conv_out_i32.npy",
                                  "npy/conv120-conv_out_i32_dq.npy")

out = layer_dump.BN(model.darknet.dark4[5].conv1.bn, out,
                   "npy/conv120-bn_A.npy",
                   "npy/conv120-bn_B.npy",
                   "npy/conv120-bn_out.npy")
out = layer_dump.Relu(model.darknet.dark4[5].conv1.relu,out,
                      "npy/conv120-relu_out.npy")

out = layer_dump.QuantConv2d_save_i16(model.darknet.dark4[5].conv2.conv,out,
                                  "npy/conv123-conv_img_q.npy",
                                  "npy/conv123-conv_img_i16.npy",
                                  "npy/conv123-conv_weight.npy",
                                  "npy/conv123-conv_weight_q.npy",
                                  "npy/conv123-conv_weight_i16.npy",
                                  "npy/conv123-conv_out.npy",
                                  "npy/conv123-conv_out_i32.npy",
                                  "npy/conv123-conv_out_i32_dq.npy")

out = layer_dump.BN(model.darknet.dark4[5].conv2.bn, out,
                   "npy/conv123-bn_A.npy",
                   "npy/conv123-bn_B.npy",
                   "npy/conv123-bn_out.npy")
out = layer_dump.Relu(model.darknet.dark4[5].conv2.relu,out,
                      "npy/conv123-relu_out.npy")


out1 = out + out1
np.save("npy/eltwise126-add_out.npy", out1.cpu().detach().numpy())

#block6
out = layer_dump.QuantConv2d_save_i16(model.darknet.dark4[6].conv1.conv,out1,
                                  "npy/conv127-conv_img_q.npy",
                                  "npy/conv127-conv_img_i16.npy",
                                  "npy/conv127-conv_weight.npy",
                                  "npy/conv127-conv_weight_q.npy",
                                  "npy/conv127-conv_weight_i16.npy",
                                  "npy/conv127-conv_out.npy",
                                  "npy/conv127-conv_out_i32.npy",
                                  "npy/conv127-conv_out_i32_dq.npy")

out = layer_dump.BN(model.darknet.dark4[6].conv1.bn, out,
                   "npy/conv127-bn_A.npy",
                   "npy/conv127-bn_B.npy",
                   "npy/conv127-bn_out.npy")
out = layer_dump.Relu(model.darknet.dark4[6].conv1.relu,out,
                      "npy/conv127-relu_out.npy")

out = layer_dump.QuantConv2d_save_i16(model.darknet.dark4[6].conv2.conv,out,
                                  "npy/conv130-conv_img_q.npy",
                                  "npy/conv130-conv_img_i16.npy",
                                  "npy/conv130-conv_weight.npy",
                                  "npy/conv130-conv_weight_q.npy",
                                  "npy/conv130-conv_weight_i16.npy",
                                  "npy/conv130-conv_out.npy",
                                  "npy/conv130-conv_out_i32.npy",
                                  "npy/conv130-conv_out_i32_dq.npy")

out = layer_dump.BN(model.darknet.dark4[6].conv2.bn, out,
                   "npy/conv130-bn_A.npy",
                   "npy/conv130-bn_B.npy",
                   "npy/conv130-bn_out.npy")
out = layer_dump.Relu(model.darknet.dark4[6].conv2.relu,out,
                      "npy/conv130-relu_out.npy")


out1 = out + out1
np.save("npy/eltwise133-add_out.npy", out1.cpu().detach().numpy())


#block7
out = layer_dump.QuantConv2d_save_i16(model.darknet.dark4[7].conv1.conv,out1,
                                  "npy/conv134-conv_img_q.npy",
                                  "npy/conv134-conv_img_i16.npy",
                                  "npy/conv134-conv_weight.npy",
                                  "npy/conv134-conv_weight_q.npy",
                                  "npy/conv134-conv_weight_i16.npy",
                                  "npy/conv134-conv_out.npy",
                                  "npy/conv134-conv_out_i32.npy",
                                  "npy/conv134-conv_out_i32_dq.npy")

out = layer_dump.BN(model.darknet.dark4[7].conv1.bn, out,
                   "npy/conv134-bn_A.npy",
                   "npy/conv134-bn_B.npy",
                   "npy/conv134-bn_out.npy")
out = layer_dump.Relu(model.darknet.dark4[7].conv1.relu,out,
                      "npy/conv134-relu_out.npy")

out = layer_dump.QuantConv2d_save_i16(model.darknet.dark4[7].conv2.conv,out,
                                  "npy/conv137-conv_img_q.npy",
                                  "npy/conv137-conv_img_i16.npy",
                                  "npy/conv137-conv_weight.npy",
                                  "npy/conv137-conv_weight_q.npy",
                                  "npy/conv137-conv_weight_i16.npy",
                                  "npy/conv137-conv_out.npy",
                                  "npy/conv137-conv_out_i32.npy",
                                  "npy/conv137-conv_out_i32_dq.npy")

out = layer_dump.BN(model.darknet.dark4[7].conv2.bn, out,
                   "npy/conv137-bn_A.npy",
                   "npy/conv137-bn_B.npy",
                   "npy/conv137-bn_out.npy")
out = layer_dump.Relu(model.darknet.dark4[7].conv2.relu,out,
                      "npy/conv137-relu_out.npy")


out1 = out + out1
np.save("npy/eltwise140-add_out.npy", out1.cpu().detach().numpy())

#block8
out = layer_dump.QuantConv2d_save_i16(model.darknet.dark4[8].conv1.conv,out1,
                                  "npy/conv141-conv_img_q.npy",
                                  "npy/conv141-conv_img_i16.npy",
                                  "npy/conv141-conv_weight.npy",
                                  "npy/conv141-conv_weight_q.npy",
                                  "npy/conv141-conv_weight_i16.npy",
                                  "npy/conv141-conv_out.npy",
                                  "npy/conv141-conv_out_i32.npy",
                                  "npy/conv141-conv_out_i32_dq.npy")

out = layer_dump.BN(model.darknet.dark4[8].conv1.bn, out,
                   "npy/conv141-bn_A.npy",
                   "npy/conv141-bn_B.npy",
                   "npy/conv141-bn_out.npy")
out = layer_dump.Relu(model.darknet.dark4[8].conv1.relu,out,
                      "npy/conv141-relu_out.npy")

out = layer_dump.QuantConv2d_save_i16(model.darknet.dark4[8].conv2.conv,out,
                                  "npy/conv144-conv_img_q.npy",
                                  "npy/conv144-conv_img_i16.npy",
                                  "npy/conv144-conv_weight.npy",
                                  "npy/conv144-conv_weight_q.npy",
                                  "npy/conv144-conv_weight_i16.npy",
                                  "npy/conv144-conv_out.npy",
                                  "npy/conv144-conv_out_i32.npy",
                                  "npy/conv144-conv_out_i32_dq.npy")

out = layer_dump.BN(model.darknet.dark4[8].conv2.bn, out,
                   "npy/conv144-bn_A.npy",
                   "npy/conv144-bn_B.npy",
                   "npy/conv144-bn_out.npy")
out = layer_dump.Relu(model.darknet.dark4[8].conv2.relu,out,
                      "npy/conv144-relu_out.npy")


out1 = out + out1
np.save("npy/eltwise147-add_out.npy", out1.cpu().detach().numpy())

out4 = out1

#dark5
out = layer_dump.QuantConv2d_save_i16(model.darknet.dark5[0].conv,out1,
                                  "npy/conv148-conv_img_q.npy",
                                  "npy/conv148-conv_img_i16.npy",
                                  "npy/conv148-conv_weight.npy",
                                  "npy/conv148-conv_weight_q.npy",
                                  "npy/conv148-conv_weight_i16.npy",
                                  "npy/conv148-conv_out.npy",
                                  "npy/conv148-conv_out_i32.npy",
                                  "npy/conv148-conv_out_i32_dq.npy")

out = layer_dump.BN(model.darknet.dark5[0].bn, out,
                   "npy/conv148-bn_A.npy",
                   "npy/conv148-bn_B.npy",
                   "npy/conv148-bn_out.npy")
out1 = layer_dump.Relu(model.darknet.dark5[0].relu,out,
                      "npy/conv148-relu_out.npy")


#block1
out = layer_dump.QuantConv2d_save_i16(model.darknet.dark5[1].conv1.conv,out1,
                                  "npy/conv151-conv_img_q.npy",
                                  "npy/conv151-conv_img_i16.npy",
                                  "npy/conv151-conv_weight.npy",
                                  "npy/conv151-conv_weight_q.npy",
                                  "npy/conv151-conv_weight_i16.npy",
                                  "npy/conv151-conv_out.npy",
                                  "npy/conv151-conv_out_i32.npy",
                                  "npy/conv151-conv_out_i32_dq.npy")

out = layer_dump.BN(model.darknet.dark5[1].conv1.bn, out,
                   "npy/conv151-bn_A.npy",
                   "npy/conv151-bn_B.npy",
                   "npy/conv151-bn_out.npy")
out = layer_dump.Relu(model.darknet.dark5[1].conv1.relu,out,
                      "npy/conv151-relu_out.npy")

out = layer_dump.QuantConv2d_save_i16(model.darknet.dark5[1].conv2.conv,out,
                                  "npy/conv154-conv_img_q.npy",
                                  "npy/conv154-conv_img_i16.npy",
                                  "npy/conv154-conv_weight.npy",
                                  "npy/conv154-conv_weight_q.npy",
                                  "npy/conv154-conv_weight_i16.npy",
                                  "npy/conv154-conv_out.npy",
                                  "npy/conv154-conv_out_i32.npy",
                                  "npy/conv154-conv_out_i32_dq.npy")

out = layer_dump.BN(model.darknet.dark5[1].conv2.bn, out,
                   "npy/conv154-bn_A.npy",
                   "npy/conv154-bn_B.npy",
                   "npy/conv154-bn_out.npy")
out = layer_dump.Relu(model.darknet.dark5[1].conv2.relu,out,
                      "npy/conv154-relu_out.npy")


out1 = out + out1
np.save("npy/eltwise157-add_out.npy", out1.cpu().detach().numpy())


#block2
out = layer_dump.QuantConv2d_save_i16(model.darknet.dark5[2].conv1.conv,out1,
                                  "npy/conv158-conv_img_q.npy",
                                  "npy/conv158-conv_img_i16.npy",
                                  "npy/conv158-conv_weight.npy",
                                  "npy/conv158-conv_weight_q.npy",
                                  "npy/conv158-conv_weight_i16.npy",
                                  "npy/conv158-conv_out.npy",
                                  "npy/conv158-conv_out_i32.npy",
                                  "npy/conv158-conv_out_i32_dq.npy")

out = layer_dump.BN(model.darknet.dark5[2].conv1.bn, out,
                   "npy/conv158-bn_A.npy",
                   "npy/conv158-bn_B.npy",
                   "npy/conv158-bn_out.npy")
out = layer_dump.Relu(model.darknet.dark5[2].conv1.relu,out,
                      "npy/conv158-relu_out.npy")

out = layer_dump.QuantConv2d_save_i16(model.darknet.dark5[2].conv2.conv,out,
                                  "npy/conv161-conv_img_q.npy",
                                  "npy/conv161-conv_img_i16.npy",
                                  "npy/conv161-conv_weight.npy",
                                  "npy/conv161-conv_weight_q.npy",
                                  "npy/conv161-conv_weight_i16.npy",
                                  "npy/conv161-conv_out.npy",
                                  "npy/conv161-conv_out_i32.npy",
                                  "npy/conv161-conv_out_i32_dq.npy")

out = layer_dump.BN(model.darknet.dark5[2].conv2.bn, out,
                   "npy/conv161-bn_A.npy",
                   "npy/conv161-bn_B.npy",
                   "npy/conv161-bn_out.npy")
out = layer_dump.Relu(model.darknet.dark5[2].conv2.relu,out,
                      "npy/conv161-relu_out.npy")


out1 = out + out1
np.save("npy/eltwise164-add_out.npy", out1.cpu().detach().numpy())

#block3
out = layer_dump.QuantConv2d_save_i16(model.darknet.dark5[3].conv1.conv,out1,
                                  "npy/conv165-conv_img_q.npy",
                                  "npy/conv165-conv_img_i16.npy",
                                  "npy/conv165-conv_weight.npy",
                                  "npy/conv165-conv_weight_q.npy",
                                  "npy/conv165-conv_weight_i16.npy",
                                  "npy/conv165-conv_out.npy",
                                  "npy/conv165-conv_out_i32.npy",
                                  "npy/conv165-conv_out_i32_dq.npy")

out = layer_dump.BN(model.darknet.dark5[3].conv1.bn, out,
                   "npy/conv165-bn_A.npy",
                   "npy/conv165-bn_B.npy",
                   "npy/conv165-bn_out.npy")
out = layer_dump.Relu(model.darknet.dark5[3].conv1.relu,out,
                      "npy/conv165-relu_out.npy")

out = layer_dump.QuantConv2d_save_i16(model.darknet.dark5[3].conv2.conv,out,
                                  "npy/conv168-conv_img_q.npy",
                                  "npy/conv168-conv_img_i16.npy",
                                  "npy/conv168-conv_weight.npy",
                                  "npy/conv168-conv_weight_q.npy",
                                  "npy/conv168-conv_weight_i16.npy",
                                  "npy/conv168-conv_out.npy",
                                  "npy/conv168-conv_out_i32.npy",
                                  "npy/conv168-conv_out_i32_dq.npy")

out = layer_dump.BN(model.darknet.dark5[3].conv2.bn, out,
                   "npy/conv168-bn_A.npy",
                   "npy/conv168-bn_B.npy",
                   "npy/conv168-bn_out.npy")
out = layer_dump.Relu(model.darknet.dark5[3].conv2.relu,out,
                      "npy/conv168-relu_out.npy")


out1 = out + out1
np.save("npy/eltwise171-add_out.npy", out1.cpu().detach().numpy())

#block4
out = layer_dump.QuantConv2d_save_i16(model.darknet.dark5[4].conv1.conv,out1,
                                  "npy/conv172-conv_img_q.npy",
                                  "npy/conv172-conv_img_i16.npy",
                                  "npy/conv172-conv_weight.npy",
                                  "npy/conv172-conv_weight_q.npy",
                                  "npy/conv172-conv_weight_i16.npy",
                                  "npy/conv172-conv_out.npy",
                                  "npy/conv172-conv_out_i32.npy",
                                  "npy/conv172-conv_out_i32_dq.npy")

out = layer_dump.BN(model.darknet.dark5[4].conv1.bn, out,
                   "npy/conv172-bn_A.npy",
                   "npy/conv172-bn_B.npy",
                   "npy/conv172-bn_out.npy")
out = layer_dump.Relu(model.darknet.dark5[4].conv1.relu,out,
                      "npy/conv172-relu_out.npy")

out = layer_dump.QuantConv2d_save_i16(model.darknet.dark5[4].conv2.conv,out,
                                  "npy/conv175-conv_img_q.npy",
                                  "npy/conv175-conv_img_i16.npy",
                                  "npy/conv175-conv_weight.npy",
                                  "npy/conv175-conv_weight_q.npy",
                                  "npy/conv175-conv_weight_i16.npy",
                                  "npy/conv175-conv_out.npy",
                                  "npy/conv175-conv_out_i32.npy",
                                  "npy/conv175-conv_out_i32_dq.npy")

out = layer_dump.BN(model.darknet.dark5[4].conv2.bn, out,
                   "npy/conv175-bn_A.npy",
                   "npy/conv175-bn_B.npy",
                   "npy/conv175-bn_out.npy")
out = layer_dump.Relu(model.darknet.dark5[4].conv2.relu,out,
                      "npy/conv175-relu_out.npy")


out1 = out + out1
np.save("npy/eltwise178-add_out.npy", out1.cpu().detach().numpy())

out5 = out1

#OUT0
out = layer_dump.QuantConv2d_save_i16(model.feature1.conv1.conv,out1,
                                  "npy/conv179-conv_img_q.npy",
                                  "npy/conv179-conv_img_i16.npy",
                                  "npy/conv179-conv_weight.npy",
                                  "npy/conv179-conv_weight_q.npy",
                                  "npy/conv179-conv_weight_i16.npy",
                                  "npy/conv179-conv_out.npy",
                                  "npy/conv179-conv_out_i32.npy",
                                  "npy/conv179-conv_out_i32_dq.npy")

out = layer_dump.BN(model.feature1.conv1.bn, out,
                   "npy/conv179-bn_A.npy",
                   "npy/conv179-bn_B.npy",
                   "npy/conv179-bn_out.npy")
out = layer_dump.Relu(model.feature1.conv1.relu,out,
                      "npy/conv179-relu_out.npy")

out = layer_dump.QuantConv2d_save_i16(model.feature1.conv2.conv,out,
                                  "npy/conv182-conv_img_q.npy",
                                  "npy/conv182-conv_img_i16.npy",
                                  "npy/conv182-conv_weight.npy",
                                  "npy/conv182-conv_weight_q.npy",
                                  "npy/conv182-conv_weight_i16.npy",
                                  "npy/conv182-conv_out.npy",
                                  "npy/conv182-conv_out_i32.npy",
                                  "npy/conv182-conv_out_i32_dq.npy")

out = layer_dump.BN(model.feature1.conv2.bn, out,
                   "npy/conv182-bn_A.npy",
                   "npy/conv182-bn_B.npy",
                   "npy/conv182-bn_out.npy")
out = layer_dump.Relu(model.feature1.conv2.relu,out,
                      "npy/conv182-relu_out.npy")

out = layer_dump.QuantConv2d_save_i16(model.feature1.conv3.conv,out,
                                  "npy/conv185-conv_img_q.npy",
                                  "npy/conv185-conv_img_i16.npy",
                                  "npy/conv185-conv_weight.npy",
                                  "npy/conv185-conv_weight_q.npy",
                                  "npy/conv185-conv_weight_i16.npy",
                                  "npy/conv185-conv_out.npy",
                                  "npy/conv185-conv_out_i32.npy",
                                  "npy/conv185-conv_out_i32_dq.npy")

out = layer_dump.BN(model.feature1.conv3.bn, out,
                   "npy/conv185-bn_A.npy",
                   "npy/conv185-bn_B.npy",
                   "npy/conv185-bn_out.npy")
out = layer_dump.Relu(model.feature1.conv3.relu,out,
                      "npy/conv185-relu_out.npy")


out = layer_dump.QuantConv2d_save_i16(model.feature1.conv4.conv,out,
                                  "npy/conv188-conv_img_q.npy",
                                  "npy/conv188-conv_img_i16.npy",
                                  "npy/conv188-conv_weight.npy",
                                  "npy/conv188-conv_weight_q.npy",
                                  "npy/conv188-conv_weight_i16.npy",
                                  "npy/conv188-conv_out.npy",
                                  "npy/conv188-conv_out_i32.npy",
                                  "npy/conv188-conv_out_i32_dq.npy")

out = layer_dump.BN(model.feature1.conv4.bn, out,
                   "npy/conv188-bn_A.npy",
                   "npy/conv188-bn_B.npy",
                   "npy/conv188-bn_out.npy")
out = layer_dump.Relu(model.feature1.conv4.relu,out,
                      "npy/conv188-relu_out.npy")

out = layer_dump.QuantConv2d_save_i16(model.feature1.conv5.conv,out,
                                  "npy/conv191-conv_img_q.npy",
                                  "npy/conv191-conv_img_i16.npy",
                                  "npy/conv191-conv_weight.npy",
                                  "npy/conv191-conv_weight_q.npy",
                                  "npy/conv191-conv_weight_i16.npy",
                                  "npy/conv191-conv_out.npy",
                                  "npy/conv191-conv_out_i32.npy",
                                  "npy/conv191-conv_out_i32_dq.npy")

out = layer_dump.BN(model.feature1.conv5.bn, out,
                   "npy/conv191-bn_A.npy",
                   "npy/conv191-bn_B.npy",
                   "npy/conv191-bn_out.npy")
out = layer_dump.Relu(model.feature1.conv5.relu,out,
                      "npy/conv191-relu_out.npy")

feature1 = out

out = layer_dump.QuantConv2d_save_i16(model.out_head1.conv1.conv,out,
                                  "npy/conv194-conv_img_q.npy",
                                  "npy/conv194-conv_img_i16.npy",
                                  "npy/conv194-conv_weight.npy",
                                  "npy/conv194-conv_weight_q.npy",
                                  "npy/conv194-conv_weight_i16.npy",
                                  "npy/conv194-conv_out.npy",
                                  "npy/conv194-conv_out_i32.npy",
                                  "npy/conv194-conv_out_i32_dq.npy")

out = layer_dump.BN(model.out_head1.conv1.bn, out,
                   "npy/conv194-bn_A.npy",
                   "npy/conv194-bn_B.npy",
                   "npy/conv194-bn_out.npy")
out = layer_dump.Relu(model.out_head1.conv1.relu,out,
                      "npy/conv194-relu_out.npy")

out = layer_dump.QuantConv2d_save_i16(model.out_head1.conv2.conv,out,
                                  "npy/conv197-conv_img_q.npy",
                                  "npy/conv197-conv_img_i16.npy",
                                  "npy/conv197-conv_weight.npy",
                                  "npy/conv197-conv_weight_q.npy",
                                  "npy/conv197-conv_weight_i16.npy",
                                  "npy/conv197-conv_out.npy",
                                  "npy/conv197-conv_out_i32.npy",
                                  "npy/conv197-conv_out_i32_dq.npy")

out_large = out


#Upsample
out = layer_dump.QuantConv2d_save_i16(model.cbl1.conv,feature1,
                                  "npy/conv199-conv_img_q.npy",
                                  "npy/conv199-conv_img_i16.npy",
                                  "npy/conv199-conv_weight.npy",
                                  "npy/conv199-conv_weight_q.npy",
                                  "npy/conv199-conv_weight_i16.npy",
                                  "npy/conv199-conv_out.npy",
                                  "npy/conv199-conv_out_i32.npy",
                                  "npy/conv199-conv_out_i32_dq.npy")

out = layer_dump.BN(model.cbl1.bn, out,
                   "npy/conv199-bn_A.npy",
                   "npy/conv199-bn_B.npy",
                   "npy/conv199-bn_out.npy")
out = layer_dump.Relu(model.cbl1.relu,out,
                      "npy/conv199-relu_out.npy")

sample = nn.Upsample(scale_factor=2, mode='bilinear')
out = sample(out)
np.save("npy/interp202-bilinear_out.npy", out.cpu().detach().numpy())


x1_in = torch.cat([out, out4], 1)
np.save("npy/concat203-out",x1_in.cpu().detach().numpy())



#OUT1
out = layer_dump.QuantConv2d_save_i16(model.feature2.conv1.conv,x1_in,
                                  "npy/conv204-conv_img_q.npy",
                                  "npy/conv204-conv_img_i16.npy",
                                  "npy/conv204-conv_weight.npy",
                                  "npy/conv204-conv_weight_q.npy",
                                  "npy/conv204-conv_weight_i16.npy",
                                  "npy/conv204-conv_out.npy",
                                  "npy/conv204-conv_out_i32.npy",
                                  "npy/conv204-conv_out_i32_dq.npy")

out = layer_dump.BN(model.feature2.conv1.bn, out,
                   "npy/conv204-bn_A.npy",
                   "npy/conv204-bn_B.npy",
                   "npy/conv204-bn_out.npy")
out = layer_dump.Relu(model.feature2.conv1.relu,out,
                      "npy/conv204-relu_out.npy")


out = layer_dump.QuantConv2d_save_i16(model.feature2.conv2.conv,out,
                                  "npy/conv207-conv_img_q.npy",
                                  "npy/conv207-conv_img_i16.npy",
                                  "npy/conv207-conv_weight.npy",
                                  "npy/conv207-conv_weight_q.npy",
                                  "npy/conv207-conv_weight_i16.npy",
                                  "npy/conv207-conv_out.npy",
                                  "npy/conv207-conv_out_i32.npy",
                                  "npy/conv207-conv_out_i32_dq.npy")

out = layer_dump.BN(model.feature2.conv2.bn, out,
                   "npy/conv207-bn_A.npy",
                   "npy/conv207-bn_B.npy",
                   "npy/conv207-bn_out.npy")
out = layer_dump.Relu(model.feature2.conv2.relu,out,
                      "npy/conv207-relu_out.npy")


out = layer_dump.QuantConv2d_save_i16(model.feature2.conv3.conv,out,
                                  "npy/conv210-conv_img_q.npy",
                                  "npy/conv210-conv_img_i16.npy",
                                  "npy/conv210-conv_weight.npy",
                                  "npy/conv210-conv_weight_q.npy",
                                  "npy/conv210-conv_weight_i16.npy",
                                  "npy/conv210-conv_out.npy",
                                  "npy/conv210-conv_out_i32.npy",
                                  "npy/conv210-conv_out_i32_dq.npy")

out = layer_dump.BN(model.feature2.conv3.bn, out,
                   "npy/conv210-bn_A.npy",
                   "npy/conv210-bn_B.npy",
                   "npy/conv210-bn_out.npy")
out = layer_dump.Relu(model.feature2.conv3.relu,out,
                      "npy/conv210-relu_out.npy")

out = layer_dump.QuantConv2d_save_i16(model.feature2.conv4.conv,out,
                                  "npy/conv213-conv_img_q.npy",
                                  "npy/conv213-conv_img_i16.npy",
                                  "npy/conv213-conv_weight.npy",
                                  "npy/conv213-conv_weight_q.npy",
                                  "npy/conv213-conv_weight_i16.npy",
                                  "npy/conv213-conv_out.npy",
                                  "npy/conv213-conv_out_i32.npy",
                                  "npy/conv213-conv_out_i32_dq.npy")

out = layer_dump.BN(model.feature2.conv4.bn, out,
                   "npy/conv213-bn_A.npy",
                   "npy/conv213-bn_B.npy",
                   "npy/conv213-bn_out.npy")
out = layer_dump.Relu(model.feature2.conv4.relu,out,
                      "npy/conv213-relu_out.npy")



out = layer_dump.QuantConv2d_save_i16(model.feature2.conv5.conv,out,
                                  "npy/conv216-conv_img_q.npy",
                                  "npy/conv216-conv_img_i16.npy",
                                  "npy/conv216-conv_weight.npy",
                                  "npy/conv216-conv_weight_q.npy",
                                  "npy/conv216-conv_weight_i16.npy",
                                  "npy/conv216-conv_out.npy",
                                  "npy/conv216-conv_out_i32.npy",
                                  "npy/conv216-conv_out_i32_dq.npy")

out = layer_dump.BN(model.feature2.conv5.bn, out,
                   "npy/conv216-bn_A.npy",
                   "npy/conv216-bn_B.npy",
                   "npy/conv216-bn_out.npy")
out = layer_dump.Relu(model.feature2.conv5.relu,out,
                      "npy/conv216-relu_out.npy")

feature2 = out


out = layer_dump.QuantConv2d_save_i16(model.out_head2.conv1.conv,out,
                                  "npy/conv219-conv_img_q.npy",
                                  "npy/conv219-conv_img_i16.npy",
                                  "npy/conv219-conv_weight.npy",
                                  "npy/conv219-conv_weight_q.npy",
                                  "npy/conv219-conv_weight_i16.npy",
                                  "npy/conv219-conv_out.npy",
                                  "npy/conv219-conv_out_i32.npy",
                                  "npy/conv219-conv_out_i32_dq.npy")

out = layer_dump.BN(model.out_head2.conv1.bn, out,
                   "npy/conv219-bn_A.npy",
                   "npy/conv219-bn_B.npy",
                   "npy/conv219-bn_out.npy")
out = layer_dump.Relu(model.out_head2.conv1.relu,out,
                      "npy/conv219-relu_out.npy")


out = layer_dump.QuantConv2d_save_i16(model.out_head2.conv2.conv,out,
                                  "npy/conv222-conv_img_q.npy",
                                  "npy/conv222-conv_img_i16.npy",
                                  "npy/conv222-conv_weight.npy",
                                  "npy/conv222-conv_weight_q.npy",
                                  "npy/conv222-conv_weight_i16.npy",
                                  "npy/conv222-conv_out.npy",
                                  "npy/conv222-conv_out_i32.npy",
                                  "npy/conv222-conv_out_i32_dq.npy")

out_medium = out

#Upsample
out = layer_dump.QuantConv2d_save_i16(model.cbl2.conv,feature2,
                                  "npy/conv224-conv_img_q.npy",
                                  "npy/conv224-conv_img_i16.npy",
                                  "npy/conv224-conv_weight.npy",
                                  "npy/conv224-conv_weight_q.npy",
                                  "npy/conv224-conv_weight_i16.npy",
                                  "npy/conv224-conv_out.npy",
                                  "npy/conv224-conv_out_i32.npy",
                                  "npy/conv224-conv_out_i32_dq.npy")

out = layer_dump.BN(model.cbl2.bn, out,
                   "npy/conv224-bn_A.npy",
                   "npy/conv224-bn_B.npy",
                   "npy/conv224-bn_out.npy")
out = layer_dump.Relu(model.cbl2.relu,out,
                      "npy/conv224-relu_out.npy")



out = sample(out)
np.save("npy/interp227-bilinear_out.npy", out.cpu().detach().numpy())


x2_in = torch.cat([out, out3], 1)
np.save("npy/concat228-out",x1_in.cpu().detach().numpy())


#OUT2
out = layer_dump.QuantConv2d_save_i16(model.feature3.conv1.conv,x2_in,
                                  "npy/conv229-conv_img_q.npy",
                                  "npy/conv229-conv_img_i16.npy",
                                  "npy/conv229-conv_weight.npy",
                                  "npy/conv229-conv_weight_q.npy",
                                  "npy/conv229-conv_weight_i16.npy",
                                  "npy/conv229-conv_out.npy",
                                  "npy/conv229-conv_out_i32.npy",
                                  "npy/conv229-conv_out_i32_dq.npy")

out = layer_dump.BN(model.feature3.conv1.bn, out,
                   "npy/conv229-bn_A.npy",
                   "npy/conv229-bn_B.npy",
                   "npy/conv229-bn_out.npy")


out = layer_dump.Relu(model.feature3.conv1.relu,out,
                      "npy/conv229-relu_out.npy")

out = layer_dump.QuantConv2d_save_i16(model.feature3.conv2.conv,out,
                                  "npy/conv232-conv_img_q.npy",
                                  "npy/conv232-conv_img_i16.npy",
                                  "npy/conv232-conv_weight.npy",
                                  "npy/conv232-conv_weight_q.npy",
                                  "npy/conv232-conv_weight_i16.npy",
                                  "npy/conv232-conv_out.npy",
                                  "npy/conv232-conv_out_i32.npy",
                                  "npy/conv232-conv_out_i32_dq.npy")

out = layer_dump.BN(model.feature3.conv2.bn, out,
                   "npy/conv232-bn_A.npy",
                   "npy/conv232-bn_B.npy",
                   "npy/conv232-bn_out.npy")
out = layer_dump.Relu(model.feature3.conv2.relu,out,
                      "npy/conv232-relu_out.npy")



out = layer_dump.QuantConv2d_save_i16(model.feature3.conv3.conv,out,
                                  "npy/conv235-conv_img_q.npy",
                                  "npy/conv235-conv_img_i16.npy",
                                  "npy/conv235-conv_weight.npy",
                                  "npy/conv235-conv_weight_q.npy",
                                  "npy/conv235-conv_weight_i16.npy",
                                  "npy/conv235-conv_out.npy",
                                  "npy/conv235-conv_out_i32.npy",
                                  "npy/conv235-conv_out_i32_dq.npy")

out = layer_dump.BN(model.feature3.conv3.bn, out,
                   "npy/conv235-bn_A.npy",
                   "npy/conv235-bn_B.npy",
                   "npy/conv235-bn_out.npy")
out = layer_dump.Relu(model.feature3.conv3.relu,out,
                      "npy/conv235-relu_out.npy")


out = layer_dump.QuantConv2d_save_i16(model.feature3.conv4.conv,out,
                                  "npy/conv238-conv_img_q.npy",
                                  "npy/conv238-conv_img_i16.npy",
                                  "npy/conv238-conv_weight.npy",
                                  "npy/conv238-conv_weight_q.npy",
                                  "npy/conv238-conv_weight_i16.npy",
                                  "npy/conv238-conv_out.npy",
                                  "npy/conv238-conv_out_i32.npy",
                                  "npy/conv238-conv_out_i32_dq.npy")

out = layer_dump.BN(model.feature3.conv4.bn, out,
                   "npy/conv238-bn_A.npy",
                   "npy/conv238-bn_B.npy",
                   "npy/conv238-bn_out.npy")
out = layer_dump.Relu(model.feature3.conv4.relu,out,
                      "npy/conv238-relu_out.npy")


out = layer_dump.QuantConv2d_save_i16(model.feature3.conv5.conv,out,
                                  "npy/conv241-conv_img_q.npy",
                                  "npy/conv241-conv_img_i16.npy",
                                  "npy/conv241-conv_weight.npy",
                                  "npy/conv241-conv_weight_q.npy",
                                  "npy/conv241-conv_weight_i16.npy",
                                  "npy/conv241-conv_out.npy",
                                  "npy/conv241-conv_out_i32.npy",
                                  "npy/conv241-conv_out_i32_dq.npy")

out = layer_dump.BN(model.feature3.conv5.bn, out,
                   "npy/conv241-bn_A.npy",
                   "npy/conv241-bn_B.npy",
                   "npy/conv241-bn_out.npy")
out = layer_dump.Relu(model.feature3.conv5.relu,out,
                      "npy/conv241-relu_out.npy")


out = layer_dump.QuantConv2d_save_i16(model.out_head3.conv1.conv,out,
                                  "npy/conv244-conv_img_q.npy",
                                  "npy/conv244-conv_img_i16.npy",
                                  "npy/conv244-conv_weight.npy",
                                  "npy/conv244-conv_weight_q.npy",
                                  "npy/conv244-conv_weight_i16.npy",
                                  "npy/conv244-conv_out.npy",
                                  "npy/conv244-conv_out_i32.npy",
                                  "npy/conv244-conv_out_i32_dq.npy")

out = layer_dump.BN(model.out_head3.conv1.bn, out,
                   "npy/conv244-bn_A.npy",
                   "npy/conv244-bn_B.npy",
                   "npy/conv244-bn_out.npy")
out = layer_dump.Relu(model.out_head3.conv1.relu,out,
                      "npy/conv244-relu_out.npy")


out = layer_dump.QuantConv2d_save_i16(model.out_head3.conv2.conv,out,
                                  "npy/conv247-conv_img_q.npy",
                                  "npy/conv247-conv_img_i16.npy",
                                  "npy/conv247-conv_weight.npy",
                                  "npy/conv247-conv_weight_q.npy",
                                  "npy/conv247-conv_weight_i16.npy",
                                  "npy/conv247-conv_out.npy",
                                  "npy/conv247-conv_out_i32.npy",
                                  "npy/conv247-conv_out_i32_dq.npy")

out_small = out

