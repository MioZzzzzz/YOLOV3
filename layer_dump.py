import torch
import numpy as np
from pytorch_quantization.tensor_quant import _tensor_quant
import torch.nn.functional as F
from weighted_deviation import mat_deviation as mat_devi
import math

#import sys
#sys.path.append(r"D:\BaiduNetdiskDownload\PyTorch\data_dump")

def Model( model, image, out_file ):
    output = model( image )

    np.save( out_file, output.cpu().detach().numpy() )
    return output

def QuantConv2d( conv, image):
    weight = conv.weight.data
    image_q, scale_i = _tensor_quant( image,
            conv.input_quantizer.amax, num_bits=16,
            unsigned=False, narrow_range=False )



    #print(scale_i)



    weight_q, scale_w = _tensor_quant( weight,
            conv.weight_quantizer.amax, num_bits=16,
            unsigned=False, narrow_range=False )



    output_q = F.conv2d( image_q, weight_q, conv.bias,
            conv.stride, conv.padding, conv.dilation,
            conv.groups)

    output_q = output_q.to(torch.float32)



    #print(output_q)

    batches = output_q.shape[0]
    channels= output_q.shape[1]

    scale_w = scale_w.reshape( (channels) )
    output = torch.ones( output_q.shape, dtype=torch.float32)
    for ch in range( 0, channels ):
        factor = scale_w[ch] * scale_i
        for bat in range( 0, batches ):
            output[bat][ch] = output_q[bat][ch] / factor
    return output

def QuantConv2d_save_i8( conv, image, scale_input, image_q_file,weight_file, scale_weight, weight_q_file, out_file, out_q_file,  scale_output):
    weight = conv.weight.data
    image_q, scale_i = _tensor_quant( image,
            conv.input_quantizer.amax, num_bits=8,
            unsigned=False, narrow_range=False )

    weight_q, scale_w = _tensor_quant( weight,
            conv.weight_quantizer.amax, num_bits=8,
            unsigned=False, narrow_range=False )

    output_q = F.conv2d( image_q, weight_q, conv.bias,
            conv.stride, conv.padding, conv.dilation,
            conv.groups)

    output_q = output_q.to(torch.float32)
    #print(output_q)

    batches = output_q.shape[0]
    channels= output_q.shape[1]

    scale_w = scale_w.reshape( (channels) )
    output = torch.ones( output_q.shape, dtype=torch.float32)
    out_scale = scale_w*scale_i
    for ch in range( 0, channels ):
        factor = scale_w[ch] * scale_i
        #print("out_dq:", factor)
        for bat in range( 0, batches ):
            output[bat][ch] = output_q[bat][ch] / factor

    np.save( image_q_file, image_q.to(torch.int8).cpu().detach().numpy() )
    np.save( weight_q_file, weight_q.to(torch.int8).cpu().detach().numpy() )

    np.save( out_q_file, output_q.to(torch.int32).cpu().detach().numpy() )
    np.save(scale_weight, scale_w.cpu().detach().numpy())
    np.save(scale_input, scale_i.cpu().detach().numpy())
    np.save(scale_output, out_scale.cpu().detach().numpy)
    np.save(weight_file, weight.cpu().detach().numpy())
    np.save(out_file, output.cpu().detach().numpy())
    return output


def QuantConv2d_save_i16( conv, image, scale_input, image_q_file,weight_file, scale_weight, weight_q_file, out_file, out_q_file,  scale_output):
    weight = conv.weight.data
    image_q, scale_i = _tensor_quant( image,
            conv.input_quantizer.amax, num_bits=16,
            unsigned=False, narrow_range=False )

    weight_q, scale_w = _tensor_quant( weight,
            conv.weight_quantizer.amax, num_bits=16,
            unsigned=False, narrow_range=False )

    output_q = F.conv2d( image_q, weight_q, conv.bias,
            conv.stride, conv.padding, conv.dilation,
            conv.groups)

    output_q = output_q.to(torch.float32)
    #print(output_q)

    batches = output_q.shape[0]
    channels= output_q.shape[1]

    scale_w = scale_w.reshape( (channels) )
    output = torch.ones( output_q.shape, dtype=torch.float32)
    out_scale = scale_w*scale_i
    for ch in range( 0, channels ):
        factor = scale_w[ch] * scale_i
        #print("out_dq:", factor)
        for bat in range( 0, batches ):
            output[bat][ch] = output_q[bat][ch] / factor

    np.save( image_q_file, image_q.to(torch.int16).cpu().detach().numpy() )
    np.save( weight_q_file, weight_q.to(torch.int16).cpu().detach().numpy() )

    np.save( out_q_file, output_q.to(torch.int32).cpu().detach().numpy() )
    np.save(scale_weight, scale_w.cpu().detach().numpy())
    np.save(scale_input, scale_i.cpu().detach().numpy())
    np.save(scale_output, out_scale.cpu().detach().numpy)
    np.save(weight_file, weight.cpu().detach().numpy())
    np.save(out_file, output.cpu().detach().numpy())
    return output




def BN( bn, image, A_file, B_file, out_file):
    output = bn( image )

    weight = bn.weight.data
    bias = bn.bias.data
    mean = bn.running_mean.data
    var = bn.running_var.data
    eps = 1e-3


    A = weight/torch.sqrt(var + eps)

    B = bias - (weight*mean)/torch.sqrt(var + eps)




    np.save(A_file, A.cpu().detach().numpy())
    np.save(B_file, B.cpu().detach().numpy())
    np.save( out_file, output.cpu().detach().numpy() )
    # np.save( weight_file, weight.cpu().detach().numpy() )
    # np.save( bias_file, bias.cpu().detach().numpy() )
    # np.save( mean_file, mean.cpu().detach().numpy() )
    # np.save( var_file, var.cpu().detach().numpy() )

    return output

def Relu( relu, image, out_file ):
    output = relu( image )

    np.save( out_file, output.cpu().detach().numpy() )
    return output



