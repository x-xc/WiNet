# Copyright (c) 2019, Adobe Inc. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International Public License. To view a copy of this license, visit
# https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.

"""
Consider the width, length or depth of images is even number,
"""
import torch
from torch.autograd import Function

class DWTFunction_3D(Function):
    @staticmethod
    def forward(ctx, input,
                matrix_Low_0, matrix_Low_1, matrix_Low_2,
                matrix_High_0, matrix_High_1, matrix_High_2):
        ctx.save_for_backward(matrix_Low_0, matrix_Low_1, matrix_Low_2,
                              matrix_High_0, matrix_High_1, matrix_High_2)
        L = torch.matmul(matrix_Low_0, input)
        H = torch.matmul(matrix_High_0, input)
        LL = torch.matmul(L, matrix_Low_1).transpose(dim0 = 2, dim1 = 3)
        LH = torch.matmul(L, matrix_High_1).transpose(dim0 = 2, dim1 = 3)
        HL = torch.matmul(H, matrix_Low_1).transpose(dim0 = 2, dim1 = 3)
        HH = torch.matmul(H, matrix_High_1).transpose(dim0 = 2, dim1 = 3)
        LLL = torch.matmul(matrix_Low_2, LL).transpose(dim0 = 2, dim1 = 3)
        LLH = torch.matmul(matrix_Low_2, LH).transpose(dim0 = 2, dim1 = 3)
        LHL = torch.matmul(matrix_Low_2, HL).transpose(dim0 = 2, dim1 = 3)
        LHH = torch.matmul(matrix_Low_2, HH).transpose(dim0 = 2, dim1 = 3)
        HLL = torch.matmul(matrix_High_2, LL).transpose(dim0 = 2, dim1 = 3)
        HLH = torch.matmul(matrix_High_2, LH).transpose(dim0 = 2, dim1 = 3)
        HHL = torch.matmul(matrix_High_2, HL).transpose(dim0 = 2, dim1 = 3)
        HHH = torch.matmul(matrix_High_2, HH).transpose(dim0 = 2, dim1 = 3)
        return LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH

    @staticmethod
    def backward(ctx, grad_LLL, grad_LLH, grad_LHL, grad_LHH,
                      grad_HLL, grad_HLH, grad_HHL, grad_HHH):
        matrix_Low_0, matrix_Low_1, matrix_Low_2, matrix_High_0, matrix_High_1, matrix_High_2 = ctx.saved_variables
        grad_LL = torch.add(torch.matmul(matrix_Low_2.t(), grad_LLL.transpose(dim0 = 2, dim1 = 3)), torch.matmul(matrix_High_2.t(), grad_HLL.transpose(dim0 = 2, dim1 = 3))).transpose(dim0 = 2, dim1 = 3)
        grad_LH = torch.add(torch.matmul(matrix_Low_2.t(), grad_LLH.transpose(dim0 = 2, dim1 = 3)), torch.matmul(matrix_High_2.t(), grad_HLH.transpose(dim0 = 2, dim1 = 3))).transpose(dim0 = 2, dim1 = 3)
        grad_HL = torch.add(torch.matmul(matrix_Low_2.t(), grad_LHL.transpose(dim0 = 2, dim1 = 3)), torch.matmul(matrix_High_2.t(), grad_HHL.transpose(dim0 = 2, dim1 = 3))).transpose(dim0 = 2, dim1 = 3)
        grad_HH = torch.add(torch.matmul(matrix_Low_2.t(), grad_LHH.transpose(dim0 = 2, dim1 = 3)), torch.matmul(matrix_High_2.t(), grad_HHH.transpose(dim0 = 2, dim1 = 3))).transpose(dim0 = 2, dim1 = 3)
        grad_L = torch.add(torch.matmul(grad_LL, matrix_Low_1.t()), torch.matmul(grad_LH, matrix_High_1.t()))
        grad_H = torch.add(torch.matmul(grad_HL, matrix_Low_1.t()), torch.matmul(grad_HH, matrix_High_1.t()))
        grad_input = torch.add(torch.matmul(matrix_Low_0.t(), grad_L), torch.matmul(matrix_High_0.t(), grad_H))
        return grad_input, None, None, None, None, None, None, None, None


class IDWTFunction_3D(Function):
    @staticmethod
    def forward(ctx, input_LLL, input_LLH, input_LHL, input_LHH,
                     input_HLL, input_HLH, input_HHL, input_HHH,
                     matrix_Low_0, matrix_Low_1, matrix_Low_2,
                     matrix_High_0, matrix_High_1, matrix_High_2):
        ctx.save_for_backward(matrix_Low_0, matrix_Low_1, matrix_Low_2,
                              matrix_High_0, matrix_High_1, matrix_High_2)
       
        input_LL = torch.add(torch.matmul(matrix_Low_2.t(), input_LLL.transpose(dim0 = 2, dim1 = 3)), torch.matmul(matrix_High_2.t(), input_HLL.transpose(dim0 = 2, dim1 = 3))).transpose(dim0 = 2, dim1 = 3)
        input_LH = torch.add(torch.matmul(matrix_Low_2.t(), input_LLH.transpose(dim0 = 2, dim1 = 3)), torch.matmul(matrix_High_2.t(), input_HLH.transpose(dim0 = 2, dim1 = 3))).transpose(dim0 = 2, dim1 = 3)
        input_HL = torch.add(torch.matmul(matrix_Low_2.t(), input_LHL.transpose(dim0 = 2, dim1 = 3)), torch.matmul(matrix_High_2.t(), input_HHL.transpose(dim0 = 2, dim1 = 3))).transpose(dim0 = 2, dim1 = 3)
        input_HH = torch.add(torch.matmul(matrix_Low_2.t(), input_LHH.transpose(dim0 = 2, dim1 = 3)), torch.matmul(matrix_High_2.t(), input_HHH.transpose(dim0 = 2, dim1 = 3))).transpose(dim0 = 2, dim1 = 3)
        input_L = torch.add(torch.matmul(input_LL, matrix_Low_1.t()), torch.matmul(input_LH, matrix_High_1.t()))
        input_H = torch.add(torch.matmul(input_HL, matrix_Low_1.t()), torch.matmul(input_HH, matrix_High_1.t()))
        output = torch.add(torch.matmul(matrix_Low_0.t(), input_L), torch.matmul(matrix_High_0.t(), input_H))
        return output
    @staticmethod
    def backward(ctx, grad_output):
        matrix_Low_0, matrix_Low_1, matrix_Low_2, matrix_High_0, matrix_High_1, matrix_High_2 = ctx.saved_variables
        grad_L = torch.matmul(matrix_Low_0, grad_output)
        grad_H = torch.matmul(matrix_High_0, grad_output)
        grad_LL = torch.matmul(grad_L, matrix_Low_1).transpose(dim0 = 2, dim1 = 3)
        grad_LH = torch.matmul(grad_L, matrix_High_1).transpose(dim0 = 2, dim1 = 3)
        grad_HL = torch.matmul(grad_H, matrix_Low_1).transpose(dim0 = 2, dim1 = 3)
        grad_HH = torch.matmul(grad_H, matrix_High_1).transpose(dim0 = 2, dim1 = 3)
        grad_LLL = torch.matmul(matrix_Low_2, grad_LL).transpose(dim0 = 2, dim1 = 3)
        grad_LLH = torch.matmul(matrix_Low_2, grad_LH).transpose(dim0 = 2, dim1 = 3)
        grad_LHL = torch.matmul(matrix_Low_2, grad_HL).transpose(dim0 = 2, dim1 = 3)
        grad_LHH = torch.matmul(matrix_Low_2, grad_HH).transpose(dim0 = 2, dim1 = 3)
        grad_HLL = torch.matmul(matrix_High_2, grad_LL).transpose(dim0 = 2, dim1 = 3)
        grad_HLH = torch.matmul(matrix_High_2, grad_LH).transpose(dim0 = 2, dim1 = 3)
        grad_HHL = torch.matmul(matrix_High_2, grad_HL).transpose(dim0 = 2, dim1 = 3)
        grad_HHH = torch.matmul(matrix_High_2, grad_HH).transpose(dim0 = 2, dim1 = 3)
        return grad_LLL, grad_LLH, grad_LHL, grad_LHH, grad_HLL, grad_HLH, grad_HHL, grad_HHH, None, None, None, None, None, None