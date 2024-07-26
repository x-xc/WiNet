# Copyright (c) 2019, Adobe Inc. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International Public License. To view a copy of this license, visit
# https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.
import numpy as np
import math, pywt
from torch.nn import Module
from DWT_IDWT.DWT_IDWT_Functions import *

# __all__ = ['DWT_1D', 'IDWT_1D', 'DWT_2D', 'IDWT_2D', 'DWT_3D', 'IDWT_3D', 'DWT_2D_tiny']

# When use dwt/idwt multiply times in a network for different image scales, 
# DWT_IDWT_layer2 is faster than DWT_IDWT_layer

class DWT_3D(Module):
    """
    input: (N, C, D, H, W)
    output: -- LLL (N, C, D/2, H/2, W/2)
            -- LLH (N, C, D/2, H/2, W/2)
            -- LHL (N, C, D/2, H/2, W/2)
            -- LHH (N, C, D/2, H/2, W/2)
            -- HLL (N, C, D/2, H/2, W/2)
            -- HLH (N, C, D/2, H/2, W/2)
            -- HHL (N, C, D/2, H/2, W/2)
            -- HHH (N, C, D/2, H/2, W/2)
    """
    def __init__(self, wavename):
        """
        :param band_low:  low-frequency filter of wavelet 
        :param band_high: high-frequency filter of wavelet 
        """
        super(DWT_3D, self).__init__()
        wavelet = pywt.Wavelet(wavename)
        self.band_low = wavelet.rec_lo
        self.band_high = wavelet.rec_hi
        assert len(self.band_low) == len(self.band_high)
        self.band_length = len(self.band_low)
        assert self.band_length % 2 == 0
        self.band_length_half = math.floor(self.band_length / 2)
        
        self.old_l = set()
        
        self.matrix_low_0 = []
        self.matrix_low_1 = []
        self.matrix_low_2 = []
        
        self.matrix_high_0 = []
        self.matrix_high_1 = []
        self.matrix_high_2 = []
        
    def get_matrix(self):
        """
        generate matrix
        :return:
        """
        if self.input_depth in self.old_l: return
        
        L1 = np.max((self.input_height, self.input_width))
        L = math.floor(L1 / 2)
        matrix_h = np.zeros( ( L,      L1 + self.band_length - 2 ),dtype=np.float32)
        matrix_g = np.zeros( ( L1 - L, L1 + self.band_length - 2 ),dtype=np.float32)
        end = None if self.band_length_half == 1 else (-self.band_length_half+1)

        index = 0
        for i in range(L):
            for j in range(self.band_length):
                matrix_h[i, index+j] = self.band_low[j]
            index += 2
        matrix_h_0 = matrix_h[0:(math.floor(self.input_height / 2)), 0:(self.input_height + self.band_length - 2)]
        matrix_h_1 = matrix_h[0:(math.floor(self.input_width / 2)),  0:(self.input_width + self.band_length - 2)]
        matrix_h_2 = matrix_h[0:(math.floor(self.input_depth / 2)),  0:(self.input_depth + self.band_length - 2)]

        index = 0
        for i in range(L1 - L):
            for j in range(self.band_length):
                matrix_g[i, index+j] = self.band_high[j]
            index += 2
        matrix_g_0 = matrix_g[0:(self.input_height - math.floor(self.input_height / 2)),0:(self.input_height + self.band_length - 2)]
        matrix_g_1 = matrix_g[0:(self.input_width - math.floor(self.input_width / 2)),0:(self.input_width + self.band_length - 2)]
        matrix_g_2 = matrix_g[0:(self.input_depth - math.floor(self.input_depth / 2)),0:(self.input_depth + self.band_length - 2)]

        matrix_h_0 = matrix_h_0[:,(self.band_length_half-1):end]
        matrix_h_1 = matrix_h_1[:,(self.band_length_half-1):end]
        matrix_h_1 = np.transpose(matrix_h_1)
        matrix_h_2 = matrix_h_2[:,(self.band_length_half-1):end]

        matrix_g_0 = matrix_g_0[:,(self.band_length_half-1):end]
        matrix_g_1 = matrix_g_1[:,(self.band_length_half-1):end]
        matrix_g_1 = np.transpose(matrix_g_1)
        matrix_g_2 = matrix_g_2[:,(self.band_length_half-1):end]
        
        if torch.cuda.is_available():
            self.matrix_low_0.append(torch.tensor(matrix_h_0).cuda())
            self.matrix_low_1.append(torch.tensor(matrix_h_1).cuda())
            self.matrix_low_2.append(torch.tensor(matrix_h_2).cuda())
            self.matrix_high_0.append(torch.tensor(matrix_g_0).cuda())
            self.matrix_high_1.append(torch.tensor(matrix_g_1).cuda())
            self.matrix_high_2.append(torch.tensor(matrix_g_2).cuda())
        else:
            self.matrix_low_0.append(torch.tensor(matrix_h_0))
            self.matrix_low_1.append(torch.tensor(matrix_h_1))
            self.matrix_low_2.append(torch.tensor(matrix_h_2))
            self.matrix_high_0.append(torch.tensor(matrix_g_0))
            self.matrix_high_1.append(torch.tensor(matrix_g_1))
            self.matrix_high_2.append(torch.tensor(matrix_g_2))

    def forward(self, input):
        assert len(input.size()) == 5
        self.input_depth  = input.size()[-3]
        self.input_height = input.size()[-2]
        self.input_width  = input.size()[-1]
        #assert self.input_height > self.band_length and self.input_width > self.band_length and self.input_depth > self.band_length
        self.get_matrix()
        self.old_l.add(self.input_depth)
        i = list(self.old_l).index(self.input_depth)
       
        return DWTFunction_3D.apply(input, self.matrix_low_0[i], self.matrix_low_1[i], self.matrix_low_2[i],
                                           self.matrix_high_0[i], self.matrix_high_1[i], self.matrix_high_2[i])

class IDWT_3D(Module):
    """
    input:  -- LLL (N, C, D/2, H/2, W/2)
            -- LLH (N, C, D/2, H/2, W/2)
            -- LHL (N, C, D/2, H/2, W/2)
            -- LHH (N, C, D/2, H/2, W/2)
            -- HLL (N, C, D/2, H/2, W/2)
            -- HLH (N, C, D/2, H/2, W/2)
            -- HHL (N, C, D/2, H/2, W/2)
            -- HHH (N, C, D/2, H/2, W/2)
    output: (N, C, D, H, W)
    """
    def __init__(self, wavename, n=0):
        """
        :param band_low:  low-frequency filter of wavelet 
        :param band_high: high-frequency filter of wavelet 
        """
        super(IDWT_3D, self).__init__()
        wavelet = pywt.Wavelet(wavename)
        self.band_low = wavelet.dec_lo
        self.band_high = wavelet.dec_hi
        self.band_low.reverse()
        self.band_high.reverse()
        assert len(self.band_low) == len(self.band_high)
        self.band_length = len(self.band_low)
        assert self.band_length % 2 == 0
        self.band_length_half = math.floor(self.band_length / 2)
        
        self.old_l = set()
        
        self.matrix_low_0 = []
        self.matrix_low_1 = []
        self.matrix_low_2 = []
        
        self.matrix_high_0 = []
        self.matrix_high_1 = []
        self.matrix_high_2 = []
        
    def get_matrix(self):
        """
        generate matrix
        :return:
        """
        if self.input_depth in self.old_l: return
        
        L1 = np.max((self.input_height, self.input_width))
        L = math.floor(L1 / 2)
        matrix_h = np.zeros( ( L,      L1 + self.band_length - 2 ), dtype='float32')
        matrix_g = np.zeros( ( L1 - L, L1 + self.band_length - 2 ), dtype='float32')
        end = None if self.band_length_half == 1 else (-self.band_length_half+1)

        index = 0
        for i in range(L):
            for j in range(self.band_length):
                matrix_h[i, index+j] = self.band_low[j]
            index += 2
        matrix_h_0 = matrix_h[0:(math.floor(self.input_height / 2)), 0:(self.input_height + self.band_length - 2)]
        matrix_h_1 = matrix_h[0:(math.floor(self.input_width  / 2)), 0:(self.input_width  + self.band_length - 2)]
        matrix_h_2 = matrix_h[0:(math.floor(self.input_depth  / 2)), 0:(self.input_depth  + self.band_length - 2)]

        index = 0
        for i in range(L1 - L):
            for j in range(self.band_length):
                matrix_g[i, index+j] = self.band_high[j]
            index += 2
        matrix_g_0 = matrix_g[0:(self.input_height - math.floor(self.input_height / 2)) ,0:(self.input_height + self.band_length - 2)]
        matrix_g_1 = matrix_g[0:(self.input_width  - math.floor(self.input_width  / 2)), 0:(self.input_width  + self.band_length - 2)]
        matrix_g_2 = matrix_g[0:(self.input_depth  - math.floor(self.input_depth  / 2)), 0:(self.input_depth  + self.band_length - 2)]

        matrix_h_0 = matrix_h_0[:,(self.band_length_half-1):end]
        matrix_h_1 = matrix_h_1[:,(self.band_length_half-1):end]
        matrix_h_1 = np.transpose(matrix_h_1)
        matrix_h_2 = matrix_h_2[:,(self.band_length_half-1):end]

        matrix_g_0 = matrix_g_0[:,(self.band_length_half-1):end]
        matrix_g_1 = matrix_g_1[:,(self.band_length_half-1):end]
        matrix_g_1 = np.transpose(matrix_g_1)
        matrix_g_2 = matrix_g_2[:,(self.band_length_half-1):end]
        
        if torch.cuda.is_available():
            self.matrix_low_0.append(torch.tensor(matrix_h_0).cuda())
            self.matrix_low_1.append(torch.tensor(matrix_h_1).cuda())
            self.matrix_low_2.append(torch.tensor(matrix_h_2).cuda())
            self.matrix_high_0.append(torch.tensor(matrix_g_0).cuda())
            self.matrix_high_1.append(torch.tensor(matrix_g_1).cuda())
            self.matrix_high_2.append(torch.tensor(matrix_g_2).cuda())
        else:
            self.matrix_low_0.append(torch.tensor(matrix_h_0))
            self.matrix_low_1.append(torch.tensor(matrix_h_1))
            self.matrix_low_2.append(torch.tensor(matrix_h_2))
            self.matrix_high_0.append(torch.tensor(matrix_g_0))
            self.matrix_high_1.append(torch.tensor(matrix_g_1))
            self.matrix_high_2.append(torch.tensor(matrix_g_2))

    def forward(self, LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH):
        assert len(LLL.size()) == len(LLH.size()) == len(LHL.size()) == len(LHH.size()) == 5
        assert len(HLL.size()) == len(HLH.size()) == len(HHL.size()) == len(HHH.size()) == 5
        self.input_depth  = LLL.size()[-3] + HHH.size()[-3]
        self.input_height = LLL.size()[-2] + HHH.size()[-2]
        self.input_width  = LLL.size()[-1] + HHH.size()[-1]
        #assert self.input_height > self.band_length and self.input_width > self.band_length and self.input_depth > self.band_length
        self.get_matrix()
        self.old_l.add(self.input_depth)
        i = list(self.old_l).index(self.input_depth)
        return IDWTFunction_3D.apply(LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH,
                                     self.matrix_low_0[i], self.matrix_low_1[i], self.matrix_low_2[i],
                                     self.matrix_high_0[i], self.matrix_high_1[i], self.matrix_high_2[i])

