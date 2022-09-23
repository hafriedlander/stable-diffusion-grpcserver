"""
MIT License

Copyright (c) 2022 Christopher Friesen
https://github.com/parlance-zz

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


g_diffuser_lib.py - shared functions and diffusers operations

"""

import time
import datetime
import argparse
import uuid
import pathlib
import json

import numpy as np
import PIL
from PIL import Image
from PIL import ImageFilter
from PIL import ImageEnhance
import skimage
from skimage.exposure import match_histograms
from skimage import color
from skimage import transform

import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
from diffusers import StableDiffusionInpaintPipeline # we don't need the img2img pipeline because inpaint is a superset of its functionality
#from diffusers import LMSDiscreteScheduler          # broken at the moment I believe

def save_debug_img(np_image, name):
    image_path = "_debug_" + name + ".png"
    if type(np_image) == np.ndarray:
        if np_image.ndim == 2:
            mode = "L"
        elif np_image.shape[2] == 4:
            mode = "RGBA"
        else:
            mode = "RGB"
        pil_image = PIL.Image.fromarray(np.clip(np.absolute(np_image)*255., 0., 255.).astype(np.uint8), mode=mode)
        pil_image.save(image_path)
    else:
        np_image.save(image_path)
    return image_path

# ************* in/out-painting code begins here *************

# helper fft routines that keep ortho normalization and auto-shift before and after fft, and can handle multi-channel images

def fft2(data):
    if data.ndim > 2: # multiple channels
        out_fft = np.zeros((data.shape[0], data.shape[1], data.shape[2]), dtype=np.complex128)
        for c in range(data.shape[2]):
            c_data = data[:,:,c]
            out_fft[:,:,c] = np.fft.fft2(np.fft.fftshift(c_data),norm="ortho")
            out_fft[:,:,c] = np.fft.ifftshift(out_fft[:,:,c])
    else: # single channel
        out_fft = np.zeros((data.shape[0], data.shape[1]), dtype=np.complex128)
        out_fft[:,:] = np.fft.fft2(np.fft.fftshift(data),norm="ortho")
        out_fft[:,:] = np.fft.ifftshift(out_fft[:,:])
    
    return out_fft
   
def ifft2(data):
    if data.ndim > 2: # multiple channels
        out_ifft = np.zeros((data.shape[0], data.shape[1], data.shape[2]), dtype=np.complex128)
        for c in range(data.shape[2]):
            c_data = data[:,:,c]
            out_ifft[:,:,c] = np.fft.ifft2(np.fft.fftshift(c_data),norm="ortho")
            out_ifft[:,:,c] = np.fft.ifftshift(out_ifft[:,:,c])
    else: # single channel
        out_ifft = np.zeros((data.shape[0], data.shape[1]), dtype=np.complex128)
        out_ifft[:,:] = np.fft.ifft2(np.fft.fftshift(data),norm="ortho")
        out_ifft[:,:] = np.fft.ifftshift(out_ifft[:,:])
        
    return out_ifft
            
def get_gaussian(width, height, std=3.14, edge_filter=False): # simple gaussian kernel

    window_scale_x = float(width / min(width, height))  # for non-square aspect ratios we still want a circular gaussian
    window_scale_y = float(height / min(width, height)) 
    window = np.zeros((width, height))
    
    x = (np.arange(width) / width * 2. - 1.) * window_scale_x
    kx = np.exp(-x*x * std)
    if window_scale_x != window_scale_y:
        y = (np.arange(height) / height * 2. - 1.) * window_scale_y
        ky = np.exp(-y*y * std)
    else:
        y = x
        ky = kx
    gaussian = np.outer(kx, ky)
    
    if edge_filter:
        return gaussian * (1. -std*np.add.outer(x*x,y*y)) # normalized gaussian 2nd derivative
    else:
        return gaussian

def convolve(data1, data2):      # fast convolution with fft
    if data1.ndim != data2.ndim: # promote to rgb if mismatch
        if data1.ndim < 3: data1 = np_img_grey_to_rgb(data1)
        if data2.ndim < 3: data2 = np_img_grey_to_rgb(data2)
    return ifft2(fft2(data1) * fft2(data2))

def gaussian_blur(data, std=3.14):
    width = data.shape[0]
    height = data.shape[1]
    kernel = get_gaussian(width, height, std)
    return np.real(convolve(data, kernel / np.sqrt(np.sum(kernel*kernel))))
 
def normalize_image(data):
    normalized = data - np.min(data)
    normalized_max = np.max(normalized)
    assert(normalized_max > 0.)
    return normalized / normalized_max
 
def np_img_rgb_to_grey(data):
    if data.ndim == 2: return data
    return np.sum(data, axis=2)/3.
    
def np_img_grey_to_rgb(data):
    if data.ndim == 3: return data
    return np.expand_dims(data, 2) * np.ones((1, 1, 3))

def hsv_blend_image(image, match_to, hsv_mask=None):
    width = image.shape[0]
    height = image.shape[1]
    if type(hsv_mask) != np.ndarray:
        hsv_mask = np.ones((width, height, 3))
        
    image_hsv = color.rgb2hsv(image)
    match_to_hsv = color.rgb2hsv(match_to)
    return color.hsv2rgb(image_hsv * (1.-hsv_mask) + hsv_mask * match_to_hsv)
    
# prepare masks for in/out-painting
def get_blend_mask(np_mask_rgb, args):  # np_mask_rgb is an np array of rgb data in (0..1)
                                                                 # mask_blend_factor ( > 0.) adjusts blend hardness, with 1. corresponding closely to the original mask and higher values approaching the hard edge of the original mask
                                                                 # strength overrides (if > 0.) the maximum opacity in the user mask to support style transfer type applications
    assert(np_mask_rgb.ndim == 3) # needs to be a 3 channel mask
    width = np_mask_rgb.shape[0]
    height = np_mask_rgb.shape[1]
    
    if args.debug: save_debug_img(np_mask_rgb, "np_mask_rgb")
    if args.strength == 0.:
        max_opacity = np.max(np_mask_rgb)
    else:
        max_opacity = np.clip(args.strength, 0., 1.)
    
    final_blend_mask = 1. - (1.-normalize_image(gaussian_blur(1.-np_mask_rgb, std=1000.))) * max_opacity
    if args.debug: save_debug_img(final_blend_mask, "final_blend_mask")
    return final_blend_mask

"""

 Why does this need to exist? I thought SD already did in/out-painting?:
 
 This seems to be a common misconception. Non-latent diffusion models such as Dall-e can be readily used for in/out-painting
 but the current SD in-painting pipeline is just regular img2img with a mask, and changing that would require training a
 completely new model (at least to my understanding). In order to get good results, SD needs to have information in the
 (completely) erased area of the image. Adding to the confusion is that the PNG file format is capable of saving color data in
 (completely) erased areas of the image but most applications won't do this by default, and copying the image data to the "clipboard"
 will erase the color data in the erased regions (at least in Windows). Code like this or patchmatch that can generate a
 seed image (or "fixed code") will (at least for now) be required for seamless out-painting.
 
 Although there are simple effective solutions for in-painting, out-painting can be especially challenging because there is no color data
 in the masked area to help prompt the generator. Ideally, even for in-painting we'd like work effectively without that data as well.

 By taking a fourier transform of the unmasked source image we get a function that tells us the presence, orientation, and scale of features
 in that source. Shaping the init/seed/fixed code noise to the same distribution of feature scales, orientations, and positions/phases
 increases (visual) output coherence by helping keep features aligned and of similar orientation and size. This technique is applicable to any continuous
 generation task such as audio or video, each of which can be conceptualized as a series of out-painting steps where the last half of the input "frame" is erased.
 TLDR: The fourier transform of the unmasked source image is a strong prior for shaping the noise distribution of in/out-painted areas
 
 For multi-channel data such as color or stereo sound the "color tone" of the noise can be bled into the noise with gaussian convolution and
 a final histogram match to the unmasked source image ensures the palette of the source is mostly preserved. SD is extremely sensitive to
 careful color and "texture" matching to ensure features are appropriately "bound" if they neighbor each other in the transition zone.
 
 The effects of both of these techiques in combination include helping the generator integrate the pre-existing view distance and camera angle,
 as well as being more likely to complete partially erased features (like appropriately completing a partially erased arm, house, or tree).
 
 Please note this implementation is written for clarity and correctness rather than performance.
 
 Todo: To be investigated is the idea of using the same technique directly in latent space. Spatial properties are (at least roughly?) preserved
 in latent space so the fourier transform should be usable there for the same reason convolutions are usable there. The ideas presented here
 could also be combined or augmented with other existing techniques.
 Todo: It would be trivial to add brightness, contrast, and overall palette control using simple parameters
 Todo: There are some simple optimizations that can increase speed significantly, e.g. re-using FFTs and gaussian kernels
 
 Parameters:
 
 - np_init should be an np array of the RGB source image in range (0..1)
 - noise_q modulates the fall-off of the target distribution and can be any positive number, lower means higher detail in the in/out-painted area (range > 0, default 1., good values are usually near 1.5)
 - mask_hardened, final_blend_mask and window_mask are pre-prepared masks for which another function is provided above. Quality is highly sensitive
   to the construction of these masks.

 Dependencies: numpy, scikit-image

 This code is provided under the MIT license -  Copyright (c) 2022 Christopher Friesen
 To anyone who reads this I am seeking employment in related areas.
 Donations would also be greatly appreciated and will be used to fund further development. (ETH @ 0x8e4BbD53bfF9C0765eE1859C590A5334722F2086)
 
 Questions or comments can be sent to parlance@fifth-harmonic.com (https://github.com/parlance-zz/)
 
"""


def get_matched_noise(np_init, final_blend_mask, args): 

    width = np_init.shape[0]
    height = np_init.shape[1]
    num_channels = np_init.shape[2]
    
    # todo: experiment with transforming everything to HSV space FIRST
    windowed_image = np_init * (1.-final_blend_mask)
    if args.debug: save_debug_img(windowed_image, "windowed_src_img")
    
    assert(args.noise_q > 0.)
    noise_rgb = np.exp(-1j*2*np.pi * np.random.random_sample((width, height))) * 25. # todo: instead of 25 match with stats
    noise_rgb *= np.random.random_sample((width, height)) ** (50. * args.noise_q) # todo: instead of 50 match with stats
    noise_rgb = np.real(noise_rgb)
    colorfulness = 0. # todo: we also VERY BADLY need to control contrast and BRIGHTNESS
    noise_rgb = ((noise_rgb+0.5)*colorfulness + np_img_rgb_to_grey(noise_rgb+0.5)*(1.-colorfulness))-0.5
    
    schrodinger_kernel = get_gaussian(width, height, std=1j*2345234) * noise_rgb # todo: find a good magic number
    shaped_noise_rgb = np.absolute(convolve(schrodinger_kernel, windowed_image))
    if args.debug: save_debug_img(shaped_noise_rgb, "shaped_noise_rgb")
    
    offset = 0.1 # 1e-17 # 0.0125 # todo: create mask offset function that can set a lower offset
    hsv_blend_mask = (1. - final_blend_mask) * np.clip(final_blend_mask-1e-20, 0., 1.)**offset
    hsv_blend_mask = normalize_image(hsv_blend_mask)
    
    #max_opacity = np.max(hsv_blend_mask)
    hsv_blend_mask = np.minimum(normalize_image(gaussian_blur(hsv_blend_mask, std=4000.)) + 1e-8, 1.)
    offset_hsv_blend_mask = np.maximum(np.absolute(np.log(hsv_blend_mask)) ** (1/2), 0.)
    offset_hsv_blend_mask -= np.min(offset_hsv_blend_mask)
    hardness = 12500 # 7.5 # 1e-8 # 0.3 
    hsv_blend_mask = normalize_image(np.exp(-hardness * offset_hsv_blend_mask**2))
    #hsv_blend_mask[:,:,0] *= 1. # todo: experiment with this again
    #hsv_blend_mask[:,:,1] *= 0.05
    #hsv_blend_mask[:,:,2] *= 0.618
    #hsv_blend_mask *= 0.95
    if args.debug: save_debug_img(hsv_blend_mask, "hsv_blend_mask")
    
    shaped_noise_rgb = hsv_blend_image(shaped_noise_rgb, np_init, hsv_blend_mask)
    if args.debug: save_debug_img(shaped_noise_rgb, "shaped_noise_post_hsv_blend")
    
    all_mask = np.ones((width, height), dtype=bool)
    ref_mask = normalize_image(np_img_rgb_to_grey(1.-final_blend_mask))
    img_mask = ref_mask <= 0.99
    ref_mask = ref_mask > 0.1
    if args.debug:
        save_debug_img(ref_mask.astype(np.float64), "histo_ref_mask")
        save_debug_img(img_mask.astype(np.float64), "histo_img_mask")
    
    # todo: experiment with these again
    """
    matched_noise_rgb = shaped_noise_rgb.copy()
    #matched_noise_rgb[img_mask,:] = skimage.exposure.match_histograms(
    matched_noise_rgb[all_mask,:] = skimage.exposure.match_histograms(
        #shaped_noise_rgb[img_mask,:], 
        shaped_noise_rgb[all_mask,:],
        np_init[ref_mask,:],
        channel_axis=1
    )
    #shaped_noise_rgb = shaped_noise_rgb*(1.-0.618) + matched_noise_rgb*0.618
    shaped_noise_rgb = shaped_noise_rgb*0.25 + matched_noise_rgb*0.75
    save_debug_img(shaped_noise_rgb, "shaped_noise_rgb_post-histo-match")
    """
    
    """
    #shaped_noise_rgb[img_mask,:] = skimage.exposure.match_histograms(
    shaped_noise_rgb[all_mask,:] = skimage.exposure.match_histograms(
        shaped_noise_rgb[all_mask,:], 
        #shaped_noise_rgb[img_mask,:], 
        np_init[ref_mask,:]**.25,
        channel_axis=1
    )
    """
    
    shaped_noise_rgb = np_init * (1.-final_blend_mask) + shaped_noise_rgb * final_blend_mask
    save_debug_img(shaped_noise_rgb, "shaped_noise_rgb_post-final-blend")
    
    return np.clip(shaped_noise_rgb, 0., 1.) 
    
# ************* in/out-painting code ends here *************
