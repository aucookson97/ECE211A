# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 09:56:18 2020
ECE211A Homework 2
@author: aidan
"""

from PIL import Image, ImageDraw, ImageFont
import skimage.metrics as metrics
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import random

FONT_PATH = 'C:\\Windows\\Fonts'
IMG_SIZE = 256

# Turn an Image Into a List of Patches
def extract_patches(img, patch_size=8):
    
    patches = []
    num_patches = IMG_SIZE // patch_size
    
    for y in range(num_patches):
        for x in range(num_patches):
            patch = img[y*patch_size:(y+1)*patch_size, x*patch_size:(x+1)*patch_size]
            patches.append(patch)
    return patches

def save_patches(y_patches, m_patches, grid_size=5):
    indices = random.sample(range(len(y_patches)), grid_size**2)
    
    #  %matplotlib qt
    f_y, axarr_y = plt.subplots(grid_size, grid_size)
    f_m, axarr_m = plt.subplots(grid_size, grid_size)

    
    i = 0
    for y in range(grid_size):
        for x in range(grid_size):
            axarr_y[y, x].imshow(y_patches[indices[i]], cmap='gray')
            axarr_m[y, x].imshow(m_patches[indices[i]], cmap='gray')
            i += 1
            
            axarr_y[y, x].set_xticklabels([])
            axarr_y[y, x].set_yticklabels([])
            axarr_m[y, x].set_xticklabels([])
            axarr_m[y, x].set_yticklabels([])
            
    f_y.savefig('y_patches.png')
    f_m.savefig('m_patches.png')
            
    # patch_size = y_patches[0].shape[0]
    # img_y = np.zeros((patch_size*grid_size, patch_size*grid_size))
    # img_m = np.zeros((patch_size*grid_size, patch_size*grid_size))
    # indices = random.sample(range(len(y_patches)), grid_size**2)
    
    # i = 0
    # for y in range(grid_size):
    #     for x in range(grid_size):
    #         img_y[y*patch_size:(y+1)*patch_size, x*patch_size:(x+1)*patch_size] = y_patches[indices[i]]
    #         img_m[y*patch_size:(y+1)*patch_size, x*patch_size:(x+1)*patch_size] = m_patches[indices[i]]
    #         i += 1
            
    #cv2.imwrite('y_patches.png', img_y)
    #cv2.imwrite('m_patches.png', img_m)


def draw_text(img, invert=False):
    #font = ImageFont.truetype("sans-serif.ttf", 16)
    font = ImageFont.truetype('Adequate-ExtraLight.ttf', 30)
    d = ImageDraw.Draw(img)
    
    fill_col = 255 if not invert else 0
    
    for i in range(0, 6):
        d.text((15, 10 + 40 * i), "ECE211A HW2", fill=fill_col, font=font)
    return img

# Create and Save Images to Files
def prepare_images():
    # Load two Images
    img1 = cv2.imread('img_1.jpg')
    img2 = cv2.imread('img_2.jpg')
    
    # Convert Images to Grayscale
    y_train = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    y_test = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('y_train.png', y_train)
    cv2.imwrite('y_test.png', y_test)
    
    # Draw Text to Test image
    y_test_missing = draw_text(Image.open('y_test.png'))
    
    y_test_missing.save('y_test_missing.png')
    
    y_test_missing = cv2.imread('y_test_missing.png')
    #cv2.imshow('y_test_missing', y_test_missing)
   # cv2.waitKey(0)
    
    # Create Text Mask
    m_test_missing = draw_text(Image.new('1', (IMG_SIZE, IMG_SIZE), color=255), invert=True)
    
    m_test_missing.save('m_test_missing.png')    
    #m_test_missing = cv2.imread('m_test_missing.png')
    #cv2.imshow('m_test_missing', m_test_missing)
    #cv2.waitKey(0)
    
def calc_metrics(img1, img2):
   print ('PSNR: {}'.format(metrics.peak_signal_noise_ratio(img1, img2)))
   print ('SSIM: {}'.format(metrics.structural_similarity(img1, img2)))
    
# Load Images from Files
def load_images():
    y_train = cv2.imread('y_train.png', 0)
    y_test = cv2.imread('y_test.png', 0)
    y_test_missing = cv2.imread('y_test_missing.png', 0)
       
    m_test_missing = cv2.imread('m_test_missing.png', 0)
    return (y_train, y_test, y_test_missing, m_test_missing)

if __name__=='__main__':
    prepare_images()
    
    (y_train, y_test, y_test_missing, m_test_missing) = load_images()
    
    calc_metrics(y_test, y_test_missing)
    
    y_patches = extract_patches(y_test_missing)
    m_patches = extract_patches(m_test_missing)
    
    save_patches(y_patches, m_patches)
    
    
    