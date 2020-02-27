# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 09:56:18 2020
ECE211A Homework 2
@author: aidan
"""

from PIL import Image, ImageDraw, ImageFont
from sklearn.linear_model import orthogonal_mp
import skimage.metrics as metrics
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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

# Save a Sampled Grid of Dictionary Atoms
def save_patch(filename, patches, grid_size=5):
    indices = random.sample(range(patches.shape[0]), grid_size**2)
    
    f_a = plt.figure(figsize = (grid_size,grid_size))
    gs = gridspec.GridSpec(grid_size, grid_size)
    gs.update(wspace=0.025, hspace=0.05)
    
    #f_a, axarr_a = plt.subplots(grid_size, grid_size)


    for i in range(grid_size**2):
        ax = plt.subplot(gs[i])
        ax.imshow(patches[indices[i]], cmap='gray')
        
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    f_a.savefig(filename)

# Save a Sampled Grid of Patches and Masks
def save_patches(y_patches, m_patches, grid_size=5):
    indices = random.sample(range(y_patches.shape[0]), grid_size**2)
    
    f_y = plt.figure(figsize = (grid_size,grid_size))
    gs = gridspec.GridSpec(grid_size, grid_size)
    gs.update(wspace=0.025, hspace=0.05)

    for i in range(grid_size**2):
        ax = plt.subplot(gs[i])
        ax.imshow(y_patches[indices[i]], cmap='gray')
        
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
    f_y.savefig('y_patches_2.png')
    
    f_m = plt.figure(figsize = (grid_size,grid_size))

    for i in range(grid_size**2):
        ax = plt.subplot(gs[i])
        ax.imshow(m_patches[indices[i]], cmap='gray')
        
        ax.set_xticklabels([])
        ax.set_yticklabels([])  
    
    f_m.savefig('m_patches_2.png')


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

# Returns Unit Norm Dictionary of Patches
def random_dict(num_atoms=512, patch_size=8, value_range=256):
    d = np.random.randint(0, value_range, size=(num_atoms, patch_size, patch_size)).astype(np.float32) 
    for atom in d:
        atom /= np.linalg.norm(atom)
    return d

# Orthogonal Matching Pursuit
def OMP(A, patch, S=20):
    coef = orthogonal_mp(A, patch, n_nonzero_coefs=S)
    return coef

if __name__=='__main__':
    #prepare_images()
    
    (y_train, y_test, y_test_missing, m_test_missing) = load_images()
    
    #calc_metrics(y_test, y_test_missing)
    
    y_patches = extract_patches(y_test_missing)
    m_patches = extract_patches(m_test_missing)
    
    #save_patches(y_patches, m_patches)
    
    A_dict = random_dict()
    #save_patches(np.asarray(y_patches), np.asarray(m_patches))
    
    print (OMP(A_dict, y_patches[0]))
    
    