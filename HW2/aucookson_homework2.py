# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 09:56:18 2020
ECE211A Homework 2
@author: aidan
"""

from PIL import Image, ImageDraw, ImageFont
from sklearn.linear_model import orthogonal_mp
#import skimage.metrics as metrics
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

# Turn a List of Patches into an Image
def recombine_patches(patches):
    new_img = np.zeros((IMG_SIZE, IMG_SIZE))
    patch_size = patches.shape[1]
    num_patches = IMG_SIZE // patch_size
    i = 0  
    for y in range(num_patches):
        for x in range(num_patches):
            new_img[y*patch_size:(y+1)*patch_size, x*patch_size:(x+1)*patch_size] = patches[i]
            i += 1
    return new_img

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


# Draw Text onto an Image
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
def random_dict(num_atoms=512, patch_size=8):
    d = np.random.random(size=(patch_size*patch_size, num_atoms)).astype(np.float32) 
    d = d / np.linalg.norm(d, axis=0, keepdims=True)
    return d

def K_SVD(y_train, tol=100, num_iter=100, S=20):
    patch_size = 8
    num_patches = (IMG_SIZE // patch_size)**2
    y_patches_train = np.asarray(extract_patches(y_train)).reshape((num_patches, patch_size**2))
    A_ksvd = random_dict()
    num_atoms = A_ksvd.shape[1]
    y_patches = []
    
    # FlattenPatches
    for patch in y_patches_train:
         y_patches.append(patch.flatten())
    
    for i in range(num_iter):
        print ('Iteration {}'.format(i+1))
        
        # Calculate x
        x = np.zeros((num_atoms, num_patches))
        for i, patch in enumerate(y_patches):
            x[:, i] = orthogonal_mp(A_ksvd, patch, n_nonzero_coefs=S)
            
        # Residual Threshold Termination
        error = residual_error(y_patches_train, A_ksvd, x)
        print ('\t Residual Error: {}'.format(error))
        if error <= tol: 
            return (A_ksvd, True)
        
        # Update Dictionary atom-by-atom
        for atom in range(num_atoms):
            wk = np.nonzero(x[atom, :])[0]
            
            if np.sum(wk) == 0:
                continue

            A_ksvd[:, atom] = 0   
            
            Ek = y_patches_train.copy().T
            #for j in range(num_atoms):  
               # aj = np.expand_dims(A_ksvd[:, j], axis=1) #64, 1
                #x_tj = np.expand_dims(x[j, :], axis=0) #1, 1024
                #Ek = Ek - aj.dot(x_tj)
            Ek = Ek - A_ksvd.dot(x)

            Ek_ohm = Ek[:, wk]
            
            # Update A, x with SVD
            U, D, V = np.linalg.svd(Ek_ohm, full_matrices=True)
            A_ksvd[:, atom] = U[:, 0]
            x[atom, wk] = (V[:, 0] * D[0]).T
            
    return (A_ksvd, False)
    
# Calculate Residual = ||Y - Ax||    
def residual_error(y_patches, A, x):
    error = 0.0
    for i, patch in enumerate(y_patches):
        estimate = A.dot(x[:, i])
        error += np.linalg.norm(patch - estimate)
    return error
                
# Recover Image Patches Using OMP
def recover_OMP(A, y_patches, m_patches, S=20):
    patches_recovered = np.zeros((1024, 8, 8))
    
    for i in range(len(y_patches)):
        patches_recovered[i] = A.dot(_OMP_Patch(A, y_patches[i], m_patches[i])).reshape((8, 8))
    return recombine_patches(patches_recovered)
        
# Orthogonal Matching Pursuit
def _OMP_Patch(A, y_patch, m_patch, S=20):
    y_patch = y_patch.flatten()
    m_patch = m_patch.flatten()
    y_patch_masked = y_patch[np.where(m_patch != 0)]
    A_masked = A[np.where(m_patch != 0)]
    coef = orthogonal_mp(A_masked, y_patch_masked, n_nonzero_coefs=S)
    return coef


if __name__=='__main__':
    #prepare_images()
    
    (y_train, y_test, y_test_missing, m_test_missing) = load_images()
    
    y_train = y_train / 255
    y_test = y_test / 255 
    y_test_missing = y_test_missing / 255
    
    #calc_metrics(y_test, y_test_missing)
    
    y_patches_missing = extract_patches(y_test_missing)
    y_patches_test = extract_patches(y_test)
    m_patches = extract_patches(m_test_missing)
    A_ksvd, converged = K_SVD(y_train, tol = 10, num_iter=16)
    img_recovered = recover_OMP(A_ksvd, y_patches_missing, m_patches)
    #save_patches(np.asarray(y_patches), np.asarray(m_patches))

    #img_recovered = recover_OMP(random_dict(), y_patches_missing, m_patches)
    plt.imshow(img_recovered, cmap='gray')
    plt.show()
    #cv2.imwrite('omp_result.png',img_recovered)

    
    
