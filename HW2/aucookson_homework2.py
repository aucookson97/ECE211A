# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 09:56:18 2020
ECE211A Homework 2
@author: aidan
"""

from PIL import Image, ImageDraw, ImageFont
from sklearn.linear_model import OrthogonalMatchingPursuit
import skimage.metrics as metrics
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import cv2
import random

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
        
    f_y.savefig('y_patches.png')
    
    f_m = plt.figure(figsize = (grid_size,grid_size))

    for i in range(grid_size**2):
        ax = plt.subplot(gs[i])
        ax.imshow(m_patches[indices[i]], cmap='gray')
        
        ax.set_xticklabels([])
        ax.set_yticklabels([])  
    
    f_m.savefig('m_patches.png')

# Draw Text onto an Image
def draw_text(img, invert=False):
    #font = ImageFont.truetype("sans-serif.ttf", 16)
    font = ImageFont.truetype('Adequate-ExtraLight.ttf', 30)
    d = ImageDraw.Draw(img)
    d.fontmode='1'
    
    fill_col = 255 if not invert else 0
    
    for i in range(0, 6):
        d.text((15, 10 + 40 * i), "ECE211A HW2", fill=fill_col, font=font)
    return img

# Create and Save Images to Files
def prepare_images():
    # Load two Images
    img1 = cv2.imread('img_3.png')
    img2 = cv2.imread('img_4.png')
    
    # Convert Images to Grayscale
    y_train = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    y_test = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('y_train_2.png', y_train)
    cv2.imwrite('y_test_2.png', y_test)
    
    # Draw Text to Test image
    y_test_missing = draw_text(Image.open('y_test_2.png'))
    
    y_test_missing.save('y_test_2_missing.png')
    
    y_test_missing = cv2.imread('y_test_2_missing.png')
    
    # Create Text Mask
    m_test_missing = draw_text(Image.new('1', (IMG_SIZE, IMG_SIZE), color=255), invert=True)
    
    m_test_missing.save('m_test_2_missing.png')    
    
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

# Learn Dictionary using K_SVD
def K_SVD(y_train, tol=100, num_iter=100, S=20):
    A_ksvd = random_dict() # 64 x 512
    num_atoms = A_ksvd.shape[1]
    
   # Flatten Patches to 64 x 1
    y_patches = []  
    for patch in y_train:
        y_patches.append(patch.flatten())        
    Y = np.asarray(y_patches).T # 64 x 1024
    
    for i in range(num_iter):
        print ('Iteration {}'.format(i+1))
        
        # Calculate coef
        omp = OrthogonalMatchingPursuit(S, fit_intercept=False)
        omp.fit(A_ksvd, Y)  
        coef = omp.coef_.T # 512 x 1024
        
        # Residual Threshold Termination
        error = np.linalg.norm(A_ksvd.dot(coef) - Y)
        print ('\t Residual Error: {}'.format(error))
        if error <= tol: 
            return (A_ksvd, True)
        
        # Update Dictionary atom-by-atom
        for atom in range(num_atoms):
            wk = coef[atom, :] != 0
            
            if np.sum(wk) == 0:
                continue

            # Ignore Row 'atom'
            A_ksvd[:, atom] = 0   
            
            Ek = Y - A_ksvd.dot(coef)
            Ek = Ek[:, wk]
            # Update A, x with SVD
            U, D, V = np.linalg.svd(Ek, full_matrices=False)
            A_ksvd[:, atom] = U[:, 0]
            coef[atom, wk] = (V.T[:, 0] * D[0]).T
            
    return (A_ksvd, False)
                
# Recover Image Patches Using OMP
def recover_OMP(A, y_patches, m_patches, S=20):
    patches_recovered = np.zeros((1024, 8, 8)) 
    omp = OrthogonalMatchingPursuit(S, fit_intercept=False)
    for i in range(len(y_patches)):
        patches_recovered[i] = A.dot(_OMP_Patch(A, y_patches[i], m_patches[i], omp)).reshape((8, 8))
        
    return recombine_patches(patches_recovered)
        
# Orthogonal Matching Pursuit
def _OMP_Patch(A, y_patch, m_patch, omp, S=20):  
    y_patch = y_patch.flatten()
    m_patch = m_patch.flatten().astype(bool)
    y_patch_masked = y_patch[m_patch]
    A_masked = A[m_patch]
    omp.fit(A_masked, y_patch_masked)
    return omp.coef_

# Returns Unit Norm Dictionary of Patches
def random_dict(num_atoms=512, patch_size=8):
    d = np.random.random(size=(patch_size*patch_size, num_atoms)).astype(np.float32) 
    for i in range(num_atoms):
        d[:, i] = d[:,i] / np.linalg.norm(d[:,i])
    return d

# Plot 25 Atoms with highest std
def sort_by_std(A):
    grid_size = 5
    std = np.std(A, axis=0)
    ind = np.argmin(std)
    f_a = plt.figure(figsize = (grid_size,grid_size))
    gs = gridspec.GridSpec(grid_size, grid_size)
    gs.update(wspace=0.025, hspace=0.05)

    for i in range(grid_size**2):
        ax = plt.subplot(gs[i])
        ax.imshow((A[:, ind[i]]).reshape((8, 8)), cmap='gray')
        
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
    f_a.savefig('A_ksvd_std.png')

if __name__=='__main__':
    #prepare_images()
    
    (y_train, y_test, y_test_missing, m_test_missing) = load_images()
    
    y_train = y_train / 255
    y_test = y_test / 255 
    y_test_missing = y_test_missing / 255
    
   # calc_metrics(y_test, y_test_missing)
    
    y_patches_missing = extract_patches(y_test_missing)
    y_patches_test = extract_patches(y_test)
    y_patches_train = extract_patches(y_train)
    m_patches = extract_patches(m_test_missing)
    A_ksvd, converged = K_SVD(y_patches_train, tol = 1e-3, num_iter=100)
    sort_by_std(A_ksvd)
    img_recovered = recover_OMP(A_ksvd, y_patches_missing, m_patches)
    #save_patches(np.asarray(y_patches_missing), np.asarray(m_patches))
    #img_recovered = recover_OMP(random_dict(), y_patches_missing, m_patches)
    #calc_metrics(y_test, )
    #calc_metrics(y_test, img_recovered)
   # cv2.imwrite('y_test_random.png', img_recovered)
    plt.figure(4)
    plt.imshow(img_recovered, cmap='gray')
    plt.show()
    #cv2.imwrite('omp_result.png',img_recovered)

    
    
