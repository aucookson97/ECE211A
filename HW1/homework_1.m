% Prepare Ground Truth Image as 'x'
img_raw = imread('208001.jpg'); % Load Raw Image from Dataset
rect = [0 145 256 255];
img_cropped = imcrop(img_raw, rect); % Crop Image to 256 x 256 x 3
img_grey = rgb2gray(img_cropped); % Convert Image to GreyScale
x = im2double(img_grey);

% 2D Convolution on x using Normalized Identity Kernel 'h'
kernel_size = 21;
h = eye(21) ./ kernel_size; % Normalized Identity Matrix
y_noiseless = conv2(x, h); % Convolve input with kernel (x ** h), zero-padding edges (size 276 x 276)

% Add Gaussian Noise to Noiseless y
var = (.01 * std2(x))^2; % Desired Gaussian Variance
y_noisy = imnoise(y_noiseless, 'gaussian', 0, var);

% Calculate PSNR and SSIM of Noiseless and Noisy Y
orig_size = size(y_noiseless, 1);
rect_crop_conv = [(orig_size-256)/2 (orig_size-256)/2 255 255]; % Take Center 256 x 256 of y images
y_noiseless_crop = imcrop(y_noiseless, rect_crop_conv);
y_noisy_crop = imcrop(y_noisy, rect_crop_conv);

R = 255; % Image Data Range

psnr_x_ynoiseless = PSNR(x, y_noiseless_crop, R);
psnr_x_ynoisy = PSNR(x, y_noisy_crop, R);
[ssim_x_ynoiseless, ~] = ssim(y_noiseless_crop, x);
[ssim_x_ynoisy, ~] = ssim(y_noisy_crop, x);

% Naive Deconvolution of Noiseless and Noisy Y
H = fft2(h, 276, 276) + 1e-12;
Y_noisy = fft2(y_noisy);
Y_noiseless = fft2(y_noiseless);

rect_crop = [0 0 256 256];
x_hat_noisy_naive = imcrop(ifft2(Y_noisy ./ H), rect_crop); % F-1(Y / H)
x_hat_noiseless_naive = imcrop(ifft2(Y_noiseless ./ H), rect_crop);


% Calculate PSNR and SSIM of X_Hat Naive Method
psnr_x_noiseless_naive = PSNR(x, x_hat_noiseless_naive, R);
psnr_x_noisy_naive = PSNR(x, x_hat_noisy_naive, R);
[ssim_x_noiseless_naive, ~] = ssim(x_hat_noiseless_naive, x);
[ssim_x_noisy_naive, ~] = ssim(x_hat_noisy_naive, x);

% Wiener Deconvolution of Noiseles and Noisy Y
X = fft2(x, 276, 276);
psd_x = X .* conj(X);
psd_h = H .* conj(H);
x_hat_noiseless_wiener = ifft2((psd_h ./ (psd_h + 1 ./ psd_x)) .* (Y_noiseless ./ H));
x_hat_noiseless_wiener = imcrop(x_hat_noiseless_wiener, rect_crop);
x_hat_noisy_wiener = ifft2((psd_h ./ (psd_h + 1 ./ psd_x)) .* (Y_noisy ./ H));
x_hat_noisy_wiener = imcrop(x_hat_noisy_wiener, rect_crop);

% Calculate PSNR and SSIM of X_Hat Wiener Exact
psnr_x_noisy_wiener = PSNR(x, x_hat_noisy_wiener, R);
[ssim_x_noisy_wiener, ~] = ssim(x_hat_noisy_wiener, x);

% Power Spectral Density Estimation
X = fftshift(fft2(x, 276, 276));
noise = y_noisy - y_noiseless;
N = fftshift(fft2(noise, 276, 276));
psd_x = log10(X .* conj(X));
psd_n = log10(N .* conj(N));

% Prepare Images
img_2 = imread('69015.jpg');
img_3 = imread('295087.jpg');

rect_2 = [0 0 256 256];
img_cropped_2 = imcrop(img_2, rect_2); % Crop Image to 256 x 256 x 3
img_grey_2 = rgb2gray(img_cropped_2); % Convert Image to GreyScale
x_2 = im2double(img_grey_2);

rect_3 = [100 50 255 255];
img_cropped_3 = imcrop(img_3, rect_3); % Crop Image to 256 x 256 x 3
img_grey_3 = rgb2gray(img_cropped_3); % Convert Image to GreyScale
x_3 = im2double(img_grey_3);

X_2 = fftshift(fft2(x_2));
X_3 = fftshift(fft2(x_3));

psd_x_2 = log10(X_2 .* conj(X_2));
psd_x_3 = log10(X_3 .* conj(X_3));

% Estimate PSD Proportional to 1/w^2
[X, Y] = meshgrid(linspace(-137, 138, 276));

%psd_est = (log10(1 ./ (X.^2))) .* (log10(1 ./ (Y.^2)));
psd_est = (6 + log10(1 ./ (X.^2) + 1 ./ (Y.^2)));
psd_est(isinf(psd_est)) = 6;

x_hat_wiener_est = ifft2((psd_h ./ (psd_h + 1 ./ psd_est)) .* (Y_noisy ./ H));
x_hat_wiener_est = imcrop(x_hat_wiener_est, rect_crop);

imshow(x_hat_wiener_est);
%colormap(contrast(psd_x));





