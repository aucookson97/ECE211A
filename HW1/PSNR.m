function psnr = PSNR(x1, x2, R)
%PSNR Peak Signal-to-Noise Ratio of Two Images
    mse = immse(x1, x2);
    psnr = 10*log10(R^2 / mse);
end

