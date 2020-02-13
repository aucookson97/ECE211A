N = 100;
t = linspace(0,2*pi,50);
r = (N-10)/2;
circle = poly2mask(r*cos(t)+N/2+0.5, r*sin(t)+N/2+0.5,N,N);
img = repmat(circle,3,3);

x = linspace(-3,3,300);
y = x';
[X Y] = meshgrid(x,y);
Z = exp(-((X.^2)+(Y.^2))/(2*1/64));

C = conv2(img, Z);

Fimg = fft2(img, 599, 599);

FC = fft2(C);

R = FC ./ Fimg;

r = ifft2(R);