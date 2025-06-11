
% Task2.m
% This script demonstrates how to call the harrisCornerDetector function,

% Load input image
inputImage = imread('pika.png');

% Parameters (you can adjust these as needed)
windowSize = 7;    % Gaussian filter window size
sigma = 1.5;       % Gaussian sigma
k = 0.04;          % Harris response constant (0.04 to 0.06)
threshold = 0.01;  % Threshold fraction of max(R)

% Call the Harris Corner Detector function
cornerMap = harrisCornerDetector(inputImage, windowSize, sigma, k, threshold);

% Display results
figure;
subplot(1,2,1); imshow(inputImage); title('Original Image');
subplot(1,2,2); imshow(inputImage); hold on;
[yCoords, xCoords] = find(cornerMap);
plot(xCoords, yCoords, 'ro', 'MarkerSize', 5, 'LineWidth', 1.5);
title('Detected Corners');
hold off;


% Function Definition: harrisCornerDetector

function cornerMap = harrisCornerDetector(inputImage, windowSize, sigma, k, threshold)
    
    % harrisCornerDetector
    %
    % This function implements the Harris Corner Detector algorithm.
    

    % 1) Ensure grayscale and convert to double
    inputImage = im2double(inputImage);
    if size(inputImage,3) == 3
        inputImage = rgb2gray(inputImage);
    end

    [h, w] = size(inputImage);

    % 2) Compute gradients using Sobel
    sobelX = [1 0 -1; 2 0 -2; 1 0 -1];
    sobelY = [1  2  1; 0  0  0; -1 -2 -1];

    Gx = imfilter(inputImage, sobelX, 'replicate');
    Gy = imfilter(inputImage, sobelY, 'replicate');

    Ix2 = Gx.^2;
    Iy2 = Gy.^2;
    Ixy = Gx.*Gy;

    % 3) Gaussian smoothing of the products
    halfSize = floor(windowSize/2);
    [X, Y] = meshgrid(-halfSize:halfSize, -halfSize:halfSize);
    gaussKernel = exp(-(X.^2 + Y.^2)/(2*sigma^2));
    gaussKernel = gaussKernel / sum(gaussKernel(:));

    Ix2_blur = imfilter(Ix2, gaussKernel, 'replicate');
    Iy2_blur = imfilter(Iy2, gaussKernel, 'replicate');
    Ixy_blur = imfilter(Ixy, gaussKernel, 'replicate');

    % 4) Harris response: R = det(M) - k*(trace(M))^2
    
    R = (Ix2_blur.*Iy2_blur - Ixy_blur.^2) - k*(Ix2_blur + Iy2_blur).^2;

    % 5) Thresholding
    Rmax = max(R(:));
    cornerCandidates = (R > threshold * Rmax);

    % 6) Non-maximum suppression in a 3x3 neighborhood
    se = ones(3,3);
    localMax = imdilate(R, se); % maximum in 3x3 neighborhood
    cornerMap = (R == localMax) & cornerCandidates;

    % Convert to logical
    cornerMap = logical(cornerMap);
end
