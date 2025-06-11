
% Task1.m
% This script demonstrates how to call the extractHOGFeatures function using

% Load the input image
inputImage = imread('pika.png');

% Call the HOG feature extraction function
extractedHOGFeatures = extractHOGFeatures(inputImage);

% Display the length of the extracted features
disp(['Length of extracted HOG features: ', num2str(length(extractedHOGFeatures))]);


% Function Definition: extractHOGFeatures

function extractedHOGFeatures = extractHOGFeatures(inputImage)

    %---------------------%
    % 1) Input Validation %
    %---------------------%
    inputImage = im2double(inputImage); % Convert to double for processing

    % If RGB, convert to grayscale
    if size(inputImage,3) == 3
        inputImage = rgb2gray(inputImage);
    end

    % Get dimensions
    [h, w] = size(inputImage);

    % Ensure dimensions are divisible by 8, if not, resize
    newH = h - mod(h,8);
    newW = w - mod(w,8);
    if newH == 0, newH = 8; end
    if newW == 0, newW = 8; end
    if (newH ~= h) || (newW ~= w)
        inputImage = imresize(inputImage, [newH newW]);
    end

    % Update dimensions after resizing
    [h, w] = size(inputImage);

    %------------------------------%
    % 2) Gradient Computation (Sobel)
    %------------------------------%

    % Define Sobel kernels
    sobelX = [1 0 -1; 2 0 -2; 1 0 -1]; % Gx filter
    sobelY = [1  2  1; 0  0  0; -1 -2 -1]; % Gy filter

    % Apply Sobel filters to compute gradients
    Gx = imfilter(inputImage, sobelX, 'replicate');
    Gy = imfilter(inputImage, sobelY, 'replicate');

    % Compute gradient magnitude and orientation
    magnitude = sqrt(Gx.^2 + Gy.^2);
    orientation = atan2(Gy, Gx); % range: [-pi, pi]

    % Convert orientation from radians to degrees [0,180)
    orientation = (orientation * (180/pi));
    orientation(orientation<0) = orientation(orientation<0) + 180;

    %-----------------------------------%
    % 3) Cell Histograms (8x8 cells, 9 bins)
    %-----------------------------------%
    cellSize = 8;        % Each cell is 8x8
    numBins = 9;         % 9 orientation bins
    binSize = 180 / numBins; % 20 degrees per bin

    % Number of cells along height and width
    numCellsY = h / cellSize;
    numCellsX = w / cellSize;

    % Pre-allocate cell histogram array: numCellsY x numCellsX x numBins
    cellHist = zeros(numCellsY, numCellsX, numBins);

    % Compute histograms for each cell
    for i = 1:numCellsY
        for j = 1:numCellsX
            % Extract the cell region
            rowStart = (i-1)*cellSize + 1;
            rowEnd   = i*cellSize;
            colStart = (j-1)*cellSize + 1;
            colEnd   = j*cellSize;

            cellOri = orientation(rowStart:rowEnd, colStart:colEnd);
            cellMag = magnitude(rowStart:rowEnd, colStart:colEnd);

            % Flatten cell arrays to vector form
            cellOri = cellOri(:);
            cellMag = cellMag(:);

            % Assign each pixel's gradient to a histogram bin
            binIdx = floor(cellOri / binSize) + 1;
            binIdx(binIdx > numBins) = numBins; % Prevent out-of-range index

            % Accumulate magnitudes into histogram bins using accumarray
            histVals = accumarray(binIdx, cellMag, [numBins 1], @sum, 0);

            cellHist(i,j,:) = histVals;
        end
    end

    %---------------------------------------%
    % 4) Block Normalization (2x2 cells per block)
    %---------------------------------------%
    % Each block: 2x2 cells, each cell has 9 bins, total 36 features per block.
    
    blockSize = 2; 
    blockHistSize = blockSize * blockSize * numBins; % 2x2x9 = 36

    % Number of blocks in each dimension
    numBlocksY = numCellsY - 1;
    numBlocksX = numCellsX - 1;

    % Pre-allocate a vector for all HOG features
    hogFeatures = [];

    % Loop over each block position
    for by = 1:numBlocksY
        for bx = 1:numBlocksX
            % Extract the 2x2 block of cell histograms
            blockHists = [...
                cellHist(by,   bx,   :);
                cellHist(by,   bx+1, :);
                cellHist(by+1, bx,   :);
                cellHist(by+1, bx+1,:)];

            % Convert to a vector
            blockHists = blockHists(:);

            % L2 Normalization
            epsVal = 1e-5; % small constant to avoid division by zero
            normVal = sqrt(sum(blockHists.^2) + epsVal^2);
            blockHists = blockHists / normVal;

            % Concatenate to global HOG descriptor
            hogFeatures = [hogFeatures; blockHists];
        end
    end

    %-----------------------------------%
    % 5) Output the final HOG descriptor
    %-----------------------------------%
    extractedHOGFeatures = hogFeatures;
end
