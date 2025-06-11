### HOG & Harris Corner Detection (MATLAB) — SCC366 Media Coding & Processing · CW 2

A compact MATLAB project that re-implements two classic feature-extraction algorithms **from scratch** and analyses their robustness.&#x20;

**Contents**

| File / folder      | Purpose                                                                                                                                                    |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`Task1.m`**      | Implements `extractHOGFeatures()` → Histogram of Oriented Gradients (8 × 8 cells, 9 bins, 2 × 2 L2-normalised blocks with stride = 1, Sobel gradients).    |
| **`Task2.m`**      | Implements `harrisCornerDetector()` → parameterised Harris corner detection (Gaussian blur *w × w*, σ, *k*, threshold *p*), returns & overlays corner map. |
| **Sample images/** | Default grayscale test images used by the scripts.                                                                                                         |
| **`Report.pdf`**   | One-page analysis of Harris robustness to intensity shift & scaling, translation, and spatial scaling (Task 3).                                            |

**Quick start**

```matlab
% Histogram of Oriented Gradients
I = imread('Sample_images/cameraman.png');      % RGB handled inside the function
hog = extractHOGFeatures(I);
disp(length(hog));

% Harris Corner Detector
corners = harrisCornerDetector(I, 5, 1.2, 0.04, 0.01);   % w, sigma, k, p
imshow(corners); title('Detected corners');
```

**Highlights**

* Pure MATLAB — only basic functions (`imfilter`, `filter2`, `rgb2gray`, etc.); no toolbox detectors.
* Vectorised operations; loops avoided for efficiency.
* Clear inline comments and input validation (grayscale check, divisible-by-8 resizing, odd kernel size).
* Report summarises theoretical expectations **and** experimental results for each transformation scenario.

Clone, tweak the parameters, or drop in your own images to explore how HOG descriptors and Harris corners behave under real-world changes.
