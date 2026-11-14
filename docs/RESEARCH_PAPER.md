# A Comparative Study of Basic CNN and DnCNN for Image Denoising under Multiple Noise Types

**Bandi Navadeep, Rajan Nayak, Rohit**  
[Department], [College/University], [City, Country]

## Abstract
This paper presents an implementation-level comparison of a compact convolutional neural network (Basic CNN) and a 17-layer denoising convolutional neural network (DnCNN) for image denoising. The models are evaluated under Gaussian, salt-and-pepper, and speckle noise at three severity levels. The evaluation uses 100 DIV2K validation images per condition, for 1,800 model predictions across 18 architecture-condition combinations. Results are summarized using peak signal-to-noise ratio (PSNR) and structural similarity (SSIM). Averaged over the nine noise/severity conditions, DnCNN achieves 34.2023 dB PSNR and 0.9184 SSIM, while Basic CNN achieves 23.6174 dB and 0.7740 SSIM. DnCNN scores higher on both metrics in every evaluated condition. Since the models were trained with different training budgets and data-generation procedures, the findings compare the implemented trained systems rather than providing a strictly controlled architecture-only comparison.

**Keywords:** image denoising, convolutional neural network, DnCNN, residual learning, Gaussian noise, salt-and-pepper noise, speckle noise, PSNR, SSIM

## 1. Introduction
Noise introduced during image acquisition, transmission, or processing can degrade visual quality. Denoising seeks to recover an estimate of the original image from a noisy observation. Convolutional neural networks learn spatial filters and nonlinear mappings from training examples. DnCNN uses residual learning to estimate noise directly, offering an alternative to predicting the clean image.

This work implements two approaches and compares them in a common evaluation pipeline and a Streamlit demonstration application. The objective is practical evaluation across three synthetic noise families and three severity levels.

## 2. Related work
DnCNN, introduced by Zhang et al., uses residual learning and batch normalization in a deep convolutional network for image denoising. The DIV2K dataset introduced in the NTIRE 2017 challenge provides high-quality high-resolution images for image restoration research. SSIM, proposed by Wang et al., complements pixel-error measures by comparing local image structure.

## 3. Methodology
### 3.1 Dataset and preprocessing
One hundred images from the DIV2K validation set are used. Images are converted to RGB and resized to 128×128 for evaluation. Synthetic noisy observations are generated from clean reference images.

### 3.2 Noise models
Gaussian noise is added with standard deviations 10/255, 25/255, and 50/255. Salt-and-pepper noise uses corruption probabilities 0.01, 0.03, and 0.05. Speckle noise uses variances 0.01, 0.04, and 0.08. These represent low, medium, and high severity, respectively.

### 3.3 Basic CNN
The Basic CNN consists of four 3×3 convolution layers with channels 3, 32, 64, 32, and 3. ReLU is applied after the first three convolutions, and a sigmoid is applied at the output. The network predicts the clean image directly. It was trained using MSE loss and Adam with learning rate 0.001, batch size 16, 128×128 training images, and 2 epochs for each noise/severity configuration.

### 3.4 DnCNN
The DnCNN implementation has depth 17 and 64 channels. It predicts a noise residual, which is subtracted from the noisy input to produce the denoised image. Training used MSE loss, Adam with initial learning rate 0.001, cosine annealing, 64×64 patches, batch size 48, 300 steps per epoch, and 40 epochs. The training pipeline used mixed-noise augmentation and produced one checkpoint per noise family.

### 3.5 Evaluation metrics
MSE is the average squared difference between clean and predicted pixel values. PSNR is computed from MSE using the image data range and is expressed in decibels. SSIM evaluates local luminance, contrast, and structural similarity. Higher PSNR and SSIM indicate better reconstruction; lower MSE indicates better reconstruction. The aggregate summary available for this study contains PSNR and SSIM, while the interactive application also calculates MSE for individual uploaded images.

## 4. Experimental design
For each of the three noise families and three severity levels, the same set of 100 validation images is evaluated by both architectures. This yields 9 conditions × 2 models × 100 images = 1,800 predictions. The two models receive the corresponding noisy input under the app/evaluation pipeline.

The training procedures differ: Basic CNN uses two epochs per severity-specific checkpoint, whereas DnCNN uses 40 epochs and mixed-noise training. Accordingly, the comparison should be interpreted as a comparison of the trained implementations and not as a controlled study of architecture alone.

## 5. Results
### 5.1 Condition-wise performance
| Noise | Severity | Basic PSNR | DnCNN PSNR | Basic SSIM | DnCNN SSIM |
|---|---|---:|---:|---:|---:|
| Gaussian | Low | 24.3866 | 32.3880 | 0.8463 | 0.9344 |
| Gaussian | Medium | 23.6477 | 28.2716 | 0.7783 | 0.8626 |
| Gaussian | High | 22.3273 | 24.8134 | 0.6864 | 0.7273 |
| Salt-and-pepper | Low | 24.2557 | 47.9235 | 0.8185 | 0.9985 |
| Salt-and-pepper | Medium | 23.6713 | 42.9701 | 0.7552 | 0.9955 |
| Salt-and-pepper | High | 22.8940 | 40.3946 | 0.7239 | 0.9918 |
| Speckle | Low | 24.3969 | 33.1806 | 0.8252 | 0.9574 |
| Speckle | Medium | 23.7249 | 29.8117 | 0.7818 | 0.9167 |
| Speckle | High | 23.2522 | 28.0670 | 0.7507 | 0.8816 |

### 5.2 Overall comparison
Across the nine conditions, the unweighted mean PSNR is 23.6174 dB for Basic CNN and 34.2023 dB for DnCNN. Mean SSIM is 0.7740 and 0.9184, respectively. DnCNN exceeds Basic CNN in both metrics in all nine conditions.

The especially high DnCNN PSNR under salt-and-pepper noise should be interpreted carefully. Impulse corruption is sparse, so a global pixel-averaged MSE can remain small when most pixels are unchanged or accurately reconstructed. Future evaluations should additionally report error restricted to corrupted pixels and inspect qualitative outputs.

## 6. Discussion
The results indicate that the implemented DnCNN system reconstructs images more accurately than the implemented Basic CNN under the tested synthetic conditions. Residual learning provides a natural formulation for denoising because the model estimates the corruption component. However, the difference in training duration, model capacity, and noise augmentation is a confounding factor. A matched training protocol is needed to determine how much of the observed gap is attributable specifically to architecture.

## 7. Conclusion
This project compared Basic CNN and DnCNN for three synthetic noise types at three severity levels using 100 DIV2K validation images per condition. DnCNN achieved higher average PSNR and SSIM and outperformed Basic CNN in all nine conditions. A Streamlit application demonstrates the models on user-provided images and displays image-quality metrics. Future work will use matched training budgets, additional real-world datasets, robust timing measurements, and impulse-noise-specific evaluation.

## References
[1] K. Zhang, W. Zuo, Y. Chen, D. Meng, and L. Zhang, “Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising,” *IEEE Transactions on Image Processing*, vol. 26, no. 7, pp. 3142–3155, 2017.

[2] E. Agustsson and R. Timofte, “NTIRE 2017 Challenge on Single Image Super-Resolution: Dataset and Study,” in *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops*, 2017, pp. 1122–1131.

[3] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, “Image Quality Assessment: From Error Visibility to Structural Similarity,” *IEEE Transactions on Image Processing*, vol. 13, no. 4, pp. 600–612, 2004.

**Submission note:** This is a student project manuscript draft, not a claim of peer-reviewed publication. Add institutional details, verify all references against the required citation style, and include figures from the final application/evaluation before submission.
