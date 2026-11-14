# CNN-Based Image Denoising: Project Report

**Team:** Bandi Navadeep, Rajan Nayak, Rohit  
**Institution:** [Add college/university]  
**Department / Course:** [Add department and course]  
**Guide:** [Add guide name]  
**Submission date:** [Add date]

## Abstract
This project implements and compares two convolutional neural network approaches for image denoising: a compact Basic CNN and a 17-layer DnCNN. The systems are evaluated on synthetic Gaussian, salt-and-pepper, and speckle noise at low, medium, and high severities. A Streamlit application allows a user to upload an image, generate a noisy input, view both model outputs, compare MSE/PSNR/SSIM, and download results. The evaluation uses 100 DIV2K validation images for each of nine noise/severity settings, giving 18 model-condition experiments and 1,800 predictions. Across the nine conditions, DnCNN achieved mean PSNR 34.2023 dB and mean SSIM 0.9184, compared with 23.6174 dB and 0.7740 for Basic CNN. DnCNN was higher on both metrics in every evaluated condition.

## 1. Introduction
Image noise can reduce visual quality and interfere with later image-processing tasks. Image denoising estimates a clean image from a corrupted observation. Convolutional neural networks are useful because they learn local image patterns and mappings from noisy inputs to clean targets. This project provides a practical comparison between direct image reconstruction with a compact CNN and residual noise estimation with DnCNN.

## 2. Objectives
1. Implement a Basic CNN and a DnCNN denoising model.
2. Evaluate both systems on Gaussian, salt-and-pepper, and speckle noise.
3. Compare low, medium, and high noise severities.
4. Measure reconstruction quality using MSE, PSNR, and SSIM.
5. Build a Streamlit interface for visual and numerical comparison.

## 3. Dataset and noise generation
The evaluation uses 100 images from the DIV2K validation set. Images are converted to RGB and resized to 128×128 for evaluation.

| Noise | Low | Medium | High |
|---|---:|---:|---:|
| Gaussian standard deviation | 10/255 | 25/255 | 50/255 |
| Salt-and-pepper probability | 0.01 | 0.03 | 0.05 |
| Speckle variance | 0.01 | 0.04 | 0.08 |

Noise is synthesized from clean validation images. Each model is evaluated against the corresponding clean reference.

## 4. Model architectures
### 4.1 Basic CNN
The Basic CNN contains four 3×3 convolution layers with channel progression 3→32→64→32→3. ReLU activations follow the first three convolution layers, and a sigmoid constrains the final output to the [0,1] range. The network predicts the clean image directly.

### 4.2 DnCNN
The DnCNN uses depth 17 and 64 feature channels. It estimates the noise component from the noisy image. The denoised result is obtained by subtracting the predicted noise from the input. This is residual learning: the network learns the corruption rather than reconstructing every clean-image detail from scratch.

## 5. Training configuration
The Basic CNN was trained with MSE loss and Adam (learning rate 0.001), using 128×128 images, batch size 16, and 2 epochs for each of nine noise/severity configurations.

The DnCNN training configuration used MSE loss, Adam (initial learning rate 0.001), a cosine annealing learning-rate schedule, 64×64 patches, batch size 48, 300 steps per epoch, and 40 epochs. Its training procedure used mixed-noise augmentation. The project contains three DnCNN checkpoints, one for each noise family.

**Fair-comparison note:** These training protocols are not identical. Consequently, results compare the trained systems as implemented; they do not isolate the effect of network architecture under equal training budgets and identical data distributions.

## 6. Evaluation metrics
- **MSE:** mean squared pixel error between reference and prediction; lower is better.
- **PSNR:** logarithmic measure derived from MSE; higher is better.
- **SSIM:** compares local luminance, contrast, and structural similarity; higher is better.

The Streamlit application calculates all three metrics for a selected image. The aggregate evaluation CSV reports PSNR and SSIM; aggregate MSE is not reported here because it is not present in the summary file.

## 7. Results
Each row below averages 100 DIV2K validation images.

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

### Overall averages
| Metric | Basic CNN | DnCNN |
|---|---:|---:|
| Mean PSNR (dB), across 9 conditions | 23.6174 | 34.2023 |
| Mean SSIM, across 9 conditions | 0.7740 | 0.9184 |

DnCNN achieved higher PSNR and SSIM in all nine conditions. The very high salt-and-pepper PSNR values should be interpreted in context: salt-and-pepper corruption affects a relatively small fraction of pixels, while global MSE averages errors over all pixels. The result is mathematically consistent with that metric definition but does not alone describe how well every corrupted pixel was restored.

## 8. Application workflow
1. User uploads an image.
2. The app converts it to RGB and resizes it to 128×128.
3. User selects a noise family and severity.
4. The app generates one noisy input.
5. The same noisy input is passed to Basic CNN and DnCNN.
6. The interface displays the original, noisy, and two denoised outputs.
7. MSE, PSNR, and SSIM are displayed and outputs can be downloaded.

## 9. Conclusion
For the evaluated DIV2K validation images and synthetic noise settings, DnCNN outperformed the Basic CNN in PSNR and SSIM across all nine conditions. The Streamlit application makes this comparison accessible through visual outputs and image-quality metrics. The result supports the effectiveness of residual denoising in this implementation, while the differing training protocols mean that further controlled experiments are needed to attribute the full performance gap to architecture alone.

## 10. Limitations and future work
- Train both architectures with matched epochs, optimizer schedules, data, and augmentation to isolate architectural effects.
- Evaluate on additional datasets and real-world noisy photographs.
- Report confidence intervals and per-image distributions, not only condition means.
- Add metrics focused on corrupted pixels for impulse noise.
- Benchmark inference time with warm-up runs and repeated measurements; the existing CSV timing values are too small to treat as reliable.
- Consider larger input resolutions and patch-based inference.

## References
1. K. Zhang, W. Zuo, Y. Chen, D. Meng, and L. Zhang, “Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising,” *IEEE Transactions on Image Processing*, vol. 26, no. 7, pp. 3142–3155, 2017.
2. E. Agustsson and R. Timofte, “NTIRE 2017 Challenge on Single Image Super-Resolution: Dataset and Study,” in *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops*, 2017, pp. 1122–1131.
3. Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, “Image Quality Assessment: From Error Visibility to Structural Similarity,” *IEEE Transactions on Image Processing*, vol. 13, no. 4, pp. 600–612, 2004.
