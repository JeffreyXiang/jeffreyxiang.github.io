---
order_id: gram-hd
title: "GRAM-HD: 3D-Consistent Image Generation at High Resolution with Generative Radiance Manifolds"
authors: "<span class='me'>Jianfeng Xiang</span>, Jiaolong Yang, Yu Deng, Xin Tong"
location: "2023 International Conference on Computer Vision, ICCV 2023"
teaser: "/videos/gram-hd.mp4"
teaser_type: "video"
page_url: "https://jeffreyxiang.github.io/GRAM-HD/"
abstract: "Recent works have shown that 3D-aware GANs trained on unstructured single image collections can generate multiview images of novel instances. The key underpinnings to achieve this are a 3D radiance field generator and a volume rendering process. However, existing methods either cannot generate high-resolution images (e.g., up to 256X256) due to the high computation cost of neural volume rendering, or rely on 2D CNNs for image-space upsampling which jeopardizes the 3D consistency across different views. This paper proposes a novel 3D-aware GAN that can generate high resolution images (up to 1024X1024) while keeping strict 3D consistency as in volume rendering. Our motivation is to achieve super-resolution directly in the 3D space to preserve 3D consistency. We avoid the otherwise prohibitively-expensive computation cost by applying 2D convolutions on a set of 2D radiance manifolds defined in the recent generative radiance manifold (GRAM) approach, and apply dedicated loss functions for effective GAN training at high resolution. Experiments on FFHQ and AFHQv2 datasets show that our method can produce high-quality 3D-consistent results that significantly outperform existing methods."
---