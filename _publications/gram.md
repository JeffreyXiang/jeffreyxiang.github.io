---
id: "gram"
title: "GRAM: Generative Radiance Manifolds for 3D-Aware Image Generation"
authors: "Yu Deng, Jiaolong Yang, <span class='me'>Jianfeng Xiang</span>, Xin Tong"
location: "2022 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2022, Oral Presentation"
teaser: "/videos/gram.mp4"
teaser_type: "video"
page_url: "https://yudeng.github.io/GRAM/"
abstract: "3D-aware image generative modeling aims to generate 3D-consistent images with explicitly controllable camera poses. Recent works have shown promising results by training neural radiance field (NeRF) generators on unstructured 2D image collections, but they still can not generate highly-realistic images with fine details. A critical reason is that the high memory and computation cost of volumetric representation learning greatly restricts the number of point samples for radiance integration during training. Deficient point sampling not only limits the expressive power of the generator to handle high-frequency details but also impedes effective GAN training due to the noise caused by unstable Monte Carlo sampling. We propose a novel approach that regulates point sampling and radiance field learning on 2D manifolds, embodied as a set of implicit surfaces in the 3D volume learned with GAN training. For each viewing ray, we calculate the ray-surface intersections and accumulate their radiance predicted by the network. We show that by training and rendering such radiance manifolds, our generator can produce high quality images with realistic fine details and strong visual 3D consistency."
---