---
order_id: octworld
title: "OctWorld: Long-Range World-Consistent Video Generation with Octree-Based 3D Mapping"
authors: "Zelong Lv, Sicheng Xu, <span class='me'>Jianfeng Xiang</span>, Ruicheng Wang, Yue Dong, Yu Deng, Guangzhong Sun, Jiaolong Yang"
location: "2026 European Conference on Computer Vision, ECCV 2026"
venue: "ECCV 2026"
teaser: "/videos/octworld.mp4"
teaser_type: "video"
page_url: "https://microsoft.github.io/OctWorld/"
abstract: "We present OctWorld, a video diffusion framework with persistent 3D memory for generating explorable, world-consistent, and high-fidelity visual scenes. Given a single image, OctWorld performs stable autoregressive world generation along user-specified camera trajectories. We focus on long-range generation, characterized by extended camera paths and wide viewpoint coverage, where preserving spatial consistency is particularly challenging when previously generated regions are revisited. To address this problem, we introduce OctMap, an extensible and spatially adaptive 3D memory that progressively fuses generated visual observations and their corresponding depth maps into a global representation. OctMap employs TSDF fusion within a dynamic sparse octree whose spatial resolution adapts to image evidence. This design preserves geometric and appearance details across diverse scene scales while maintaining low memory overhead. Experiments demonstrate that OctWorld generates long-range, spatially consistent videos and outperforms prior methods on both existing benchmarks and challenging long-range generation settings. OctMap also provides clear advantages over point-based caches and fixed-resolution TSDF volumes."
---
