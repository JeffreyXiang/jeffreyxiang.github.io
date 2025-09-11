---
layout: blog
title: "用C++画图（四）：包围盒"
excerpt: "本文介绍了二维图形的AABB（轴向平行包围盒）实现方法，包括圆、矩形、三角形、椭圆及贝塞尔曲线等。通过包围盒优化计算区域，提高绘制效率，并展示了圆化与环化后的AABB处理。"
series: "Draw with C++"
blog_id: draw-with-cpp-4
permalink: /zh/blogs/draw-with-cpp-4
teaser: /imgs/draw-with-cpp-4/teaser.png
lang: zh
github: https://github.com/JeffreyXiang/DrawWithCpp/tree/master/src/4_AABB
---

## 前言

今天我们来接着讲《用C++画图》的第四期。上一篇文章的末尾表明我这次本来是要讲CSG的，可最近我用前面已完成的部分画图时发现一个效率上的问题。我们知道，通过图形的SDF可以判断像素位于图形的内部还是外部，内部上色即可。可是如果不限制对于每个图元的计算区域，我们绘制每个图元时都要遍历整个画布的像素。这样一来，如果整个图片由大量小图元构成，那计算量将会大大浪费。

在第二篇 [用C++画图（二）：绘图算法](/zh/blogs/draw-with-cpp-2/#有符号距离场算法) 中，我就提到了只对“可能被填充的区域”中的像素进行判断，来减少运算量。当时是只对线段和圆单独计算，那今天我们的目的是以接口的形式为每个图元判断“可能被填充的区域”，或者另一个学名：包围盒。

## 包围盒

以下是 [百度百科](https://baike.baidu.com/item/%E5%8C%85%E5%9B%B4%E7%9B%92) 对包围盒的介绍：

> 包围盒是一种求解离散点集最优包围空间的算法，基本思想是用体积稍大且特性简单的几何体（称为包围盒）来近似地代替复杂的几何对象。  
> 常见的包围盒算法有AABB包围盒、包围球、方向包围盒OBB以及固定方向凸包FDH。  
> 碰撞检测问题在虚拟现实、计算机辅助设计与制造、游戏及机器人等领域有着广泛的应用，甚至成为关键技术。而包围盒算法是进行碰撞干涉初步检测的重要方法之一。

简而言之，包围盒就是一个包含了原来的复杂几何体的简单几何体，这个简单几何体拥有边界便于计算的性质。所以说为了填充原复杂图形，只需对包围盒内的像素计算即可。

那么边界最易于计算的几何体是啥呢？很显然，就是四边都平行于像素网格的矩形。在常用图形处理软件（如PS）中你可以发现，所有的图元都是使用这种包围盒来描述的：

<img src="/imgs/draw-with-cpp-4/1.png" style="width: 49%">
<img src="/imgs/draw-with-cpp-4/2.png" style="width: 49%">

这种包围盒名为：AABB(Axially Aligned Bounding Box，轴向平行包围盒)，描述它也非常容易。对于一个二维的AABB，只需要四个量便能确定，即为：

```cpp
typedef struct
{
    double xMin;    //横坐标下界
    double yMin;    //纵坐标下界
    double xMax;    //横坐标上界
    double yMax;    //纵坐标上界
}AABBdata;
````

那么我们在总图元类 `Figure` 中添加一个返回AABB的接口：

```cpp
virtual AABBdata AABB() = 0;
```

接下来在各个派生类中完成接口即可。

### 圆

<img src="/imgs/draw-with-cpp-4/3.png" style="width: 200px; float: left; margin-right: 10px">

```cpp
AABBdata AABB()
{
    AABBdata res;
    res.xMin = center.x - radius;
    res.yMin = center.y - radius;
    res.xMax = center.x + radius;
    res.yMax = center.y + radius;
    return res;
}
```

### 线段（胶囊）

<img src="/imgs/draw-with-cpp-4/4.png" style="width: 200px; float: left; margin-right: 10px">

```cpp
AABBdata AABB()
{
    AABBdata res;
    res.xMin = min(endpoint1.x, endpoint2.x) - radius;
    res.yMin = min(endpoint1.y, endpoint2.y) - radius;
    res.xMax = max(endpoint1.x, endpoint2.x) + radius;
    res.yMax = max(endpoint1.y, endpoint2.y) + radius;
    return res;
}
```

### 变宽胶囊

<img src="/imgs/draw-with-cpp-4/6.png" style="width: 200px; float: left; margin-right: 10px">

```cpp
AABBdata AABB()
{
    AABBdata res;
    res.xMin = min(endpoint1.x - radius1, endpoint2.x - radius2);
    res.yMin = min(endpoint1.y - radius1, endpoint2.y - radius2);
    res.xMax = max(endpoint1.x + radius1, endpoint2.x + radius2);
    res.yMax = max(endpoint1.y + radius1, endpoint2.y + radius2);
    return res;
}
```

### 矩形

<img src="/imgs/draw-with-cpp-4/5.png" style="width: 200px; float: left; margin-right: 10px">

```cpp
AABBdata AABB()
{
    AABBdata res;
    double cos_ = cos(theta);
    double sin_ = sin(theta);
    double dx = max(abs(size.x * cos_ - size.y * sin_), abs(size.x * cos_ + size.y * sin_));
    double dy = max(abs(size.x * sin_ + size.y * cos_), abs(size.x * sin_ - size.y * cos_));
    res.xMin = center.x - dx;
    res.yMin = center.y - dy;
    res.xMax = center.x + dx;
    res.yMax = center.y + dy;
    return res;
}
```

### 三角形

<img src="/imgs/draw-with-cpp-4/7.png" style="width: 200px; float: left; margin-right: 10px">

```cpp
AABBdata AABB()
{
    AABBdata res;
    res.xMin = min(min(vertex1.x, vertex2.x), vertex3.x);
    res.yMin = min(min(vertex1.y, vertex2.y), vertex3.y);
    res.xMax = max(max(vertex1.x, vertex2.x), vertex3.x);
    res.yMax = max(max(vertex1.y, vertex2.y), vertex3.y);
    return res;
}
```

### 多边形

<img src="/imgs/draw-with-cpp-4/10.png" style="width: 200px; float: left; margin-right: 10px">

```cpp
AABBdata AABB()
{
    AABBdata res;
    res.xMin = vertex[0].x;
    res.xMax = vertex[0].x;
    res.yMin = vertex[0].y;
    res.yMax = vertex[0].y;
    for (int i = 0; i < vertexNumber; i++)
    {
        if (vertex[i].x > res.xMax)
            res.xMax = vertex[i].x;
        if (vertex[i].x < res.xMin)
            res.xMin = vertex[i].x;
        if (vertex[i].y > res.yMax)
            res.yMax = vertex[i].y;
        if (vertex[i].y < res.yMin)
            res.yMin = vertex[i].y;
    }
    return res;
}
```

### 扇形

<img src="/imgs/draw-with-cpp-4/8.png" style="width: 200px; float: left; margin-right: 10px">

```cpp
AABBdata AABB()
{
    AABBdata res;
    double dtheta = theta2 - theta1;
    double x1 = radius * cos(theta1), y1 = radius * sin(theta1);
    double x2 = radius * cos(theta2), y2 = radius * sin(theta2);
    if ((0 - theta1) - 2 * PI * floor((0 - theta1) / (2 * PI)) < dtheta)
        res.xMax = center.x + radius;
    else
        res.xMax = center.x + max(max(x1, x2), 0.);
    if ((PI / 2 - theta1) - 2 * PI * floor((PI / 2 - theta1) / (2 * PI)) < dtheta)
        res.yMax = center.y + radius;
    else
        res.yMax = center.y + max(max(y1, y2), 0.);
    if ((PI - theta1) - 2 * PI * floor((PI - theta1) / (2 * PI)) < dtheta)
        res.xMin = center.x - radius;
    else
        res.xMin = center.x + min(min(x1, x2), 0.);
    if ((PI * 3 / 2 - theta1) - 2 * PI * floor((PI * 3 / 2 - theta1) / (2 * PI)) < dtheta)
        res.yMin = center.y - radius;
    else
        res.yMin = center.y + min(min(y1, y2), 0.);
    return res;
}
```

### 弧

<img src="/imgs/draw-with-cpp-4/9.png" style="width: 200px; float: left; margin-right: 10px">

```cpp
AABBdata AABB()
{
    AABBdata res;
    double dtheta = theta2 - theta1;
    double x1 = radius1 * cos(theta1), y1 = radius1 * sin(theta1);
    double x2 = radius1 * cos(theta2), y2 = radius1 * sin(theta2);
    if ((0 - theta1) - 2 * PI * floor((0 - theta1) / (2 * PI)) < dtheta)
        res.xMax = center.x + radius1 + radius2;
    else
        res.xMax = center.x + max(x1, x2) + radius2;
    if ((PI / 2 - theta1) - 2 * PI * floor((PI / 2 - theta1) / (2 * PI)) < dtheta)
        res.yMax = center.y + radius1 + radius2;
    else
        res.yMax = center.y + max(y1, y2) + radius2;
    if ((PI - theta1) - 2 * PI * floor((PI - theta1) / (2 * PI)) < dtheta)
        res.xMin = center.x - radius1 - radius2;
    else
        res.xMin = center.x + min(x1, x2) - radius2;
    if ((PI * 3 / 2 - theta1) - 2 * PI * floor((PI * 3 / 2 - theta1) / (2 * PI)) < dtheta)
        res.yMin = center.y - radius1 - radius2;
    else
        res.yMin = center.y + min(y1, y2) - radius2;
    return res;
}
```

### 椭圆

<img src="/imgs/draw-with-cpp-4/11.png" style="width: 200px; float: left; margin-right: 10px">

```cpp
AABBdata AABB()
{
    AABBdata res;
    double cos_ = cos(theta);
    double sin_ = sin(theta);
    double dx = max(abs(a * cos_ - b * sin_), abs(a * cos_ + b * sin_));
    double dy = max(abs(a * sin_ + b * cos_), abs(a * sin_ - b * cos_));
    res.xMin = center.x - dx;
    res.yMin = center.y - dy;
    res.xMax = center.x + dx;
    res.yMax = center.y + dy;
    return res;
    }
```

### 二阶贝塞尔曲线

<img src="/imgs/draw-with-cpp-4/12.png" style="width: 200px; float: left; margin-right: 10px">

```cpp
AABBdata AABB()
{
    AABBdata res;
    res.xMin = min(min(A.x, B.x), C.x);
    res.yMin = min(min(A.y, B.y), C.y);
    res.xMax = max(max(A.x, B.x), C.x);
    res.yMax = max(max(A.y, B.y), C.y);
    return res;
}
```

## 基本操作

实行了基本操作（如上一篇 [用C++画图（三）：基本图形](/blog/7#基本操作) 中介绍的圆化和环化）之后，对应的AABB包围盒会发生变化，在父类中定义一个方法 `AABBdata tAABB()` 来计算变换后的AABB：

```cpp
AABBdata tAABB()
{
    AABBdata res = AABB();
    res.xMin -= attribute.roundedRadius + (attribute.annularRadius >= 0 ? attribute.annularRadius : 0);
    res.yMin -= attribute.roundedRadius + (attribute.annularRadius >= 0 ? attribute.annularRadius : 0);
    res.xMax += attribute.roundedRadius + (attribute.annularRadius >= 0 ? attribute.annularRadius : 0);
    res.yMax += attribute.roundedRadius + (attribute.annularRadius >= 0 ? attribute.annularRadius : 0);
    return res; 
}
```

## 画布绘制

有了AABB包围盒之后，就可以直接根据AABB来缩小画布上填充的计算范围，这样一来我们可以利用多态性在 `Image` 中实现针对所有 `Figure` 的绘制，算法如下：

```cpp
Image& draw(Figure& s)
{
    Figure::AABBdata b = s.tAABB();
    //裁剪，避免溢出
    int xMin = max((int)floor(b.xMin), 0);
    int yMin = max((int)floor(b.yMin), 0);
    int xMax = min((int)ceil(b.xMax), (int)width - 1);
    int yMax = min((int)ceil(b.yMax), (int)height - 1);
    for (int u = xMin; u <= xMax; u++)
        for (int v = yMin; v <= yMax; v++)
        {
            double SDF = s.tSDF({u, v});
            if (SDF < 0)
            setPixel(u, v, s.getAttribute().color);
        }
    return *this;
}
```

## 代码

本节代码请查看：[🔗Github: JeffreyXiang/DrawWithCpp](https://github.com/JeffreyXiang/DrawWithCpp/tree/master/src/4_AABB)

## 总结

本篇我们介绍了AABB，即轴向平行包围盒，以及它在优化计算中的作用。接着完成了各个基本图形的AABB计算方法，并利用多态性实现在画布中根据AABB直接绘制基本图形。

## 预告

可能会先讲如何加入文字，大概率是以点阵字库的方式，因为要先完成一个画图的程序来配合我的第一个应用——离散傅里叶变换计算器。
