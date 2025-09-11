---
layout: blog
title: "用C++画图（六）：文字"
excerpt: "在本节中，我们研究了点阵字库的结构和绘制方法，并由此引申了图片的变换与插入，最终实现了在图中任意位置，任意旋转，任意尺寸的文字的插入。"
series: "Draw with C++"
blog_id: draw-with-cpp-6
permalink: /zh/blogs/draw-with-cpp-6
teaser: /imgs/draw-with-cpp-6/teaser.png
lang: zh
github: https://github.com/JeffreyXiang/DrawWithCpp/tree/master/src/6_font
---

## 前言

作者这一个月投身于大学生创新训练计划，实在没有时间更新博客。现在正值暑假中期，项目告一段落，接下来则要为开学考试做准备，于是有一个几天的闲暇时期，便想到来继续我们的《用C++画图》。

上篇文章里，我们研究了如何高质量地进行图像的缩放，即如何保留更多的频域信息。本文中，我们会运用图片的缩放在我们的画布上添加文字。

## 点阵字体

在计算机技术的发展过程中，涌现了很多字体种类。现代计算机中的字体大多是 TTF 格式的。

TTF（TrueTypeFont）是Apple公司和Microsoft公司共同推出的字体文件格式，随着windows的流行，已经变成最常用的一种字体文件表示方式。其中的字符轮廓由直线和二次贝塞尔曲线片段所构成。这种特性使得 TTF 字体能以较小的空间占用实现无限的分辨率（矢量表示）。

绘制以二次贝塞尔曲线为边界的区域是较难实现的，于是在本文中，我们退而求其次，使用一种上个时代的字体技术——点阵字体。

点阵字体是把每一个字符都分成一个矩阵，然后用每个点的虚实来表示字符的轮廓。点阵字体也叫位图字体，其中每个字形都以一组二维像素信息表示。

点阵字体广泛运用与 Dos, Unix 等操作系统中，较老版本的 Windows 中，命令提示行中使用的也是点阵字体。  
如果你使用过 Free Pascal 等古老的软件，就会发现其中大量使用了点阵字体。

<div>
<figure class="image" style="width: 49%; display: inline-block">
<img src="/imgs/draw-with-cpp-6/1.png">
<figcaption>TTF字体与贝塞尔曲线</figcaption>
</figure>
<figure class="image" style="width: 50%; display: inline-block">
<img src="/imgs/draw-with-cpp-6/2.jpg">
<figcaption>点阵字体</figcaption>
</figure>
</div>

考虑到点阵字体的数据结构，其可以直接一一对应地填充出一个二值的位图，算法简单，显示速度非常快，不像矢量字体需要计算。

## 点阵字库

点阵字库就是把一个个的点阵字体按一定的顺序组合成的一个二进制文件。

我们这里采用 CSDN 博主 [@星沉地动](https://me.csdn.net/qq446252221) 的点阵字库提取工具：[🔗通用点阵字库生成工具](https://blog.csdn.net/qq446252221/article/details/53188278)  
下载链接：[🔗FontMaker-V1.2.0.zip](http://446252221.ys168.com)

程序打开后，配置选择水平扫描、高位在前、可变宽度、BIN文件，字体的大小根据需要自行设置，即可生成我们的测试用点阵字库（如图）：

<figure class="image" style="width: 50%; margin-left: auto; margin-right: auto">
<img src="/imgs/draw-with-cpp-6/3.png">
<figcaption>应用界面</figcaption>
</figure>

此应用的运行需要 VC++ 2010 之前的运行库，在较新的设备上可能不会默认安装，需要手动安装。  
若缺少此运行库将会提示程序并行配置不正确。

此工具生成的可变宽度字库，格式如下所描述：  
对于每个字符，前两个字节代表该字的宽度，后面跟着该字的点阵数据，大小为 固定高度 × 固定宽度，所有字符按 ASCII 顺序依次排列。

<img src="/imgs/draw-with-cpp-6/4.png" style="width: 50%; float: right; margin-left: 10px">

```cpp
struct character
{
    short   width;
    BYTE    data[fixedHeight * fixedWidth];
}
````

了解了这些，我们就可以尝试写一个模块来解析字库的数据：

定义 `class Font`

```cpp
class Font
{
    public:
        //单个字符的点阵
        typedef struct
        {
            uint16_t width;
            bool* data;
        }FontMatrix;

    private:
        int height;

        //所有字符的点阵数组
        FontMatrix* fontData;

        //解析特定字符的点阵
        FontMatrix getFontMatrix(char* data, char c)
        {
            FontMatrix res;

            //字符在字库中的偏移量
            int offset = (2 + height * height / 8) * (c - 32);

            //读取宽度和字符
            res.width = ((uint16_t)*(data + offset) << 8) | ((uint8_t)*(data + offset + 1));
            res.data = new bool[res.width * height];
            for (int i = 0; i < height; i++)
                for (int j = 0; j < res.width; j++)
                {
                    res.data[i * res.width + j] = *(data + offset + 2 + height / 8 * i + j / 8) & (0x80 >> j % 8);
                }
            return res;
        }

    public:
        //构造函数
        Font(const char* filename, int height)
        {
            this->height = height;

            //读取整个文件
            ifstream f(filename, ios::binary); 
            if (!f.is_open())
            {
                cout<<"Error: no such file.\n";
                exit(0);
            }
            int length;  
            f.seekg(0, ios::end);
            length = f.tellg(); 
            f.seekg(0, ios::beg);
            char* data = new char[length];
            f.read(data, length);
            f.close();

            //解析字符点阵
            fontData = new FontMatrix[96];
            for (int i = 32; i < 128; i++)
            {
                fontData[i - 32] = getFontMatrix(data, i);
            }
            delete[] data;
        }

        ~Font() { delete[] fontData; }

        //获取特定字符的点阵
        FontMatrix operator[](int idx) { return fontData[idx - 32]; }

        //获取字体高度
        int getHeight() { return height; }

        //获取字符串宽度
        int stringWidth(string str)
        {
            int width = 0;
            for (int i = 0; i < str.length(); i++)
                width += (*this)[str[i]].width;
            return width;
        }
};
```

我们写一个简单的测试程序，输出点阵为 ASCII 艺术的形式：

```cpp
int main()
{
    Font font("../font/DengXian_ASCII_128x.bin", 128);
    Font::FontMatrix dots = font['@'];
    for (size_t i = 0; i < font.getHeight(); i+=2)
    {
        for (size_t j = 0; j < dots.width; j++)
            cout << (dots.data[i * dots.width + j] ? '@' : '.');
        cout << endl;
    }
}
```

编译运行，可以看到输出：

<figure class="image" style="width: 50%; margin-left: auto; margin-right: auto">
<img src="/imgs/draw-with-cpp-6/5.png">
</figure>

确实能够解析并输出 `'@'` 这个字符，说明上面我们的模块工作正常。

## 绘制文字

在上面的模块中，我们成功的解析并获得了所有 ASCII 字符的点阵，但是点阵的输出不能总依靠 ASCII 艺术，我们要将它生成图片。  
这两者的方法是一摸一样的逐行扫描，不过在生成到图片时，我们需要预先计算出字符串需要的画布大小，之后再依次填充即可。

我们为 `class Image` 类新增一个构造函数用来生成图片的原图：

```cpp
//生成文字原图（一一对应）
Image::Image(string str, Font& font, Color color)
{
    //计算文字宽高
    width = font.stringWidth(str);
    height = font.getHeight();

    //创建画布
    data = new Color[width * height];
    for (int i = 0; i < width * height; i++)
        data[i].rgba(0, 0, 0, 0);

    //依次绘制文字
    int temp = 0;
    for (int i = 0; i < str.length(); i++)
    {
        for (int j = 0; j < height; j++)
            for (int k = 0; k < font[str[i]].width; k++)
                if (font[str[i]].data[j * font[str[i]].width + k])
                    overliePixel(k + temp, height - j - 1, color);
        temp += font[str[i]].width;
    }
}
````

生成并输出一张简单的文字原图试一试：

```cpp
int main()
{
    Font font("../font/DengXian_ASCII_128x.bin", 128);
    Image test("(@_@)", font, { 0, 0, 0 });
    test.saveBMP("../data/output.bmp");
}
```

<figure class="image" style="width: 50%; margin-left: auto; margin-right: auto">
<img src="/imgs/draw-with-cpp-6/6.png">
</figure>

如果你看到一个晕乎乎的颜文字，那么就可以接着往下看了。

## 变换与插入

现在我们拥有了带有文字的图片，那么自然而然，下一步就是将文字图片插入我们的目标图片中。

如何插入呢？

我们可以用一个形象的说法来解释：
我们将被插入的图片摆放在目标图片的上面，他将覆盖住目标图片的某些像素，我们只需要计算出这些被覆盖的像素的颜色就能够完成图片的插入。

<figure class="image" style="width: 75%; margin-left: auto; margin-right: auto">
<img src="/imgs/draw-with-cpp-6/7.png">
<figcaption>淡红色为被覆盖像素</figcaption>
</figure>

我们获得这些被覆盖像素对应在插入图中的位置，然后再进行采样，计算出颜色。

考虑到在上一章里，我们已经了解如何对图片的特定位置的颜色值进行采样。那么我们在此处只需要知道，怎么获得被覆盖像素对应在插入图中的位置。

为此，先来定义一下怎么确定插入图片的形态：
我们用插入图片上的一个锚点，这个锚点在目标图上的位置，插入图绕锚点的旋转，插入图的长宽来定义。如图：

<figure class="image" style="width: 75%; margin-left: auto; margin-right: auto">
<img src="/imgs/draw-with-cpp-6/8.png">
<figcaption>插入图的位置如图确定</figcaption>
</figure>

其中锚点在插入图上的位置（红）我们用 0 \~ 1 的比例来表示，0 代表最左（下），1 代表最右（上）。
这样表示的好处是很多的，这使得图像按特定位置对齐，例如向左对齐、居中对齐、向右对齐，只需要设置锚点的横坐标为 0、0.5、1 即可，不需要根据图像尺寸进行计算，在外部使用会很方便。

由于插入图绕锚点旋转，为了得到像素在原图上的位置，我们只需要将从锚点到要填充的像素点的矢量转回去，在加上锚点在插入图上的位置矢量即可得到在插入图上的位置，如图：

<figure class="image" style="width: 100%; margin-left: auto; margin-right: auto">
<img src="/imgs/draw-with-cpp-6/9.png">
<figcaption>目标图坐标变换到插入图坐标</figcaption>
</figure>

旋转矩阵在前面已多次提到，这里不再讲。

为了提升绘制速度，我们也给插入图片使用 AABB 轴对齐包围盒来限制绘制范围，不去遍历整个目标图像。很容易知道图片的四个角总是在 AABB 边界上，我们对转动角度分情况即可。算法为：

```cpp
//插入图片（源，锚点位置，源上锚点位置，宽度，高度，旋转角，采样方法）
Image& Image::insert(Image& src, Vector pos, Vector center, double width, double height, double theta, resampling type)
{
    theta = (theta - 360 * floor(theta / 360)) * PI / 180;
    double cos_ = cos(theta);
    double sin_ = sin(theta);
    double kx = width / src.width;      //缩放比例
    double ky = height / src.height;

    //计算AABB包围盒
    double xMin, xMax, yMin, yMax;      //AABB
    double xL = -center.x * width;      //四角坐标(L,R,T,B: 左右上下)
    double xR = (1 - center.x) * width;
    double yT = (1 - center.y) * height;
    double yB = -center.y * height;
    if (theta < PI / 2)
    {
        xMin = pos.x + xL * cos_ - yT * sin_;
        xMax = pos.x + xR * cos_ - yB * sin_;
        yMin = pos.y + xL * sin_ + yB * cos_;
        yMax = pos.y + xR * sin_ + yT * cos_;
    }
    else if (theta < PI)
    {
        xMin = pos.x + xR * cos_ - yT * sin_;
        xMax = pos.x + xL * cos_ - yB * sin_;
        yMin = pos.y + xL * sin_ + yT * cos_;
        yMax = pos.y + xR * sin_ + yB * cos_;
    }
    else if (theta < 3 * PI / 2)
    {
        xMin = pos.x + xR * cos_ - yB * sin_;
        xMax = pos.x + xL * cos_ - yT * sin_;
        yMin = pos.y + xR * sin_ + yT * cos_;
        yMax = pos.y + xL * sin_ + yB * cos_;
    }
    else
    {
        xMin = pos.x + xL * cos_ - yB * sin_;
        xMax = pos.x + xR * cos_ - yT * sin_;
        yMin = pos.y + xR * sin_ + yB * cos_;
        yMax = pos.y + xL * sin_ + yT * cos_;
    }
    xMin = max(xMin, 0.0); xMax = min(xMax, this->width - 1.0);
    yMin = max(yMin, 0.0); yMax = min(yMax, this->height - 1.0);

    double u, v;                        //插入图上的位置
    
    //按AABB遍历
    for (int i = floor(xMin); i <= ceil(xMax); i++)
        for (int j = floor(yMin); j <= ceil(yMax); j++)
        {
            //  -锚点比例-   ---------锚点到像素的矢量反向旋转----------  -化比例-   -化像素位置-       
            u = (center.x + ((i - pos.x) * cos_ + (j - pos.y) * sin_) / width) * src.width;
            v = (center.y + (-(i - pos.x) * sin_ + (j - pos.y) * cos_) / height) * src.height;
            if (u >= 0 && u < src.width && v >= 0 && v < src.height)    //在插入图内
                overliePixel(i, j, src.resample(u, v, kx, ky, type));
        }
    return *this;
}
```

我们在上一节里生成的放大的石头图片中插入两个晕乎乎的颜文字：

```cpp
int main()
{
    Image test("(@_@)", font, { 255, 255, 255 });
    Image image;
    image.readBMP("../data/test2.bmp");
    image.insert(test, { 600, 300 }, { 0.5, 0.5 }, 200, -15, Image::BICUBIC);
    image.insert(test, { 1200, 850 }, { 0.5, 0.5 }, 200, 200, 30, Image::BICUBIC);
    image.saveBMP("../data/output.bmp");
}
```

<figure class="image" style="width: 75%; margin-left: auto; margin-right: auto">
<img src="/imgs/draw-with-cpp-6/10.png">
</figure>

石头也变得晕乎乎了。

对上面的过程简单包装即可作为一个直接在图片中插入文字的方法，这里就不再赘述，具体实现可以查看代码中的 `addText()`、`addTitle()`。

## 代码

本节代码请查看：
[🔗Github: JeffreyXiang/DrawWithCpp](https://github.com/JeffreyXiang/DrawWithCpp/tree/master/src/6_font)

## 总结

在本节中，我们研究了点阵字库的结构和绘制方法，并由此引申了图片的变换与插入，最终实现了在图中任意位置，任意旋转，任意尺寸的文字的插入。

## 预告

下一节中，我们将迎来一个实用化的绘图应用例子——函数图像绘制。
