# CLAUDE.md — jeffreyxiang.github.io 开发指南

本文件面向后续在本仓库工作的 AI agent / 开发者，描述项目架构、内容模型、约定和常见任务的操作步骤。

## 1. 项目概览

- **性质**：Jianfeng Xiang（向剑锋）的个人学术主页，部署在 GitHub Pages：`https://jeffreyxiang.github.io/`。
- **技术栈**：Jekyll（通过 `github-pages` gem，由 GitHub Pages 构建），Liquid 模板，kramdown（GFM 输入）+ rouge 高亮，纯手写 CSS，jQuery 3.6.1，MathJax 3（CDN），giscus 评论。
- **没有**：npm/Node 构建、Sass（`_sass` 目录不存在）、Jekyll 插件（`plugins: []`）、`_data` 目录、测试、CI 配置。push 到 `main` 即由 GitHub Pages 自动构建发布。
- 页面主要语言为英文；博客分中文 / 英文版本。

## 2. 目录结构

```
_config.yml              站点配置：集合定义 + 出版物/应用的排序列表（重要）
Gemfile                  github-pages + tzinfo
index.html               首页（个人信息卡片 + 简介 + 链接）
publications.html        出版物列表（遍历 site.publication_order）
blogs.html               博客列表（按 blog_id 分组，jQuery 两列瀑布流）
applications.html        应用列表（遍历 site.application_order）
projects.html            占位页，只有标题，无内容
resume.html              独立的完整 HTML 简历，无 front matter，不走 layout
_layouts/
  default.html           通用布局：topbar + #main + bottombar
  blog.html              博客布局：左侧 TOC + 正文 + 语言切换 + MathJax + giscus
_includes/
  topbar.html            顶部导航（Home/Publications/Blogs/Projects/Applications）
  bottombar.html         页脚（版权 + 社交图标）
  toc.html               allejo/jekyll-toc v1.2.1，纯 Liquid 生成目录（勿修改）
_publications/*.md       出版物数据（只有 front matter，output: false）
_blogs/*.md              博客文章（output: true，靠 permalink 定 URL）
_applications/*.md       应用卡片数据（只有 front matter，output: false）
applications/conference-countdown/
                         会议倒计时小应用：main.html(带 front matter) + main.js + main.css + dates.json
css/
  default.css            全站主样式：首页、出版物、博客列表、应用列表
  topbar.css / bottombar.css
  sharp.css              "锐利"风格覆盖（灰色标题、2px 实心阴影），仅 blog 布局引用
  article.css            博客正文样式（rouge 高亮配色、figure、table、post-header 等）
  toc.css                博客侧栏目录样式
js/
  jquery-3.6.1.min.js
  scroll_spy.js          博客 TOC 滚动高亮
fonts/                   Avenir / Avenir Next / Config 字体（站点实际只加载 avenir-next）
icons/                   IcoMoon 图标字体（style.css 提供 icon-logo-* 类）+ 若干 SVG
imgs/<blog_id>/          每篇博客的图片目录（teaser + 正文配图）
videos/<name>.mp4        出版物 teaser 视频
misc/                    logo/favicon 的 .xcf 源文件，不被页面引用
```

## 3. 内容模型（集合）

`_config.yml` 中定义了 4 个集合（`collections_dir: .`，所以集合目录直接在根目录）：

| 集合 | output | 用途 |
|---|---|---|
| `blogs` | true | 博客文章，各自生成页面 |
| `projects` | true | 已声明但 **`_projects/` 目录不存在**，`projects.html` 为空壳 |
| `publications` | false | 仅作数据源，供 `publications.html` 渲染 |
| `applications` | false | 仅作数据源，供 `applications.html` 渲染 |

### 3.1 出版物 `_publications/<order_id>.md`

只有 front matter，没有正文：

```yaml
---
order_id: trellis.2          # 必须与 _config.yml 中 publication_order 的条目一致
title: "..."
authors: "<span class='me'>Jianfeng Xiang</span>, ..."   # 本人用 span.me 高亮
location: "2026 IEEE Conference on ..., CVPR 2026"
venue: "CVPR 2026"           # 可选，蓝色徽章
award: "Best Student Paper Award"  # 可选；含 "best" 且含 "paper"（不分大小写）→ 金色奖杯徽章，否则深色徽章
teaser: "/videos/trellis2.mp4"
teaser_type: "video"         # "video" | "image"
page_url: "https://..."      # 渲染为 "Link"
abstract: "..."
---
```

**展示顺序完全由 `_config.yml` 的 `publication_order` 决定**。新增出版物必须同时：
1. 新建 `_publications/<order_id>.md`；
2. 在 `publication_order` 中按想要的位置加入 `order_id`（最新的放最前）；
3. teaser 视频放到 `videos/`（图片可放 `imgs/`）。

只建文件不加到列表 → 不显示；列表里有但文件缺失 → 渲染出空卡片。

### 3.2 应用 `_applications/<order_id>.md`

```yaml
---
order_id: conference-countdown   # 对应 _config.yml 的 application_order
name: Conference<br>Countdown    # 可含 <br> 控制换行
icon: "/icons/icon-conference-countdown.svg"
link: "/applications/conference-countdown"   # 站内页面或外部 GitHub 链接
---
```

同样需要在 `application_order` 中注册。站内应用的实现放在 `applications/<name>/`，入口 HTML 带 front matter（`layout: default` + `permalink`）。

### 3.3 博客 `_blogs/YYYY-MM-DD-<slug>[.zh].md`

```yaml
---
layout: blog
title: "..."
excerpt: "..."               # 列表页 teaser 悬停时显示；不写则 Jekyll 取首段
series: "Learn Ray Tracing with Taichi"   # 可选，目前模板未使用
blog_id: taichi-raytracing-1 # 同一篇文章的中英文版本共享同一个 blog_id
permalink: /zh/blogs/taichi-raytracing-1  # 必须为 /<lang>/blogs/<blog_id>
teaser: /imgs/taichi-raytracing-1/teaser.jpg
lang: zh                     # "zh" | "en"
github: https://github.com/...   # 可选，显示 GitHub 图标
zhihu: https://zhuanlan.zhihu.com/p/...   # 可选，显示知乎图标
---
```

关键机制：
- **URL**：全靠 `permalink`，`/en/blogs/<blog_id>` 或 `/zh/blogs/<blog_id>`。布局里的语言切换就是把 URL 中的 `/en/` 和 `/zh/` 互相替换，因此两个语言版本的 permalink 必须严格对称。
- **双语**：同 `blog_id` 不同 `lang` 视为同一篇。列表页每个 `blog_id` 只显示一张卡片（优先 en），并给出可用语言链接。文件名惯例：英文无后缀，中文加 `.zh.md`（老的纯中文文章没有后缀也可以）。
- **日期**：取自文件名前缀；`future: false`，所以未来日期的文章不会发布。时区 `Asia/Shanghai`。
- **列表排序**：按日期倒序后再 `group_by: blog_id`。
- **评论**：giscus，`data-term="{{ page.blog_id }}"`，中英文版本共享同一个讨论串。
- **TOC**：自动取正文的 h2~h3。正文从 `##` 开始写，不要用 `#`（标题由布局输出）。
- **公式**：MathJax，行内 `$...$`，块级 `$$...$$`。
- **图片**：放在 `imgs/<blog_id>/`，正文里用绝对路径 `/imgs/...`。需要说明文字时直接写 HTML：
  ```html
  <figure>
    <img src="/imgs/flexgemm/xxx.png" alt="...">
    <figcaption>...</figcaption>
  </figure>
  ```
  并排图片用 `<figure class="image" style="width: 49%; display: inline-block">`。
  （个别旧文章用了 `https://jeffreyxiang.github.io/imgs/...` 的完整 URL，新内容建议用站内绝对路径。）
- 代码块用 GFM ``` 围栏，rouge 高亮配色在 `css/article.css` 顶部。

## 4. 布局与样式

- `default.html`：加载 avenir-next 字体、icons、topbar/bottombar/default.css；`sharp.css` 被注释掉。
- `blog.html`：额外加载 `sharp.css`、`article.css`、`toc.css`、jQuery、`scroll_spy.js`、MathJax；页面结构为 `#main > aside(TOC, 200px) + main(article)`。
- 视觉风格：浅灰卡片（`rgb(248,248,248)`）+ `2px 2px 0 rgb(224,224,224)` 实心阴影 + 灰色大标题（`.title`，64px/900）。新增组件请沿用这套风格。
- **响应式**：`#main` 宽度 `100%`，最大 1032px，左右各留 16px。两个断点：
  - `max-width: 1056px`：博客列表从双列瀑布流变成单列，博客页隐藏左侧 TOC，改为正文上方可折叠的 `<details class="toc-mobile">`。
  - `max-width: 768px`：手机布局。顶栏菜单收进汉堡按钮（纯 CSS，`#topbar_toggle` checkbox 控制），首页、论文、应用、倒计时页的卡片纵向堆叠，页脚也改为上下排列。
  - 新增组件时，请把媒体查询写在对应 CSS 文件末尾的 Responsive 区块。断点保持 1056 / 768，并在 390px 宽度下确认页面不会横向溢出。
- 博客列表的双列分配由 `blogs.html` 里的 jQuery 完成，宽度跨过 1056px 时会重新排列；单列时按日期倒序。
- MathJax 不会给**行内**公式自动换行，过长的 `$...$` 会在手机上撑宽页面，请拆短或改用 `$$...$$`（块级公式可以横向滚动）。
- `resume.html` 完全独立：自带 `<style>`（CSS 变量、Google Fonts Inter/JetBrains Mono），底部脚本调用 GitHub API 实时拉取 `data-repo="owner/repo"` 元素的 star 数并累加到 `#stat-stars`。导航栏中没有指向它的链接，访问地址为 `/resume.html`（GitHub Pages 也支持 `/resume`）。
- 页面中的标签和类名采用 `snake_case`（如 `publication_item_badge_gold`），博客正文样式沿用 minima 风格类名（`post-header`、`post-meta`）。

## 5. 本地开发

本机（macOS，无 Homebrew/rbenv/Docker）使用**系统自带 Ruby 2.6** 搭建了本地环境，全部文件都在已被 gitignore 的 `vendor/` 下：

```bash
vendor/local-dev/serve.sh          # jekyll serve --livereload → http://127.0.0.1:4000
vendor/local-dev/serve.sh build    # 只构建到 _site/，用来检查 Liquid 错误
```

- `vendor/local-dev/Gemfile`：本地专用，固定 `github-pages 227`（jekyll 3.9），并把 ffi、nokogiri、activesupport 等依赖锁到兼容 Ruby 2.6 的版本。**不要**用它替换根目录的 `Gemfile`，线上构建以根目录的为准。
- gem 安装在 `vendor/bundle/`。
- 脚本设置了 `SDKROOT=.../MacOSX26.5.sdk`：默认的 MacOSX27 SDK 与系统 Ruby 的头文件目录名对不上，链接器也不认识它的 `arm64e.x1` 架构，原生扩展会编译失败。
- 本地 WEBrick 对 `/applications` 这类同名目录会返回 301 跳转到 `/applications/`，这是正常现象。
- `_config.yml` 的 `exclude` 中包含 `vendor`、`CLAUDE.md`、`AGENTS.md`。如果去掉，github-pages 会把 CLAUDE.md 渲染成公开页面 `/CLAUDE.html`。
- `Gemfile.lock`、`_site/`、`.jekyll-cache/` 已被 gitignore。
- 没有 lint 或测试；修改后至少用 `serve.sh build` 确认 Liquid 不报错，再在浏览器里检查对应页面。

## 6. 常见任务速查

| 任务 | 需要修改的位置 |
|---|---|
| 新增论文 | `_publications/<id>.md` + `_config.yml: publication_order` + `videos/` 中的 teaser |
| 给论文加奖项/会议徽章 | 对应 md 中的 `award` / `venue` 字段；样式在 `css/default.css` 的 `.publication_item_badge*` |
| 新增博客 | `_blogs/YYYY-MM-DD-<slug>.md`（+ `.zh.md`），`imgs/<blog_id>/`，正确设置 `permalink`/`lang`/`blog_id` |
| 新增应用卡片 | `_applications/<id>.md` + `_config.yml: application_order` + `icons/icon-<id>.svg` |
| 更新会议倒计时 | `applications/conference-countdown/dates.json`（ISO 8601 时间，带时区偏移） |
| 改首页个人信息 | `index.html`（身份、邮箱、简介段落） |
| 改导航/页脚 | `_includes/topbar.html`、`_includes/bottombar.html` |
| 改简历 | `resume.html`（单文件，样式和脚本都在内部） |

## 7. 已知问题与注意事项

- `_config.yml` 改动后，`jekyll serve` 需要**重启**才会生效。
- `projects` 集合是空的，`projects.html` 只有标题。
- 会议倒计时的 `dates.json` 最新数据停在 2023 年。
- `toc.html` 是第三方库，不要修改。
- `misc/`、`fonts/avenir`、`fonts/config` 以及字体包自带的 demo 文件留在仓库中，但已通过 `exclude` 不再发布。注意 Jekyll 的 `exclude` 按**前缀**匹配：写成 `fonts/avenir` 会连 `fonts/avenir-next` 一起排除，所以要用 `fonts/avenir/*` 这样的写法。
- `default.html` / `blog.html` 已加上 `<!DOCTYPE html>`，页面处于标准模式。改 CSS 时不要依赖怪异模式（quirks mode）的行为。
- 版权年份用 `{{ site.time | date: "%Y" }}`，在构建时自动生成。
- 仓库中大文件较多（imgs ~44MB、fonts ~14MB、videos ~12MB）。teaser 视频建议压缩到 1–2MB。
- 首页和页脚使用的是个人邮箱，`resume.html` / `index.html` 中的工作邮箱可能和它不同。修改联系方式前请先确认要用哪个。

## 8. Git 约定

- 直接在 `main` 上提交，推送后即发布。
- 提交信息为英文小写开头的祈使句（如 `update publication details and add best paper award badge`），历史里也有大量 `update`。建议写清楚具体改了什么。
