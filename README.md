# 剥落照片分析 · Spallation Image Analysis

从宏观腐蚀照片中识别坩埚、提取碎屑与表面形貌特征，生成样品对照图和相对剥落评分。支持自选图片目录、多种图片格式和任意数量的整数小时观测时间点。

## 安装

建议使用 Python 3.11 或更高版本，在项目目录中执行：

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

macOS / Linux 激活环境时使用 `source .venv/bin/activate`。

## 准备图片

将图片放在任意文件夹，按 `整数小时数h.扩展名` 命名，例如：

```text
experiment_photos/
  0h.jpg
  5h.png
  25h.tif
  50h.jpeg
```

- 有多少时间点就分析多少，只有一个时间点也可以运行，无须凑齐 5、25、50、100 h。
- 支持 PNG、JPG、JPEG、TIF、TIFF、BMP、WebP，扩展名及 `h` 不区分大小写。
- 只读取指定文件夹的直接子文件，不递归搜索。
- 同一时间存在多种格式时，按 PNG → JPG → JPEG → TIF → TIFF → BMP → WebP 的顺序选取一张。
- TIFF 按单张图像读取，多页 TIFF 不逐页处理。

## 运行

识别坩埚并生成编号预览：

```powershell
python detect_cups.py --input-dir "D:\experiment_photos"
```

提取特征、生成样品对照图和汇总评分：

```powershell
python macro_spallation.py --input-dir "D:\experiment_photos"
```

不指定 `--input-dir` 时读取脚本所在文件夹。参数支持绝对路径、相对路径以及含中文或空格的路径。

## 输出

所有输出写入脚本目录下的 `analysis_outputs/`：

| 文件 | 内容 |
| --- | --- |
| `cup_detection/*_thumbnail.jpg` | 输入图片缩略图 |
| `cup_detection/*_detected.jpg` | 坩埚检测和样品编号预览 |
| `figures/macro_*h_contact_sheet.jpg` | 各时间点样品及碎屑标记对照图 |
| `macro_image_features.csv` | 每个样品、每个时间点的形貌特征和相对评分 |
| `spallation_proxy.csv` | 综合评分、最后观测时间及对应评分 |

检测预览由 `detect_cups.py` 生成，CSV 与样品对照图由 `macro_spallation.py` 生成。同名输出再次运行时会覆盖；本次未生成的旧文件仍保留。

## 评分和样品编号

每个时间点以碎屑面积、碎裂比例、碎屑数量、离散程度、中心外材料比例和边缘比例计算样品间的百分位评分。各项权重依次为 0.30、0.20、0.15、0.15、0.10、0.10。

按已发现时间点从早到晚赋予 1、2、…、N 的权重，各样品使用自身实际观测到的权重之和归一化，再对综合分进行样品间百分位排名。更晚时间点具有更高权重，0 h 图片如存在也参与分析。

- `spallation_proxy`：综合相对评分，越高表示图像中的碎屑与分散特征越明显。
- `latest_time_h`：该样品最后一次观测的小时数。
- `spallation_latest_proxy`：该样品最后观测时间点的相对评分。
- `spallation_100h_proxy`：100 h 相对评分；没有对应观测时留空。

评分是同一批图片内的图像特征相对指标，并非剥落质量、剥落面积百分比或绝对失效量。更换样品集合或时间点集合可能改变排名。

默认布局为左右两个 4×4 托盘，左侧样品 1–16，右侧样品 17–32，均按行编号。单托盘模式默认编号 17–32；使用其他布局前需调整 `index_circles()` 和 `grid_from_candidates()`。请先查看检测编号预览，确认编号与实验记录一致。圆检测可能根据网格补齐位置，预览可用于核对。

## 项目文件

- `detect_cups.py`：图片发现、读写、坩埚检测和编号。
- `macro_spallation.py`：特征提取、评分与对照图输出。
- `requirements.txt`：运行依赖。

仓库和源码压缩包包含代码与说明，原始实验照片、分析输出、虚拟环境和编辑器配置由 `.gitignore` 排除。
