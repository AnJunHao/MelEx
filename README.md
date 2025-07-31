# MelEx - 旋律提取

MelEx 是一个用于从钢琴演奏MIDI文件中进行旋律提取的Python库。它将参考旋律与钢琴转录对齐，从复杂的钢琴演奏中提取原始旋律。

## 介绍

MelEx 使用先进的对齐算法从钢琴演奏MIDI文件中识别和提取旋律。给定参考旋律和钢琴转录，它能产生：

- 包含提取旋律的MIDI文件
- 评估指标和统计数据
- 对齐过程的可视化图表

该库专为从事音乐数据分析、旋律提取和MIDI处理的研究人员和开发者设计。

## 安装

### 前置要求

- Python 3.13 或更高版本
- Git

### 从源码安装

1. 克隆仓库：
```bash
git clone <repository-url>
cd exmel
```

2. 使用pip安装：
```bash
pip install .
```

## 基本用法

### 数据集组织

MelEx 期望数据集按特定的文件夹结构组织。每首歌曲应该有自己的目录，包含所需的MIDI文件：

```
数据集根目录/
├── 歌曲名或ID/
│   ├── 歌曲名或ID.m.mid    # 必需：参考旋律
│   ├── 歌曲名或ID.t.mid    # 必需：钢琴转录
│   └── 歌曲名或ID.mp3      # 可选：音频文件
├── 下一首歌/
│   ├── 下一首歌.m.mid
│   ├── 下一首歌.t.mid
│   └── ...
└── ...
```

**必需文件：**
- `<歌曲名或ID>.m.mid` - 参考旋律文件
- `<歌曲名或ID>.t.mid` - 钢琴转录/演奏文件

**可选文件：**
- `<歌曲名或ID>.mp3` 或 `<歌曲名或ID>.opus` - 音频文件（用于时长匹配）

### 简单推理

使用MelEx的最简单方式是通过 `inference_pipeline` 接口：

```python
from melex import inference_pipeline

# 处理整个数据集
melodies, evaluation_df = inference_pipeline("path/to/dataset")
```

这将：
1. 处理数据集目录中的所有歌曲
2. 在 `outputs/` 中创建带时间戳的输出目录
3. 自动保存结果

### 推理管道输出

`inference_pipeline` 函数产生多种输出：

**返回值：**
- `melodies`：将歌曲名映射到提取旋律对象的字典
- `evaluation_df`：包含每首歌曲评估指标的Pandas DataFrame

**保存的文件（在输出目录中）：**
- `{歌曲名或ID}.mid` - 每首歌曲提取的旋律MIDI文件
- `{歌曲名或ID}_align.png` - 每首歌曲的对齐可视化图表
- `evaluation.xlsx` - 包含所有歌曲评估指标的Excel文件

**输出目录结构：**
```
outputs/20240101_120000/  # 带时间戳的目录
├── 歌曲名或ID.mid
├── 歌曲名或ID_align.png
├── 下一首歌.mid  
├── 下一首歌_align.png
├── ...
└── evaluation.xlsx
```

### 自定义输出选项

```python
from melex import inference_pipeline

melodies, evaluation_df = inference_pipeline(
    "path/to/dataset",
    save_dir="输出",     # 自定义输出目录
    save_midi=True,            # 保存MIDI文件（默认：True）
    save_excel=True,           # 保存评估Excel（默认：True）
    save_plot=True,            # 保存对齐图表（默认：True）
    verbose=1,                 # 进度详细程度（0-2）
    n_jobs=4                   # 并行处理（0=自动）
)
```

## 高级用法

### 数据集切片

`Dataset` 类支持灵活的切片来处理数据子集：

```python
from melex import Dataset, inference_pipeline

# 加载数据集
dataset = Dataset("数据集文件夹")

# 按索引切片（前5首歌）
subset = dataset[:5]
melodies, evaluation_df = inference_pipeline(subset)

# 按歌曲名切片
songs_of_interest = dataset[["暗号(周杰伦)", "平凡之路(朴树)", "无限(周深)"]]
melodies, evaluation_df = inference_pipeline(songs_of_interest)

# 按名称获取单首歌曲
single_song = dataset["无限(周深)"]
melodies, evaluation_df = inference_pipeline([single_song])

# 按索引获取单首歌曲
first_song = dataset[0]
```

### 配置推理精度

您可以使用 `hop_length` 参数控制速度/精度权衡：
- `hop_length=1`：最精确的对齐（默认，最慢）
- `hop_length=n`：大约快n倍，但精度较低
- 更高的值以准确性换取速度

```python
from melex import inference_pipeline, get_default_config

# 最精确（最慢）
precise_config = get_default_config(hop_length=1)
melodies, evaluation_df = inference_pipeline(
    "path/to/dataset", 
    config=precise_config
)

# 更快但精度较低
fast_config = get_default_config(hop_length=4)  # 约快4倍
melodies, evaluation_df = inference_pipeline(
    "path/to/dataset",
    config=fast_config
)
```

### 完整高级示例

```python
from melex import Dataset, inference_pipeline, get_default_config

# 加载和探索数据集
dataset = Dataset("数据集文件夹")
print(f"数据集包含 {len(dataset)} 首歌曲")

# 使用自定义配置处理子集
selected_songs = dataset[["暗号(周杰伦)", "平凡之路(朴树)", "无限(周深)"]]
config = get_default_config(hop_length=2)  # 约快2倍

melodies, evaluation_df = inference_pipeline(
    selected_songs,
    config=config,
    save_dir="输出/实验1",
    save_midi=True,
    save_excel=True, 
    save_plot=True,
    verbose=1,
    n_jobs=4  # 使用4个并行进程
)

# 检查结果
print("评估指标：")
print(evaluation_df.head())

print(f"提取了 {len(melodies)} 个旋律")
for song_name, melody in melodies.items():
    print(f"  {song_name}: {len(melody)} 个音符")
```