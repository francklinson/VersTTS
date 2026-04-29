# GPT-SoVITS Git 管理指南

## 项目概况

| 项目 | 大小 | 说明 |
|------|------|------|
| 总大小 | ~22GB | 包含所有模型文件 |
| 核心代码 | ~50MB | 不包括模型文件 |
| 模型文件 | ~4.6GB | 预训练权重 |

---

## 一、必须加入 Git 的核心代码文件

### 1. 核心配置文件
```
GPT_SoVITS/
├── configs/
│   ├── s2.json              # SoVITS V2配置
│   ├── s2v2Pro.json         # SoVITS V2Pro配置
│   ├── s2v2ProPlus.json     # SoVITS V2ProPlus配置
│   └── tts_infer.yaml       # TTS推理配置 (已修改路径)
└── config.py                # 主配置文件
```

### 2. 核心源代码

#### 2.1 TTS推理模块
```
GPT_SoVITS/
├── TTS_infer_pack/
│   ├── TTS.py               # 主TTS推理类 (已修改路径)
│   └── text_segmentation_method.py  # 文本切分方法
```

#### 2.2 AR模块 (GPT模型)
```
GPT_SoVITS/AR/
├── modules/
│   ├── __init__.py
│   ├── activation.py
│   ├── embedding.py
│   ├── optim.py
│   ├── scaling.py
│   └── transformer.py
└── utils/
    ├── __init__.py
    ├── initialize.py
    └── io.py
```

#### 2.3 文本处理模块
```
GPT_SoVITS/text/
├── __init__.py
├── chinese.py               # 中文处理
├── chinese2.py
├── english.py               # 英文处理
├── japanese.py              # 日文处理
├── korean.py                # 韩文处理
├── cantonese.py             # 粤语处理
├── cleaner.py
├── symbols.py               # 音素符号定义
├── tone_sandhi.py
├── opencpop-strict.txt
├── cmudict*.rep
├── engdict*.rep
└── symbols2.py
```

#### 2.4 特征提取模块
```
GPT_SoVITS/feature_extractor/
├── __init__.py
├── cnhubert.py              # CNHuBERT特征提取
└── whisper_enc.py           # Whisper编码器
```

#### 2.5 模型模块
```
GPT_SoVITS/module/
├── __init__.py
├── models.py                # 主模型定义
├── attentions.py            # 注意力机制
├── attentions_onnx.py
├── commons.py
├── core_vq.py
├── data_utils.py
├── ddp_utils.py
├── distrib.py
├── losses.py
├── mel_processing.py
├── models_onnx.py
├── modules.py
├── mrte_model.py
├── quantize.py
└── transforms.py
```

#### 2.6 BigVGAN声码器
```
GPT_SoVITS/BigVGAN/
├── __init__.py
├── bigvgan.py
├── activations.py
├── inference.py
├── inference_e2e.py
├── env.py
├── loss.py
├── meldataset.py
├── utils0.py
└── discriminators.py
```

#### 2.7 ERes2Net说话人编码
```
GPT_SoVITS/eres2net/
├── __init__.py
├── ERes2Net.py
├── ERes2NetV2.py
├── ERes2Net_huge.py
├── fusion.py
└── kaldi.py
```

### 3. 工具脚本
```
GPT_SoVITS/
├── utils.py                 # 主工具函数
├── inference_cli.py         # CLI推理脚本
├── inference_webui.py       # WebUI推理
├── inference_webui_fast.py  # 快速WebUI
├── process_ckpt.py          # 检查点处理
├── export_torch_script.py   # TorchScript导出
└── onnx_export.py           # ONNX导出
```

### 4. 训练脚本
```
GPT_SoVITS/
├── s1_train.py              # GPT模型训练
├── s2_train.py              # SoVITS V2训练
├── s2_train_v3.py           # SoVITS V3训练
└── stream_v2pro.py          # V2Pro流式处理
```

### 5. 其他核心文件
```
GPT_SoVITS/
├── G2PWModel/
│   ├── config.py            # G2PW配置
│   └── version              # 版本文件
└── text/g2pw/
    ├── __init__.py
    ├── dataset.py
    ├── g2pw.py
    ├── onnx_api.py
    └── utils.py
```

---

## 二、必须加入 Git 的项目级文件

### 1. 根目录配置文件
```
├── api.py                   # API接口 (原版)
├── api_v2.py                # API V2接口
├── webui.py                 # WebUI入口
├── config.py                # 全局配置
├── weight.json              # 权重配置
├── readme.md                # 项目说明
├── requirements.txt         # 依赖列表
├── download_models.py       # 模型下载脚本 (我们创建的)
├── .gitignore               # Git忽略配置
└── GIT_MANAGEMENT_GUIDE.md  # 本文件
```

### 2. 工具目录
```
tools/
├── __init__.py
├── i18n/                    # 国际化
│   ├── i18n.py
│   └── locale/              # 翻译文件
├── slice_audio.py           # 音频切片
├── slicer2.py
├── subfix_webui.py
├── audio_sr.py              # 音频超分
├── my_utils.py
└── assets.py
```

### 3. 文档
```
docs/
├── cn/
│   ├── README.md
│   └── Changelog_CN.md
├── en/
│   └── Changelog_EN.md
├── ja/
│   └── README.md
└── ko/
    └── README.md
```

---

## 三、**不加入 Git** 的文件 (通过脚本下载)

### 1. 预训练模型 (共 ~4.6GB)
```
GPT_SoVITS/pretrained_models/
├── chinese-roberta-wwm-ext-large/     # BERT模型 (~622MB)
├── chinese-hubert-base/               # HuBERT模型 (~181MB)
├── gsv-v2final-pretrained/            # V2模型 (~260MB)
│   ├── s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt
│   └── s2G2333k.pth
├── s1v3.ckpt                          # V3/V4 GPT模型 (~149MB)
├── s2Gv3.pth                          # V3 SoVITS (~734MB)
├── gsv-v4-pretrained/                 # V4模型 (~790MB)
│   ├── s2Gv4.pth
│   └── vocoder.pth
├── v2Pro/                             # V2Pro模型 (~600MB)
│   ├── s2Gv2Pro.pth
│   ├── s2Gv2ProPlus.pth
│   ├── s2Dv2Pro.pth
│   └── s2Dv2ProPlus.pth
├── s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt  # V1 GPT
├── s2G488k.pth / s2D488k.pth          # V1 SoVITS
└── ... 其他模型文件
```

### 2. G2PW模型
```
GPT_SoVITS/text/G2PWModel/
└── g2pW.onnx              # (~606MB)

text/G2PWModel_1.1.zip    # (~562MB)
```

### 3. 缓存和临时文件
```
__pycache__/
*.pyc
.cache/
*.log
.DS_Store
```

### 4. 输出文件
```
outputs/
*.wav
*.mp3
uploads/
```

### 5. 旧的/重复的文件
```
text/XXXXRT/               # 旧的临时目录
text/G2PWModel*.zip        # ZIP压缩包
GPT_SoVITS/G2PWModel_1.1.zip
GPT_SoVITS/G2PWModel.zip
```

---

## 四、文件分类统计

| 类别 | 数量 | 大小 | Git |
|------|------|------|-----|
| Python源码文件 | ~80个 | ~15MB | ✅ 必须 |
| 配置文件 | ~20个 | ~1MB | ✅ 必须 |
| 文档文件 | ~10个 | ~2MB | ✅ 必须 |
| 预训练模型 | ~20个 | ~4.6GB | ❌ 排除 |
| 缓存/临时文件 | ~110个 | ~100MB | ❌ 排除 |
| ZIP/压缩包 | ~5个 | ~1.5GB | ❌ 排除 |

---

## 五、推荐的 Git 工作流

### 1. 初始化仓库
```bash
cd /home/zhouchenghao/PycharmProjects/VersTTS/algorithms/GPT-SoVITS

# 创建.gitignore (已创建)
cat > .gitignore << 'EOF'
# 模型文件
*.pth
*.ckpt
*.pt
*.onnx
*.bin
pretrained_models/
models/

# 压缩包
*.zip
*.tar.gz

# 缓存
__pycache__/
*.pyc
.cache/

# 输出
outputs/
*.wav

# 临时目录
text/XXXXRT/
text/G2PWModel/
EOF

# 初始化Git
git init
git add .gitignore
git commit -m "Initial commit: Add .gitignore"
```

### 2. 添加核心代码
```bash
# 添加核心代码
git add GPT_SoVITS/*.py
git add GPT_SoVITS/*/*.py
git add GPT_SoVITS/configs/
git add GPT_SoVITS/text/*.py GPT_SoVITS/text/*.txt GPT_SoVITS/text/*.rep
git add tools/
git add docs/
git add *.py *.md *.txt *.json

git commit -m "Add core GPT-SoVITS source code"
```

### 3. 创建模型下载脚本
```bash
# 添加模型下载脚本
git add download_models.py
git commit -m "Add model download script"
```

---

## 六、模型下载说明

对于新克隆的仓库，需要运行以下命令下载模型：

```bash
# 安装依赖
pip install -r requirements.txt

# 下载模型
python download_models.py
```

模型将下载到：
- `GPT_SoVITS/pretrained_models/` - 预训练模型
- `GPT_SoVITS/text/G2PWModel/` - G2PW模型

---

## 七、已创建的 .gitignore 文件

已创建 `.gitignore` 文件，包含以下规则：

1. **模型文件**：*.pth, *.ckpt, *.pt, *.onnx, *.bin
2. **模型目录**：pretrained_models/, models/
3. **G2PW模型**：text/G2PWModel/, text/XXXXRT/
4. **压缩包**：*.zip, *.tar.gz
5. **Python缓存**：__pycache__/, *.pyc
6. **日志输出**：logs/, outputs/, *.wav
7. **IDE文件**：.vscode/, .idea/
8. **Notebook检查点**：.ipynb_checkpoints/

---

## 八、注意事项

1. **不要提交模型文件**：模型文件体积大，应通过脚本下载
2. **保留配置文件**：`tts_infer.yaml` 的路径已修改，需要提交
3. **保留下载脚本**：`download_models.py` 用于自动化下载
4. **忽略临时文件**：所有缓存、日志、输出文件都应忽略

---

**最后更新**: 2026-04-29
