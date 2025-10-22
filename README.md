# Vytics

一个基于AI的视频理解和分析工具，能够提取视频中的音频、转换为文本，并进行智能分析。

## 功能特性

- 🎥 视频文件处理和音频提取
- 🎵 音频转文字（支持多种语言）
- 🤖 AI驱动的内容分析
- 📊 生成详细的分析报告
- 💾 结果保存为JSON格式

## 安装要求

### 系统依赖
- Python 3.7+
- FFmpeg（用于音频提取）

### Python依赖
```bash
pip install -r requirements.txt
```

## 配置

1. 复制 `.env` 文件并配置你的API密钥：
```bash
cp .env.example .env
```

2. 编辑 `.env` 文件，添加必要的API密钥：
```
OPENAI_API_KEY=your_openai_api_key_here
# 其他必要的配置...
```

## 使用方法

### 基本用法

```python
from video_analyzer import VideoAnalyzer

# 创建分析器实例
analyzer = VideoAnalyzer()

# 分析视频文件
result = analyzer.analyze_video("path/to/your/video.mp4")

# 查看结果
print(result)
```

### 命令行使用

```bash
python video_analyzer.py --input "video_file.mp4" --output "analysis_result.json"
```

## 支持的视频格式

- MP4
- AVI
- MOV
- FLV
- MKV
- WMV

## 输出格式

分析结果将保存为JSON格式，包含以下信息：
- 视频基本信息
- 音频转录文本
- AI分析结果
- 时间戳信息

## 注意事项

- 确保视频文件路径正确
- 大文件处理可能需要较长时间
- 需要稳定的网络连接（用于AI API调用）

## 许可证

MIT License

## 贡献

欢迎提交Issue和Pull Request！

## 更新日志

### v1.0.0
- 初始版本发布
- 基本视频分析功能
- 音频转文字功能
- AI内容分析