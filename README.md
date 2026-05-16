# 视频创意 AI 评价平台

输入关键词，自动搜索视频，调用 AI 对每条视频进行广告创意评审，输出评分和推荐理由。

> **免责声明**：本平台仅用于技术演示和个人研究，严禁商业使用。数据来源于网络，分析由 AI 算法生成，仅供参考。

## 快速启动

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 API Key（见下方说明）
# 在项目根目录创建 .env 文件

# 3. 启动
python app.py
# 访问 http://localhost:5001
```

## 接入 AI 评价

在项目根目录 `.env` 文件中配置 API Key，支持 DeepSeek 和 Anthropic（Claude）：

```bash
# 方式一：DeepSeek（推荐，性价比高）
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxx

# 方式二：Anthropic Claude
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxx
```

**优先级**：DeepSeek > Anthropic > 本地模板（不接 API 时自动降级为模板生成）

### 切换 AI

- 用 DeepSeek → `.env` 只保留 `DEEPSEEK_API_KEY`
- 用 Claude → `.env` 只保留 `ANTHROPIC_API_KEY`
- 两个都配 → 走 DeepSeek
- 都不配 → 使用本地模板生成摘要，不调用 AI

配置后重启 `python app.py` 即可，启动时会打印当前使用的 AI 服务。

## 评价维度

每条视频从 5 个维度评分（0-10 分），加权计算综合分：

| 维度 | 权重 | 说明 |
|------|------|------|
| 创意表现力 | 25% | 内容深度、标签丰富度、编辑推荐 |
| 制作水准 | 20% | 画质、平台评分 |
| 行业匹配度 | 20% | 分类/标签/标题/文案与目标行业重合度 |
| 节奏控制 | 15% | 时长是否在广告黄金区间（30s-3min） |
| 互动表现 | 20% | 点赞率、收藏率、分享率 |

AI 额外生成：**一句话推荐理由**（≤20字）+ **关键元素标签**（3-5个）。

## 命令行抓取

```bash
# 获取单个视频详情
python scrape_video.py https://www.xinpianchang.com/a13266271
python scrape_video.py a13266271 --json          # 输出原始 JSON
python scrape_video.py a13266271 --save out.json  # 保存到文件
```

## 接口说明

视频数据通过第三方 API 获取（无需认证），包括搜索、详情、评论接口。

## 依赖

- Python 3.9+
- Flask 3.0+
- OpenAI SDK（DeepSeek 兼容）
- Anthropic SDK（可选）
