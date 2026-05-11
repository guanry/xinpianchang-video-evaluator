# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

新片场视频评价平台 — 用户输入关键词，自动搜索新片场视频，调用 LLM 对每条视频进行广告创意评审，返回评分和推荐理由。

## Commands

```bash
# 抓取单个视频详情
python scrape_video.py <url或id>
python scrape_video.py <url或id> --json          # 输出原始JSON
python scrape_video.py <url或id> --save out.json  # 保存到文件

# 启动Web服务
python app.py
# 访问 http://localhost:5000
```

## Architecture

```
scrape_video.py          # 视频抓取模块（CLI + library）
  ├── extract_article_id()   URL解析 → 数字ID
  ├── get_article()          调 apis.netstart.cn/xpc/article/:id
  ├── get_comments()         评论接口
  └── format_detail()        原始API数据 → 可读摘要字典

app.py                   # Flask Web应用（待构建）
  ├── /                    前端搜索页
  └── /api/search?kw=xxx  搜索+评价接口

templates/
  └── index.html          前端页面
```

## Key API: apis.netstart.cn/xpc

文档: https://apis.netstart.cn/xpc/

核心接口：
- `GET /article/:id` — 视频详情（基础数据 + 内容文案 + 制作团队）
- `GET /article/:id?from=pc` — PC版详情（content字段可能更完整）
- `GET /search?kw=&type=article&sort=hot&page=` — 搜索，返回 `data.list`，每页40条
- `GET /articles?category_id=` — 分类列表
- `GET /comments?resource_id=&type=article&page=` — 评论

全站缓存10分钟。无需认证。返回结构：`{"status": 0, "data": {...}}`，status=0 表示成功。

搜索返回的 list 项与 article/:id 返回结构一致（包含 id, title, duration, count, categories, author, content, tags 等）。

## 评价模型说明

当前评价由 LLM 实时生成，基于以下维度（0-10分）：
- creativity: 创意表现力（概念、文案、叙事角度）
- production_quality: 制作水准（拍摄、特效、调色、声音）
- industry_match: 行业匹配度（内容与目标行业/品类的贴合度）
- pacing: 节奏控制（时长利用效率、信息密度）
- overall: 综合推荐分
- summary: 一句话推荐理由（中文 ≤20字）
- key_elements: 关键元素标签（如"快剪""暖色调""故事型"）

搜索/评价时需传入 `industry` 和 `style_preference` 参数以获得定向评价。
