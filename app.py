# -*- coding: utf-8 -*-
"""
新片场视频评价平台 — Flask Web 应用
"""
import json, time, os, sys
from flask import Flask, request, jsonify, render_template

# Fix Windows console encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except:
        pass

# 强制从 .env 加载环境变量（在任何导入之前）
_ENV_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(_ENV_FILE):
    with open(_ENV_FILE, "r", encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                _k, _v = _k.strip(), _v.strip()
                if _k and _v:
                    os.environ[_k] = _v  # 强制覆盖
                    print(f"[env] Loaded {_k}", flush=True)

from evaluate import (
    evaluate_video,
    build_evaluation_prompt,
    evaluate_batch_with_llm,
    generate_local_summary,
)
from scrape_video import get_article, format_detail
import requests

app = Flask(__name__)

SEARCH_API = "https://apis.netstart.cn/xpc/search"
DETAIL_API = "https://apis.netstart.cn/xpc/article"
CACHE = {}
CACHE_TTL = 600  # 10 min，对齐上游缓存

HEADERS = {"User-Agent": "Mozilla/5.0"}


def search_videos(keyword: str, page: int = 1) -> list:
    """搜索视频，返回list"""
    cache_key = f"search:{keyword}:{page}"
    now = time.time()
    if cache_key in CACHE and CACHE[cache_key]["ts"] > now - CACHE_TTL:
        return CACHE[cache_key]["data"]

    params = {"type": "article", "kw": keyword, "sort": "hot", "page": page}
    resp = requests.get(SEARCH_API, params=params, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    body = resp.json()
    if body.get("status") != 0:
        raise Exception(f"Search API error: {body.get('message')}")

    data = body.get("data", {})
    items = data.get("list", [])
    CACHE[cache_key] = {"ts": now, "data": items}
    return items


def get_video_detail(article_id: int) -> dict:
    """获取单个视频详情（带缓存）"""
    cache_key = f"detail:{article_id}"
    now = time.time()
    if cache_key in CACHE and CACHE[cache_key]["ts"] > now - CACHE_TTL:
        return CACHE[cache_key]["data"]

    try:
        detail = get_article(article_id, from_pc=True)
    except:
        detail = get_article(article_id, from_pc=False)
    CACHE[cache_key] = {"ts": now, "data": detail}
    return detail


def normalize_video(v: dict) -> dict:
    """将搜索返回的视频项标准化为评价所需格式"""
    # 搜索返回的数据已包含大部分字段，但 content 可能不完整
    # 对于 summary 生成，搜索返回的数据通常足够
    return v


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/search")
def api_search():
    keyword = request.args.get("kw", "").strip()
    industry = request.args.get("industry", "").strip()
    style = request.args.get("style", "").strip()
    page = request.args.get("page", 1, type=int)

    if not keyword:
        return jsonify({"error": "请输入搜索关键词"}), 400

    try:
        items = search_videos(keyword, page)
    except Exception as e:
        return jsonify({"error": f"搜索失败: {str(e)}"}), 500

    if not items:
        return jsonify({"error": f"未找到与「{keyword}」相关的视频", "videos": []})

    # 取前20条
    videos = items[:20]

    # 1. 算法评分（快速，所有视频）
    scores_list = [evaluate_video(v, industry, style) for v in videos]

    # 2. LLM 批量生成 summary + key_elements
    llm_results = evaluate_batch_with_llm(videos, industry, style)

    # 3. 组装结果
    results = []
    for i, v in enumerate(videos):
        scores = scores_list[i]
        if llm_results and i < len(llm_results):
            summary = llm_results[i].get("summary", "")
            key_elements = llm_results[i].get("key_elements", [])
        else:
            summary, key_elements = generate_local_summary(v, industry)

        results.append({
            "id": v.get("id"),
            "title": v.get("title", ""),
            "cover": v.get("cover", ""),
            "duration": v.get("duration", 0),
            "web_url": v.get("web_url", ""),
            "categories": [
                {
                    "main": c.get("category_name", ""),
                    "sub": c.get("sub", {}).get("category_name", ""),
                }
                for c in v.get("categories", [])
            ],
            "tags": [t.get("name", "") for t in v.get("tags", [])],
            "author": (
                v.get("author", {}).get("userinfo", {}).get("username", "")
                if isinstance(v.get("author"), dict)
                else ""
            ),
            "stats": {
                "views": v.get("count", {}).get("count_view", 0),
                "likes": v.get("count", {}).get("count_like", 0),
                "collects": v.get("count", {}).get("count_collect", 0),
                "shares": v.get("count", {}).get("count_share", 0),
                "comments": v.get("count", {}).get("count_comment", 0),
            },
            "scores": scores,
            "summary": summary,
            "key_elements": key_elements,
        })

    total = len(items)
    return jsonify({
        "keyword": keyword,
        "industry": industry,
        "style": style,
        "total": total,
        "page": page,
        "count": len(results),
        "llm_enabled": llm_results is not None,
        "videos": results,
    })


@app.route("/api/evaluate")
def api_evaluate():
    """获取单条视频的算法评分 + 本地summary（快速模式）"""
    article_id = request.args.get("id", "").strip()
    industry = request.args.get("industry", "").strip()
    style = request.args.get("style", "").strip()

    if not article_id:
        return jsonify({"error": "请提供视频ID"}), 400

    try:
        detail = get_video_detail(int(article_id))
    except Exception as e:
        return jsonify({"error": f"获取视频失败: {str(e)}"}), 500

    scores = evaluate_video(detail, industry, style)
    summary, key_elements = generate_local_summary(detail, industry)

    return jsonify({
        "id": detail.get("id"),
        "title": detail.get("title"),
        "scores": scores,
        "summary": summary,
        "key_elements": key_elements,
    })


if __name__ == "__main__":
    from evaluate import _get_llm_client
    provider, _ = _get_llm_client()
    print("=" * 50)
    print("新片场视频评价平台")
    if provider:
        print(f"LLM 评价: 已接入 ({provider})")
    else:
        print("LLM 评价: 未配置 — 使用本地摘要")
        print("  设置 DEEPSEEK_API_KEY 或 ANTHROPIC_API_KEY 以启用 LLM 评价")
    print("访问 http://localhost:5001")
    print("=" * 50)
    app.run(debug=True, use_reloader=False, host="0.0.0.0", port=5001)
