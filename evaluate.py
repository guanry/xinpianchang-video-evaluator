# -*- coding: utf-8 -*-
"""
视频评价引擎：算法评分 + LLM 总结
"""
import json, math, re, sys
from typing import Optional


# ============================================================
# 算法评分
# ============================================================

def _safe_ratio(numerator, denominator, default=0.5):
    if denominator == 0:
        return default
    return min(1.0, numerator / denominator)


def score_creativity(video: dict) -> float:
    """创意表现力：荣誉背书 + 内容深度 + 标签稀缺度"""
    score = 5.0
    
    # 1. 荣誉背书 (核心加成)
    badge = video.get("badge", "")
    if badge == "recommend":
        score += 2.5  # 首页推荐
    elif badge == "category_recommend":
        score += 1.5  # 频道推荐
    elif badge == "hot":
        score += 1.0  # 热门
        
    # 2. 内容深度
    content = video.get("content", "") or ""
    if len(content) > 100:
        score += 1.0
    if len(content) > 300:
        score += 0.5
        
    # 3. 标签与荣誉关键字
    tags = [t.get("name", "") for t in video.get("tags", [])]
    honor_kws = ["获奖", "入围", "精选", "佳作", "原创"]
    for t in tags:
        if any(kw in t for kw in honor_kws):
            score += 0.5
            break
    if len(tags) >= 5:
        score += 0.5
        
    return min(10.0, score)


def score_production_quality(video: dict) -> float:
    """制作水准：创作者权威度 + 画质 + 平台评分"""
    score = 5.0
    
    # 1. 创作者权威度 (背书)
    author = video.get("author", {})
    if isinstance(author, dict):
        userinfo = author.get("userinfo", {})
        # VIP 等级加分 (最高1.5)
        vip_level = userinfo.get("vip_flag", 0)
        score += min(vip_level * 0.3, 1.5)
        # 认证加分
        if userinfo.get("verify_description"):
            score += 1.0
            
    # 2. 画质表现 (最高1.5)
    # quality 4: 4K, 3: 1080P, etc.
    quality = video.get("quality", 0)
    score += min(quality * 0.4, 1.5)
    
    # 3. 平台历史评分 (活跃度与认可度)
    c = video.get("count", {})
    xpc_score = c.get("score", 0)
    if xpc_score > 15000:
        score += 1.0
    elif xpc_score > 5000:
        score += 0.5
        
    return min(10.0, score)


# 行业关键词扩展词典 (保持不变)
_INDUSTRY_ALIAS = {
    "3c": ["手机", "数码", "科技", "电子", "智能", "硬件", "电脑", "平板", "耳机"],
    "科技": ["手机", "数码", "电子", "智能", "互联网", "AI", "5G"],
    "服装": ["时尚", "服饰", "穿搭", "鞋", "运动", "户外"],
    "汽车": ["出行", "新能源", "智驾", "SUV", "轿车"],
    "美妆": ["护肤", "彩妆", "化妆品", "美容", "面膜"],
    "食品": ["饮料", "零食", "餐饮", "乳业", "酒"],
    "游戏": ["手游", "端游", "电竞", "主机", "娱乐"],
    "家居": ["家电", "装修", "家具", "房产", "智能家居"],
    "旅游": ["酒店", "景区", "出行", "户外", "探索"],
    "金融": ["银行", "保险", "支付", "理财", "证券"],
    "教育": ["学习", "培训", "校园", "知识"],
    "医疗": ["健康", "医药", "医院", "保健"],
    "互联网": ["科技", "APP", "平台", "社交", "电商", "AI"],
    "影视": ["娱乐", "综艺", "短视频", "纪录片", "剧情"],
    "运动": ["户外", "健身", "跑步", "球鞋", "体育"],
    "奢侈品": ["高端", "限量", "时尚", "珠宝", "腕表"],
}


def _expand_industry(industry: str) -> set:
    """扩展行业关键词"""
    kw_set = set()
    for part in re.split(r'[，,、\s]+', industry.lower()):
        part = part.strip()
        if not part:
            continue
        kw_set.add(part)
        for alias_k, alias_v in _INDUSTRY_ALIAS.items():
            if part == alias_k or alias_k in part or part in alias_k:
                kw_set.update(v.lower() for v in alias_v)
    return kw_set


def score_industry_match(video: dict, industry: str = "") -> float:
    """行业匹配度：语义层级权重 (分类 > 标签 > 标题 > 内容)"""
    if not industry:
        return 7.5  # 默认高分
    score = 5.0
    keywords = _expand_industry(industry)

    # 1. 检查分类 (权重最高: 3.5)
    categories = video.get("categories", [])
    cat_text = " ".join(
        c.get("category_name", "") + " " + c.get("sub", {}).get("category_name", "")
        for c in categories
    ).lower()
    hits_cat = sum(1 for kw in keywords if kw in cat_text)
    score += min(hits_cat * 2.0, 3.5)

    # 2. 检查标签 (权重: 2.0)
    tag_text = " ".join(t.get("name", "").lower() for t in video.get("tags", []))
    hits_tag = sum(1 for kw in keywords if kw in tag_text)
    score += min(hits_tag * 1.0, 2.0)

    # 3. 检查标题与内容 (权重: 1.0)
    title = video.get("title", "").lower()
    content = (video.get("content", "") or "").lower()
    hits_text = sum(1 for kw in keywords if kw in title or kw in content)
    score += min(hits_text * 0.5, 1.0)

    return min(10.0, round(score, 1))


def score_pacing(video: dict) -> float:
    """节奏控制：基于视频分类的动态时长标准"""
    duration = video.get("duration", 0)
    if duration == 0:
        return 6.0
    
    # 获取主要分类
    categories = video.get("categories", [])
    main_cat = categories[0].get("category_name", "") if categories else ""
    
    # 动态标准
    if any(kw in main_cat for kw in ["纪录片", "微电影", "剧情", "访谈"]):
        # 长视频标准: 3-10min 最佳
        if 180 <= duration <= 600: return 9.0
        elif 120 <= duration < 180: return 8.0
        elif 600 < duration <= 900: return 8.0
        else: return 6.5
    else:
        # 广告/短片标准: 30s-2min 最佳
        if 30 <= duration <= 90: return 9.5
        elif 90 < duration <= 180: return 8.5
        elif 15 <= duration < 30: return 8.0
        elif 180 < duration <= 300: return 7.0
        else: return 5.5


def score_engagement(video: dict) -> float:
    """互动表现：收藏率(高权重) + 点赞率 + 分享率"""
    c = video.get("count", {})
    views = max(1, c.get("count_view", 0))
    likes = c.get("count_like", 0)
    collects = c.get("count_collect", 0)
    shares = c.get("count_share", 0)

    like_rate = likes / views
    collect_rate = collects / views
    share_rate = shares / views

    score = 5.0
    # 收藏率是核心指标 (专业度体现)
    if collect_rate > 0.015: score += 2.5
    elif collect_rate > 0.008: score += 1.5
    elif collect_rate > 0.004: score += 0.8
    
    # 点赞率
    if like_rate > 0.02: score += 1.5
    elif like_rate > 0.01: score += 0.8
    
    # 分享率
    if share_rate > 0.005: score += 1.0
    
    return min(10.0, round(score, 1))


def score_overall(scores: dict) -> float:
    """综合分：优化权重分配"""
    weights = {
        "creativity": 0.30,          # 创意第一
        "production_quality": 0.25,  # 制作第二
        "industry_match": 0.20,      # 行业匹配
        "pacing": 0.10,              # 节奏
        "engagement": 0.15,           # 互动
    }
    return round(sum(scores.get(k, 0) * v for k, v in weights.items()), 1)


def evaluate_video(video: dict, industry: str = "", style_preference: str = "") -> dict:
    """对单个视频进行算法评分"""
    scores = {
        "creativity": round(score_creativity(video), 1),
        "production_quality": round(score_production_quality(video), 1),
        "industry_match": round(score_industry_match(video, industry), 1),
        "pacing": round(score_pacing(video), 1),
        "engagement": score_engagement(video),
    }
    scores["overall"] = score_overall(scores)
    return scores


def build_evaluation_prompt(video: dict, industry: str, style_preference: str) -> str:
    """构造LLM评价prompt"""
    c = video.get("count", {})
    cats = video.get("categories", [])
    cat_str = ", ".join(
        f"{c.get('category_name','')}>{c.get('sub',{}).get('category_name','')}"
        for c in cats
    )
    tags = [t.get("name", "") for t in video.get("tags", [])]
    content = (video.get("content", "") or "")[:500]

    prompt = f"""作为戛纳广告奖评审，请对以下视频进行专业创意审计。

【视频元数据】
标题：{video.get('title', '')}
时长：{video.get('duration', 0)}秒
分类：{cat_str}
标签：{', '.join(tags)}
互动：{c.get('count_like', 0)}赞/{c.get('count_collect', 0)}收藏
文案：{content[:200]}

【行业背景】目标行业: {industry or '通用'} | 偏好风格: {style_preference or '不限'}

请给出 JSON（仅JSON无其他文字）：
{{"summary": "20字内深度点评（包含创意点与行业契合度）", "key_elements": ["专业术语1", "专业术语2", "专业术语3"]}}"""
    return prompt


def build_batch_prompt(videos: list, industry: str, style_preference: str) -> str:
    """构造批量评价prompt（20条一起），提升专业深度"""
    lines = []
    for i, v in enumerate(videos):
        c = v.get("count", {})
        cats = v.get("categories", [])
        cat_str = ", ".join(
            f"{c.get('category_name','')}>{c.get('sub',{}).get('category_name','')}"
            for c in cats
        )
        tags = [t.get("name", "") for t in v.get("tags", [])]
        content = (v.get("content", "") or "")[:150]
        lines.append(
            f"[{i+1}] 标题:{v.get('title','')} | {v.get('duration',0)}秒 | {cat_str} | "
            f"标签:{','.join(tags)} | 互动:{c.get('count_like',0)}赞/{c.get('count_collect',0)}藏 | 文案:{content}"
        )

    prompt = f"""作为戛纳广告奖评审，请对以下20条视频进行专业创意审计。

【行业背景】目标行业: {industry or '通用'} | 偏好风格: {style_preference or '不限'}

【评审任务】
1. summary: 20字内。必须包含[核心创意点]与[行业适配性]。避免空洞赞美，要具体到手法（如：非线性叙事、色彩叙事等）。
2. key_elements: 3-5个。必须使用专业术语（如：蒙太奇、高反差、视听通感、情绪留白、打破第四面墙等）。

视频列表：
{chr(10).join(lines)}

请返回严格JSON数组（仅JSON无其他文字）：
[{{\"summary\": \"专业点评\", \"key_elements\": [\"术语1\",\"术语2\"]}}, ...]"""
    return prompt


# ============================================================
# LLM API 评价（支持 Anthropic / DeepSeek）
# ============================================================

import os

# 启动时加载 .env 文件
def _load_dotenv():
    env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_file):
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    k, v = line.split("=", 1)
                    k, v = k.strip(), v.strip()
                    if k and v:
                        os.environ[k] = v

_load_dotenv()

# ---- 客户端获取 ----

def _get_llm_client():
    """自动检测 API Key，返回 (provider, client) 或 (None, None)"""
    # 优先 DeepSeek
    ds_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if ds_key:
        try:
            from openai import OpenAI
            return ("deepseek", OpenAI(api_key=ds_key, base_url="https://api.deepseek.com"))
        except ImportError:
            pass

    # 其次 Anthropic
    ant_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if ant_key:
        try:
            from anthropic import Anthropic
            return ("anthropic", Anthropic(api_key=ant_key))
        except ImportError:
            pass

    return (None, None)


def _get_anthropic_client():
    """兼容旧接口"""
    provider, client = _get_llm_client()
    return client if provider == "anthropic" else None


# ---- 批量评价 ----

def _call_deepseek(client, prompt: str) -> str:
    """通过 DeepSeek API 获取评价"""
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "你是一位拥有20年经验的顶级4A公司创意总监、戛纳广告奖评委。你的点评必须严谨、犀利、专业，使用标准广告行业术语。只输出JSON数组，不输出markdown和任何解释文字。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        max_tokens=3072,
        timeout=60,
    )
    return resp.choices[0].message.content.strip()


def _call_anthropic(client, prompt: str) -> str:
    """通过 Anthropic API 获取评价"""
    resp = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=3072,
        system="你是一位拥有20年经验的顶级4A公司创意总监、戛纳广告奖评委。你的点评必须犀利、专业，使用标准广告行业术语。只输出JSON数组，不输出markdown和任何解释文字。",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        timeout=60,
    )
    return resp.content[0].text.strip()


def evaluate_batch_with_llm(videos: list, industry: str = "", style_preference: str = "") -> list:
    """批量调 LLM API 生成 summary + key_elements

    Returns: [{"summary": "...", "key_elements": [...]}, ...]  长度 with videos一致
    失败时返回 None，上层应 fallback 到本地生成
    """
    provider, client = _get_llm_client()
    if provider is None or client is None:
        return None

    prompt = build_batch_prompt(videos, industry, style_preference)

    try:
        if provider == "deepseek":
            text = _call_deepseek(client, prompt)
        else:
            text = _call_anthropic(client, prompt)
    except Exception as e:
        print(f"[LLM] {provider} API call failed: {e}", file=sys.stderr)
        return None

    # 清理可能的 markdown 代码块包裹
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r'^```\w*\n?', '', text)
        text = re.sub(r'\n?```$', '', text)

    try:
        results = json.loads(text)
        if isinstance(results, list) and len(results) == len(videos):
            return results
        print(f"[LLM] Unexpected response structure: {type(results)}, len={len(results) if isinstance(results, list) else 'N/A'}")
        return None
    except json.JSONDecodeError as e:
        print(f"[LLM] JSON parse failed: {e}")
        print(f"[LLM] Raw response (first 500): {text[:500]}")
        return None


# ============================================================
# 简易本地摘要生成（无LLM时的fallback）
# ============================================================

_TEMPLATES = [
    "{brand}{category}佳作，{highlight}",
    "{category}类标杆案例，{highlight}",
    "{highlight}，{brand}品牌表达精准",
    "{style_tag}风格{category}，{highlight}",
    "{category}创意新范式，{highlight}",
]

def _extract_brand(title: str) -> str:
    """从标题提取品牌名"""
    m = re.match(r'^([一-鿿\w]+)[｜|·]', title)
    if m:
        return m.group(1)
    return ""

def _extract_dynamic_tags(video: dict) -> list:
    """从视频数据中动态提取关键元素标签"""
    tags = []
    duration = video.get("duration", 0)
    quality = video.get("quality", 0)
    badge = video.get("badge", "")
    c = video.get("count", {})
    views = c.get("count_view", 0)
    likes = c.get("count_like", 0)
    collect_rate = c.get("count_collect", 0) / max(1, views)

    if badge in ("recommend", "category_recommend"):
        tags.append("编辑推荐")
    if quality >= 4:
        tags.append("高清制作")
    if duration <= 60:
        tags.append("短小精悍")
    elif duration <= 180:
        tags.append("时长适中")
    elif duration >= 300:
        tags.append("深度长片")
    if views > 100000:
        tags.append("高播放量")
    elif views > 50000:
        tags.append("热度上升")
    if collect_rate > 0.02:
        tags.append("高收藏率")
    if likes / max(1, views) > 0.015:
        tags.append("高赞内容")

    # 从原标签里挑非重复的
    api_tags = [t.get("name", "") for t in video.get("tags", [])]
    for t in api_tags:
        if t not in tags and len(tags) < 6:
            tags.append(t)

    # 从分类提取
    for cat in video.get("categories", []):
        sub = cat.get("sub", {}).get("category_name", "")
        if sub and sub not in tags and len(tags) < 6:
            tags.append(sub)

    return tags[:6]


def generate_local_summary(video: dict, industry: str) -> tuple:
    """本地生成summary + key_elements（无LLM时的备选）"""
    title = video.get("title", "")
    brand = _extract_brand(title)
    cats = video.get("categories", [])
    main_cat = cats[0].get("category_name", "") if cats else ""
    sub_cat = cats[0].get("sub", {}).get("category_name", "") if cats else ""
    category = sub_cat or main_cat or "广告"
    c = video.get("count", {})
    views = c.get("count_view", 0)
    score = c.get("score", 0)

    key_el = _extract_dynamic_tags(video)

    # 生成summary：品牌 + 品类 + 亮点
    parts = []
    if brand:
        parts.append(brand)
    parts.append(category)

    # 选择亮点描述
    if score > 15000:
        highlight = "制作与创意俱佳"
    elif views > 50000:
        highlight = "口碑热度双高"
    elif views > 10000:
        highlight = "传播表现亮眼"
    elif score > 5000:
        highlight = "值得关注的新作"
    else:
        highlight = "潜力佳作"

    summary = f"{'·'.join(parts)} {highlight}"
    # 确保20字以内
    while len(summary) > 20:
        if len(parts) > 1:
            parts.pop()
            summary = f"{'·'.join(parts)} {highlight}"
        elif len(highlight) > 6:
            highlight = highlight[:6]
            summary = f"{'·'.join(parts)} {highlight}"
        else:
            break

    return summary[:20], key_el
