# -*- coding: utf-8 -*-
"""
视频评价引擎：多维元数据评分 (D1-D4) + LLM 总结
基于元数据，无需视频文件
"""
import json, math, re, os, sys, time
from datetime import datetime, timezone
from typing import Optional


# ============================================================
# 工具函数：贝叶斯平滑与百分位
# ============================================================

def bayesian_smooth(observed: float, prior_mean: float, prior_weight: float) -> float:
    """贝叶斯平滑：将小样本观测值向先验均值收缩"""
    return (observed * prior_weight + prior_mean) / (prior_weight + 1)


def safe_div(a, b, default=0.0):
    if b == 0:
        return default
    return a / b


def percentile_rank(value: float, pool: list) -> float:
    """计算 value 在 pool 中的百分位排名 (0-1)"""
    if not pool:
        return 0.5
    sorted_pool = sorted(pool)
    n = len(sorted_pool)
    # 二分查找位置
    lo, hi = 0, n
    while lo < hi:
        mid = (lo + hi) // 2
        if sorted_pool[mid] < value:
            lo = mid + 1
        else:
            hi = mid
    return lo / n


# ============================================================
# 品类对标池计算
# ============================================================

def compute_pool_baseline(pool: list) -> dict:
    """从搜索池计算品类基准参数，用于贝叶斯平滑先验

    pool: 搜索结果列表，每项含 count 字段
    """
    if not pool:
        return {
            "pool_size": 0,
            "median_view": 10000,
            "avg_collect_rate": 0.008,
            "avg_engage_rate": 0.02,
            "avg_share_rate": 0.003,
        }

    views = []
    collect_rates = []
    engage_rates = []
    share_rates = []

    for v in pool:
        c = v.get("count", {})
        vw = max(c.get("count_view", 0), 1)
        views.append(vw)
        collect_rates.append(safe_div(c.get("count_collect", 0), vw))

        # 互动率（加权）
        raw_engage = (c.get("count_like", 0) * 1 +
                      c.get("count_comment", 0) * 3 +
                      c.get("count_share", 0) * 5 +
                      c.get("count_collect", 0) * 8) / vw
        engage_rates.append(raw_engage)
        share_rates.append(safe_div(c.get("count_share", 0), vw))

    sorted_views = sorted(views)
    n = len(sorted_views)
    median_view = sorted_views[n // 2] if n > 0 else 10000

    total_view = sum(views)
    total_collect = sum(c.get("count_collect", 0) for c in (v.get("count", {}) for v in pool))
    total_share = sum(c.get("count_share", 0) for c in (v.get("count", {}) for v in pool))

    avg_collect_rate = safe_div(total_collect, total_view, 0.008)
    avg_share_rate = safe_div(total_share, total_view, 0.003)
    avg_engage_rate = sum(engage_rates) / n if n > 0 else 0.02

    # 过滤极端值
    avg_collect_rate = max(0.001, min(avg_collect_rate, 0.1))
    avg_share_rate = max(0.0001, min(avg_share_rate, 0.05))

    return {
        "pool_size": n,
        "median_view": median_view,
        "avg_collect_rate": avg_collect_rate,
        "avg_engage_rate": avg_engage_rate,
        "avg_share_rate": avg_share_rate,
    }


def compute_pool_percentiles(pool: list, baseline: dict) -> dict:
    """计算池内所有视频的平滑指标列表，用于百分位排名"""
    median_view = baseline["median_view"]
    alpha_c = baseline["avg_collect_rate"] * median_view  # 收藏先验
    alpha_e = baseline["avg_engage_rate"] * median_view    # 互动先验
    alpha_s = baseline["avg_share_rate"] * median_view      # 分享先验

    cr_smooth_list = []
    er_smooth_list = []
    vir_smooth_list = []

    for v in pool:
        c = v.get("count", {})
        vw = max(c.get("count_view", 0), 1)

        cr_smooth = safe_div(c.get("count_collect", 0) + alpha_c, vw + median_view)
        cr_smooth_list.append(cr_smooth)

        raw_engage = (c.get("count_like", 0) * 1 +
                      c.get("count_comment", 0) * 3 +
                      c.get("count_share", 0) * 5 +
                      c.get("count_collect", 0) * 8) / vw
        er_smooth = safe_div(raw_engage * vw + alpha_e, vw + median_view)
        er_smooth_list.append(er_smooth)

        vir_smooth = safe_div(c.get("count_share", 0) + alpha_s, vw + median_view)
        vir_smooth_list.append(vir_smooth)

    return {
        "cr_list": cr_smooth_list,
        "er_list": er_smooth_list,
        "vir_list": vir_smooth_list,
    }


# ============================================================
# D1 观众接受度
# ============================================================

def score_D1(video: dict, pool: list = None) -> dict:
    """观众接受度评分

    子指标：收藏率(0.35) + 互动率(0.30) + 传播力(0.35)
    使用贝叶斯平滑处理小样本偏差，搜索池内百分位归一化
    """
    c = video.get("count", {})
    vw = max(c.get("count_view", 0), 1)
    collects = c.get("count_collect", 0)
    likes = c.get("count_like", 0)
    comments = c.get("count_comment", 0)
    shares = c.get("count_share", 0)

    # 计算品类基准
    baseline = compute_pool_baseline(pool) if pool else compute_pool_baseline([])
    median_view = baseline["median_view"]

    # 贝叶斯平滑参数
    alpha_c = baseline["avg_collect_rate"] * median_view
    alpha_s = baseline["avg_share_rate"] * median_view
    alpha_e = baseline["avg_engage_rate"] * median_view

    # 子指标1：收藏率
    cr_raw = safe_div(collects, vw)
    cr_smooth = safe_div(collects + alpha_c, vw + median_view)

    # 子指标2：互动率（加权）
    raw_engage = (likes * 1 + comments * 3 + shares * 5 + collects * 8) / vw
    er_smooth = safe_div(raw_engage * vw + alpha_e, vw + median_view)

    # 子指标3：传播力
    vir_smooth = safe_div(shares + alpha_s, vw + median_view)

    # 百分位排名
    has_pool = pool and len(pool) >= 5
    if has_pool:
        pcts = compute_pool_percentiles(pool, baseline)
        cr_pct = percentile_rank(cr_smooth, pcts["cr_list"])
        er_pct = percentile_rank(er_smooth, pcts["er_list"])
        vir_pct = percentile_rank(vir_smooth, pcts["vir_list"])
        # 映射到 0-10
        cr_score = round(cr_pct * 10, 1)
        er_score = round(er_pct * 10, 1)
        vir_score = round(vir_pct * 10, 1)
        avg_pct = (cr_pct + er_pct + vir_pct) / 3
    else:
        # 无池时使用原始贝叶斯平滑值线性映射
        cr_score = round(min(cr_smooth / max(baseline["avg_collect_rate"], 0.001), 2.0) * 5, 1)
        er_score = round(min(er_smooth / max(baseline["avg_engage_rate"], 0.001), 2.0) * 5, 1)
        vir_score = round(min(vir_smooth / max(baseline["avg_share_rate"], 0.0001), 2.0) * 5, 1)
        cr_score = min(cr_score, 10.0)
        er_score = min(er_score, 10.0)
        vir_score = min(vir_score, 10.0)
        avg_pct = None

    # 综合
    D1 = cr_score * 0.35 + er_score * 0.30 + vir_score * 0.35
    D1 = round(D1, 1)

    # 置信度判断
    if vw < 500:
        confidence = "low"
    elif vw < 5000:
        confidence = "medium"
    else:
        confidence = "high"

    # 解释文本
    explanations = []
    if has_pool:
        explanations.append(f"收藏率池内前{int((1-cr_pct)*100)}%")
        explanations.append(f"互动率池内前{int((1-er_pct)*100)}%")
        explanations.append(f"传播力池内前{int((1-vir_pct)*100)}%")

    return {
        "score": D1,
        "confidence": confidence,
        "sub_scores": {
            "collect_rate": cr_score,
            "engagement_rate": er_score,
            "virality": vir_score,
        },
        "percentile": round(avg_pct * 100) if avg_pct is not None else None,
        "explanation": "；".join(explanations) if explanations else "无对标池，基于品类基准估算",
    }


# ============================================================
# D2 商业参考价值
# ============================================================

def score_D2(video: dict, pool: list = None, D3_score: float = 5.0) -> dict:
    """商业参考价值评分

    子指标：收藏信号(0.30) + 榜单加成(0.25) + 平台质量(0.20) + badge加成(0.10) + 团队信誉(0.15)
    """
    c = video.get("count", {})
    vw = max(c.get("count_view", 0), 1)

    # 1. 收藏信号（复用D1的收藏率逻辑）
    baseline = compute_pool_baseline(pool) if pool else compute_pool_baseline([])
    median_view = baseline["median_view"]
    alpha_c = baseline["avg_collect_rate"] * median_view
    cr_smooth = safe_div(c.get("count_collect", 0) + alpha_c, vw + median_view)

    if pool and len(pool) >= 5:
        pcts = compute_pool_percentiles(pool, baseline)
        cr_score = round(percentile_rank(cr_smooth, pcts["cr_list"]) * 10, 1)
    else:
        cr_score = round(min(cr_smooth / max(baseline["avg_collect_rate"], 0.001), 2.0) * 5, 1)
        cr_score = min(cr_score, 10.0)

    # 2. 榜单加成
    ranks = video.get("ranks", []) or []
    rank_weights = {
        "monthlyRanking": (1.5, 3),
        "staffPicks": (1.0, 5),
        "ad": (0.5, 10),
        "digital": (0.5, 10),
        "creative_recommend": (2.0, 1),
    }
    rank_bonus = 0.0
    for rank in ranks:
        code = rank.get("code", "") if isinstance(rank, dict) else str(rank)
        index = rank.get("index", rank.get("rank", 999)) if isinstance(rank, dict) else 999
        w = rank_weights.get(code, (0.3, 10))
        rank_bonus += max(0, w[1] - index) * w[0]
    rank_bonus = min(rank_bonus, 5.0)

    # 3. 平台质量分 (quality: 1-5 → 0-10)
    quality = video.get("quality", 3) or 3
    quality_score = quality / 5.0 * 10.0

    # 4. badge 加成
    badge_values = {
        "recommend": 1.5,
        "monthly_rank": 2.0,
        "weekly_rank": 1.0,
        "staffPicks": 2.5,
    }
    badge_bonus = 0.0
    badge = video.get("badge", "")
    if badge and badge in badge_values:
        badge_bonus += badge_values[badge]
    # display_badge
    display_badge = video.get("display_badge", {}) or {}
    db_name = display_badge.get("name", "")
    if db_name and db_name in badge_values:
        badge_bonus += badge_values[db_name] * 0.5
    badge_bonus = min(badge_bonus, 3.0)

    # 综合
    D2 = (cr_score * 0.30 +
          rank_bonus * 2.0 +           # rank_bonus 0-5, ×2 → 0-10
          quality_score * 0.20 +
          badge_bonus * 3.33 +         # badge_bonus 0-3, ×3.33 → 0-10
          D3_score * 0.15)
    D2 = round(min(D2, 10.0), 1)

    # 置信度
    has_ranks = len(ranks) > 0 if ranks else False
    if vw < 500 and not has_ranks:
        confidence = "low"
    elif has_ranks:
        confidence = "high"
    else:
        confidence = "medium"

    explanations = []
    if badge_bonus > 0:
        explanations.append(f"获得{int(badge_bonus*10)/10}分badge加成")
    if rank_bonus > 0:
        explanations.append(f"榜单加成{round(rank_bonus,1)}分")

    return {
        "score": D2,
        "confidence": confidence,
        "sub_scores": {
            "collect_signal": round(cr_score, 1),
            "rank_bonus": round(rank_bonus, 1),
            "quality_score": round(quality_score, 1),
            "badge_bonus": round(badge_bonus, 1),
            "team_credit": round(D3_score, 1),
        },
        "percentile": None,
        "explanation": "；".join(explanations) if explanations else "基于元数据的商业参考评估",
    }


# ============================================================
# D3 团队专业度
# ============================================================

def score_D3(video: dict) -> dict:
    """团队专业度评分

    子指标：导演影响力(0.35) + 团队完整度(0.40) + 行业认可度(0.25)
    """
    author = video.get("author", {}) or {}
    userinfo = author.get("userinfo", {}) or {}
    team = video.get("team", []) or []

    # --- 子指标1：导演影响力 ---
    follower_score = 0.0
    follower_count = userinfo.get("count_follower", 0) or 0
    if follower_count > 0:
        follower_score = min(math.log10(follower_count + 1) / math.log10(100000), 1.0) * 10

    recommend_score = 0.0
    article_count = userinfo.get("count_article", 0) or 0
    recommend_count = userinfo.get("count_recommend", 0) or 0
    if article_count > 0:
        recommend_rate = recommend_count / article_count
        recommend_score = min(recommend_rate / 0.5, 1.0) * 10

    pop_score = 0.0
    popularity = userinfo.get("count_popularity", 0) or 0
    if popularity > 0:
        pop_score = min(math.log10(popularity + 1) / math.log10(10000000), 1.0) * 10

    authority_raw = follower_score * 0.3 + recommend_score * 0.4 + pop_score * 0.3

    # --- 子指标2：团队完整度 ---
    core_roles = {"导演", "编剧", "摄影指导", "摄影师", "美术指导", "剪辑师", "调色师"}
    bonus_roles = {"声音设计", "混音师", "航拍", "特效", "灯光", "造型指导"}

    # 从 team 列表提取角色
    filled_roles = set()
    for member in team:
        if not isinstance(member, dict):
            continue
        role = member.get("role", "") or member.get("occupation", "") or ""
        filled_roles.add(role)
        # 也检查 author 的 occupation
    author_occupation = author.get("occupation", "") or userinfo.get("occupation", "") or ""
    if author_occupation:
        filled_roles.add(author_occupation)
    # 导演默认有
    username = userinfo.get("username", "")
    if username:
        filled_roles.add("导演")

    team_completeness = 0.0
    for role in core_roles:
        if any(role in r for r in filled_roles):
            team_completeness += 1.5
    team_completeness = min(team_completeness, 7.0)

    bonus = sum(0.5 for role in bonus_roles if any(role in r for r in filled_roles))
    team_completeness += min(bonus, 2.0)

    # 团队规模加成
    team_user_count = video.get("team_user_count", 0) or len(team)
    if team_user_count >= 20:
        team_completeness += 1.0
    elif team_user_count >= 10:
        team_completeness += 0.5

    team_completeness = min(team_completeness, 10.0)

    # --- 子指标3：行业认可度 ---
    recognition = 0.0

    # 金雀奖
    jinque_count = 0
    for member in team:
        if not isinstance(member, dict):
            continue
        mi = member.get("userinfo", {}) or {}
        jq = mi.get("count_jin_que", 0) or 0
        jinque_count += jq
    recognition += min(jinque_count, 3) * 1.5

    # 认证描述关键词
    prestige_keywords = ["金狮奖", "金雀奖", "获奖", "代表作", "学院", "国际"]
    for member in team:
        if not isinstance(member, dict):
            continue
        mi = member.get("userinfo", {}) or {}
        desc = mi.get("verify_description", "") or ""
        for kw in prestige_keywords:
            if kw in desc:
                recognition += 0.5
                break
    # 也检查作者
    verify_desc = userinfo.get("verify_description", "") or ""
    for kw in prestige_keywords:
        if kw in verify_desc:
            recognition += 0.5
            break

    recognition = min(recognition, 5.0)
    recognition_score = recognition * 2.0  # 映射到 0-10

    # --- 综合 ---
    D3 = authority_raw * 0.35 + team_completeness * 0.40 + recognition_score * 0.25
    D3 = round(min(D3, 10.0), 1)

    # 置信度
    has_team_data = len(team) > 0 or userinfo.get("count_article", 0)
    confidence = "high" if has_team_data else "unavailable"

    # 解释
    exp_parts = []
    if username:
        exp_parts.append(f"{username}(粉{follower_count}+推{recommend_count})")
    if len(team) > 0:
        exp_parts.append(f"{len(team)}人团队")
    if jinque_count > 0:
        exp_parts.append(f"金雀×{jinque_count}")

    return {
        "score": D3,
        "confidence": confidence,
        "sub_scores": {
            "director_authority": round(authority_raw, 1),
            "team_completeness": round(team_completeness, 1),
            "industry_recognition": round(recognition_score, 1),
        },
        "explanation": "；".join(exp_parts) if exp_parts else "团队数据不足",
    }


# ============================================================
# D4 内容新鲜度
# ============================================================

def score_D4(video: dict) -> dict:
    """内容新鲜度评分，基于发布时间"""
    publish_time = video.get("publish_time", 0) or 0
    if publish_time == 0:
        return {
            "score": 5.0,
            "publish_date": "未知",
            "days_since_publish": None,
        }

    now = datetime.now(timezone.utc)
    pub_dt = datetime.fromtimestamp(publish_time, tz=timezone.utc)
    days = (now - pub_dt).days

    D4 = max(0.0, 10.0 - days / 30.0)
    D4 = round(D4, 1)

    return {
        "score": D4,
        "publish_date": pub_dt.strftime("%Y-%m-%d"),
        "days_since_publish": days,
    }


# ============================================================
# 综合评分
# ============================================================

SCENARIO_WEIGHTS = {
    "default": {"D1": 0.30, "D2": 0.35, "D3": 0.25, "D4": 0.10},
    "advertiser": {"D1": 0.15, "D2": 0.50, "D3": 0.30, "D4": 0.05},
    "creator_learning": {"D1": 0.25, "D2": 0.20, "D3": 0.40, "D4": 0.15},
    "latest_discovery": {"D1": 0.25, "D2": 0.25, "D3": 0.15, "D4": 0.35},
}


def score_all_dimensions(video: dict, pool: list = None) -> dict:
    """计算所有4个维度的评分"""
    D3 = score_D3(video)
    D3_score = D3["score"]

    return {
        "D1_audience_reception": score_D1(video, pool),
        "D2_commercial_value": score_D2(video, pool, D3_score),
        "D3_team_professionalism": D3,
        "D4_freshness": score_D4(video),
    }


def apply_scenario_weights(scores: dict, scenario: str = "default") -> float:
    """场景化加权综合分"""
    weights = SCENARIO_WEIGHTS.get(scenario, SCENARIO_WEIGHTS["default"])
    total = 0.0

    key_map = {
        "D1": "D1_audience_reception",
        "D2": "D2_commercial_value",
        "D3": "D3_team_professionalism",
        "D4": "D4_freshness",
    }

    for dim_key, score_key in key_map.items():
        dim_data = scores.get(score_key, {})
        score_val = dim_data.get("score", 5.0) if isinstance(dim_data, dict) else 5.0
        total += score_val * weights[dim_key]

    return round(total, 1)


def compute_scenario_scores(scores: dict) -> dict:
    """计算所有场景的综合分"""
    return {
        s: apply_scenario_weights(scores, s)
        for s in SCENARIO_WEIGHTS
    }


# ============================================================
# 完整评分格式（对齐设计文档 §9）
# ============================================================

def build_full_evaluation(video: dict, pool: list = None,
                          category_name: str = "通用") -> dict:
    """构建完整的多维评分输出"""
    article_id = video.get("id", "")

    scores = score_all_dimensions(video, pool)
    scenarios = compute_scenario_scores(scores)

    # 缺失维度
    missing = [
        "A1_technical_specs",
        "A2_image_quality",
        "B1_composition",
        "B2_camera_movement",
        "B3_color_grading",
        "B4_sound_design",
        "C1_pacing",
        "C2_narrative",
        "C3_emotional_impact",
    ]

    pool_info = {
        "category": category_name,
        "pool_size": len(pool) if pool else 0,
        "pool_period_days": 365,
        "min_view_threshold": 100,
    }

    return {
        "article_id": article_id,
        "scored_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "pool_info": pool_info,
        "scores": scores,
        "scenarios": scenarios,
        "missing_dimensions": missing,
        "upgrade_hint": "上传视频文件可解锁剩余9个维度评分",
    }


# ============================================================
# L1 快速评分引擎（保留并增强）
# ============================================================

def c_eye_l1_fast_scorer(videos: list, industry: str = "") -> list:
    """极速打分与分流引擎 (L1级) — 注入 D1-D4 多维评分"""
    pool = videos if len(videos) >= 5 else None

    scored_results = []
    for item in videos:
        video_id = str(item.get("id"))
        title = item.get("title", "").strip()
        duration = item.get("duration", 0)
        counts = item.get("count", {})
        views = max(counts.get("count_view", 0), 1)
        collects = counts.get("count_collect", 0)
        shares = counts.get("count_share", 0)
        tags = [t.get("name", "") for t in item.get("tags", [])]
        categories = item.get("categories", [])
        main_cat = categories[0].get("category_name", "") if categories else "广告"
        sub_cat = categories[0].get("sub", {}).get("category_name", "") if categories else ""
        cover = item.get("cover", "")

        # --- 维度 1: 概念资产沉淀 ---
        narrative_mode = "多线剪辑流"
        if any(x in title or x in tags for x in ["采访", "纪录", "真实"]):
            narrative_mode = "伪纪录访谈流"
        elif duration > 120:
            narrative_mode = "长线叙事线索"
        elif any(x in tags for x in ["混剪", "快节奏"]):
            narrative_mode = "视觉碎片蒙太奇"

        # 使用与 L2 统一的 _STYLE_KEYWORD_MAP
        visual_style = "标准商业调性"
        tag_str = "".join(tags)
        for kw, style_name in _STYLE_KEYWORD_MAP.items():
            if kw in tag_str and visual_style == "标准商业调性":
                visual_style = style_name

        concept_asset = f"{narrative_mode} + {visual_style}"

        # --- 维度 2: 叙事效率预警 ---
        is_recommended = "recommend" in item.get("badge", "")
        if duration < 45 or is_recommended:
            hook_strength = "极强 (视听轰炸)"
            platform_match = "抖音/小红书信息流"
        elif duration < 120:
            hook_strength = "中等 (情绪抓手)"
            platform_match = "分众电梯/社交媒体"
        else:
            hook_strength = "偏弱 (慢热铺垫)"
            platform_match = "品牌发布会/B站长视频"

        # --- 维度 3: 同行抄作业指数 ---
        raw_score = ((collects * 2.5) + (shares * 1.5)) / (views ** 0.8)
        reference_score = min(round(raw_score * 15, 1), 10.0)
        if is_recommended:
            reference_score = max(reference_score, 8.5)

        # --- 维度 4: 品牌通感标签 ---
        synesthesia = []
        mapping = {
            "极简": "#高智感", "运动": "#爆发力", "唯美": "#松弛感",
            "手持": "#呼吸感", "暴力": "#冲击力", "延时": "#史诗感",
            "黑白": "#颗粒感", "宏大": "#神圣感", "趣味": "#网感"
        }
        for k, v in mapping.items():
            if k in tag_str:
                synesthesia.append(v)
        if not synesthesia:
            synesthesia = ["#标准商业感", "#提案稳健型"]
        synesthesia = synesthesia[:3]

        # --- 维度 5: 制作班底 ---
        team = item.get("team", [])
        team_len = len(team) if isinstance(team, list) else 0
        author = item.get("author", {})
        vip_level = author.get("userinfo", {}).get("vip_flag", 0) if isinstance(author, dict) else 0

        # 与 L2 _extract_l2_rule_based 共用统一阈值
        quality = item.get("quality", 3) or 3
        if (vip_level >= 3 or quality >= 4) and team_len >= 15:
            budget_class = "A级 (百万级大制作)"
            soul_part = "导演组/美术置景"
        elif team_len >= 8 or quality >= 3:
            budget_class = "B级 (30-50万标准TVC)"
            soul_part = "剪辑/后期调色"
        else:
            budget_class = "C级 (10万内轻量执行)"
            soul_part = "独立创作/单兵作战"

        # --- 注入 D1-D4 多维评分 ---
        dim_scores = score_all_dimensions(item, pool)
        scenarios = compute_scenario_scores(dim_scores)

        scored_results.append({
            "id": video_id,
            "title": title,
            "cover": cover,
            "duration": duration,
            "reference_score": reference_score,
            "dimensions": {
                "concept_asset": concept_asset,
                "efficiency": {"hook": hook_strength, "platform": platform_match},
                "synesthesia": synesthesia,
                "budget": budget_class,
                "soul": soul_part
            },
            "stats": {"views": views, "collects": collects, "shares": shares},
            "author": author.get("userinfo", {}).get("username", "") if isinstance(author, dict) else "",
            "web_url": item.get("web_url", ""),
            "tab_category": sub_cat or main_cat,
            # 新增 D1-D4 评分
            "scores": {
                "D1": dim_scores["D1_audience_reception"]["score"],
                "D2": dim_scores["D2_commercial_value"]["score"],
                "D3": dim_scores["D3_team_professionalism"]["score"],
                "D4": dim_scores["D4_freshness"]["score"],
            },
            "scenario_scores": scenarios,
            "dimension_details": dim_scores,
        })

    # 默认按 default 场景综合分排序
    scored_results.sort(key=lambda x: x["scenario_scores"].get("default", x["reference_score"]), reverse=True)
    return scored_results


# ============================================================
# LLM API 评价（保留）
# ============================================================

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
    """构造批量评价prompt"""
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

    prompt = f"""作为戛纳广告奖评审，请对以下视频进行专业创意审计。

【行业背景】目标行业: {industry or '通用'} | 偏好风格: {style_preference or '不限'}

【评审任务】
1. summary: 20字内。必须包含[核心创意点]与[行业适配性]。避免空洞赞美，要具体到手法（如：非线性叙事、色彩叙事等）。
2. key_elements: 3-5个。必须使用专业术语（如：蒙太奇、高反差、视听通感、情绪留白、打破第四面墙等）。

视频列表：
{chr(10).join(lines)}

请返回严格JSON数组（仅JSON无其他文字）：
[{{"summary": "专业点评", "key_elements": ["术语1","术语2"]}}, ...]"""
    return prompt


def build_ecd_audit_prompt(video: dict, industry: str, style_preference: str, l2_metadata: dict = None) -> str:
    """构造 4A ECD 风格的商业审计 prompt，融合 L2 AI 元数据"""
    tags = [t.get("name", "") for t in video.get("tags", [])]
    content = (video.get("content", "") or "")[:800]

    # 提取基本 L1 属性用于 Context
    l1_data = c_eye_l1_fast_scorer([video], industry)[0] if video else {}

    # 构建 L2 元数据上下文块
    l2_context_lines = []
    if l2_metadata:
        brand = l2_metadata.get("brand", "")
        brand_tier = l2_metadata.get("brand_tier", "")
        product = l2_metadata.get("product", "")
        visual_style = l2_metadata.get("visual_style", "")
        mood = l2_metadata.get("mood", "")
        budget_tier = l2_metadata.get("budget_tier", "")
        style_keywords = l2_metadata.get("style_keywords", [])
        commercial_type = l2_metadata.get("commercial_type", "")
        engagement_type = l2_metadata.get("engagement_type", "")

        if brand:
            l2_context_lines.append(f"- AI 识别品牌: {brand} ({brand_tier})" + (f" / {product}" if product else ""))
        if commercial_type:
            l2_context_lines.append(f"- 商业片类型: {commercial_type}")
        if visual_style:
            l2_context_lines.append(f"- AI 推断视觉风格: {visual_style}")
        if mood:
            l2_context_lines.append(f"- AI 推断情绪基调: {mood}")
        if style_keywords:
            l2_context_lines.append(f"- 风格关键词: {', '.join(style_keywords)}")
        if budget_tier:
            l2_context_lines.append(f"- AI 估算制作级别: {budget_tier}")
        if engagement_type:
            l2_context_lines.append(f"- 互动模式分类: {engagement_type}")

    l2_context = "\n".join(l2_context_lines) if l2_context_lines else "(未启用 L2 元数据分析)"

    prompt = f"""你是一位资深广告策略分析师。请仅根据以下**已有元数据**，从商业投放视角给出客观判断。**注意：你没有看到视频本身，禁止对画面、镜头、色彩、运镜等视听元素做任何具体描述或猜测。**

【已知数据】
- 作品名称: {video.get('title', '')}
- 时长: {video.get('duration', 0)}秒
- 品类标签: {l1_data.get('tab_category', '通用')}
- 风格标签: {', '.join(l1_data.get('dimensions', {}).get('synesthesia', []))}
- 制作级别: {l1_data.get('dimensions', {}).get('budget', 'B级')}
- 目标行业: {industry or '通用'} / 偏好风格: {style_preference or '不限'}
- 创作者自述: {content}

【L2 AI 元数据】(从标签/分类/团队推断)
{l2_context}

【约束】
1. 只基于上述数据分析，不编造任何视听细节。
2. 若数据不足以支撑某结论，须明确指出"基于元数据推断"或"需查看视频确认"。
3. 语言精炼、务实，避免浮夸形容词。

【输出格式】(严格 JSON，无其他文字)

[{{
  "🚨 叙事效率预警": "根据时长({video.get('duration', 0)}秒)、品类(L2推断为{l2_metadata.get('commercial_type', '未知') if l2_metadata else '未知'})和互动数据，判断该片在信息流投放中的适配性、可能的完播率风险。80-120字。",
  "💬 商业提案 PPT 话术直通车": [
    {{
      "针对同品类客户提案": "基于标签和品类定位，提炼该片可复用的创作策略和提案方向。60-80字。"
    }},
    {{
      "针对跨品类平移（{industry or '跨品类'}）提案": "分析该片的风格标签如何迁移到目标行业，给出跨界提案思路。60-80字。"
    }}
  ]
}}]
"""
    return prompt


# ============================================================
# L2 模块1：LLM 元数据推理（品牌/风格/情绪/预算提取）
# ============================================================

# 品牌名识别词表（用于规则回退）
_BRAND_PATTERNS = [
    ("华为", "国际一线"), ("HUAWEI", "国际一线"), ("Apple", "国际一线"), ("苹果", "国际一线"),
    ("小米", "国内一线"), ("OPPO", "国内一线"), ("vivo", "国内一线"), ("荣耀", "国内一线"),
    ("奔驰", "国际一线"), ("BMW", "国际一线"), ("宝马", "国际一线"), ("奥迪", "国际一线"),
    ("小鹏", "新锐"), ("蔚来", "新锐"), ("理想", "新锐"), ("比亚迪", "国内一线"),
    ("耐克", "国际一线"), ("Nike", "国际一线"), ("Adidas", "国际一线"), ("阿迪达斯", "国际一线"),
    ("可口可乐", "国际一线"), ("百事", "国际一线"), ("麦当劳", "国际一线"),
    ("腾讯", "国内一线"), ("阿里", "国内一线"), ("字节", "国内一线"),
    ("安踏", "国内一线"), ("李宁", "国内一线"),
]

_STYLE_KEYWORD_MAP = {
    "暗黑": "暗黑工业风", "梦幻": "梦幻超现实", "赛博": "赛博朋克", "复古": "胶片复古",
    "极简": "极简冷淡", "高级": "高智感", "科技": "科技金属感", "暖": "暖调自然",
    "冷": "冷调高级", "快节奏": "高密度快剪", "慢": "舒缓叙事", "延时": "史诗延时",
    "手持": "手持呼吸感", "黑白": "黑白颗粒感", "CG": "数字CG", "三维": "三维渲染",
    "动画": "动画风格", "唯美": "唯美诗意", "运动": "运动爆发力",
}

_MOOD_KEYWORD_MAP = {
    "梦幻": "轻盈梦幻", "律动": "节奏律动", "风": "自由飘逸", "暗黑": "暗黑压抑",
    "炫酷": "炫酷炸裂", "温暖": "温暖治愈", "感人": "感人至深", "搞笑": "轻松幽默",
    "燃": "热血燃爆", "青春": "青春洋溢", "孤独": "孤独寂寥", "震撼": "震撼磅礴",
    "清新": "清新自然", "浪漫": "浪漫甜蜜", "惊悚": "紧张惊悚",
}


def extract_l2_metadata(video: dict, industry: str = "", style_preference: str = "",
                        l1_data: dict = None) -> dict:
    """M1: 从视频元数据中提取 L2 级结构化信息（品牌/风格/情绪/预算）

    优先使用 LLM 推理，LLM 不可用时回退到规则匹配。
    接收 L1 数据复用已有推断，避免独立重复分析。
    """
    provider, client = _get_llm_client()
    if provider and client:
        result = _extract_l2_with_llm(video, industry, style_preference, provider, client)
        if result:
            return result

    return _extract_l2_rule_based(video, industry, style_preference, l1_data)


def _extract_l2_with_llm(video, industry, style_preference, provider, client):
    """使用 LLM 提取 L2 元数据"""
    title = video.get("title", "") or ""
    tags = [t.get("name", "") for t in (video.get("tags") or [])]
    cats = video.get("categories") or []
    main_cat = cats[0].get("category_name", "") if cats else ""
    sub_cat = cats[0].get("sub", {}).get("category_name", "") if cats else ""
    content = (video.get("content") or "")[:400]
    duration = video.get("duration", 0)
    team = video.get("team", []) or []
    team_len = len(team)
    quality = video.get("quality", 3) or 3
    author = video.get("author", {}) or {}
    ui = author.get("userinfo", {}) or {}
    username = ui.get("username", "")
    verify_desc = ui.get("verify_description", "")

    prompt = f"""你是一位广告片分析专家。请根据以下元数据，提取结构化信息。

标题: {title}
时长: {duration}秒
分类: {main_cat} > {sub_cat}
标签: {", ".join(tags) if tags else "无"}
画质等级: {quality}/5
团队人数: {team_len}人
导演: {username}
导演认证: {verify_desc}
目标行业: {industry or "通用"}
偏好风格: {style_preference or "不限"}
文案摘要: {content[:200]}

请输出严格JSON（仅JSON，无其他文字）：
{{"brand":"品牌名称","brand_tier":"国际一线/国内一线/新锐/白牌","product":"产品名","industry":"行业","commercial_type":"品牌态度片/产品功能片/促销片/招聘片/活动记录/其他","visual_style":"一句话视觉风格描述","style_keywords":["风格词1","风格词2","风格词3"],"mood":"情绪基调一句话","mood_keywords":["情绪词1","情绪词2"],"budget_tier":"A级/B级/C级","budget_reasoning":"简短理由"}}"""

    try:
        if provider == "deepseek":
            text = _call_deepseek(client, prompt, L2_SYSTEM_PROMPT)
        else:
            text = _call_anthropic(client, prompt, L2_SYSTEM_PROMPT)
    except Exception as e:
        print(f"[L2-LLM] API call failed: {e}", file=sys.stderr)
        return None

    try:
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r'^```\w*\n?', '', text)
            text = re.sub(r'\n?```$', '', text)
        data = json.loads(text)
        if isinstance(data, dict) and "brand" in data:
            return data
    except (json.JSONDecodeError, TypeError):
        pass
    return None


def _extract_l2_rule_based(video, industry="", style_preference="", l1_data=None):
    """规则匹配回退：从标签/分类/团队推断 L2 元数据。复用 L1 已有推断。"""
    title = video.get("title", "") or ""
    tags = [t.get("name", "") for t in (video.get("tags") or [])]
    tag_str = " ".join(tags)
    cats = video.get("categories") or []
    main_cat = cats[0].get("category_name", "") if cats else ""
    sub_cat = cats[0].get("sub", {}).get("category_name", "") if cats else ""
    team = video.get("team", []) or []
    team_len = len(team)
    quality = video.get("quality", 3) or 3
    author = video.get("author", {}) or {}
    ui = author.get("userinfo", {}) or {}
    username = ui.get("username", "")
    duration = video.get("duration", 0)

    # 品牌识别
    brand = ""
    brand_tier = "白牌"
    product = ""
    m = re.match(r'^([一-鿿\w]+)[｜|·\-—]', title)
    if m:
        brand = m.group(1)
    for kw, tier in _BRAND_PATTERNS:
        if kw.lower() in title.lower() or kw.lower() in tag_str.lower():
            brand = kw
            brand_tier = tier
            break
    if not brand:
        brand = username or "未知品牌"
        brand_tier = "新锐"

    # 行业推断
    inferred_industry = industry or sub_cat or main_cat or "通用"

    # 视觉风格推断 — 优先复用 L1 已有结论
    l1_dimensions = (l1_data or {}).get("dimensions", {}) or {}
    visual_style = l1_dimensions.get("visual_style", "") or "标准商业调性"
    style_keywords = []
    for kw, style_name in _STYLE_KEYWORD_MAP.items():
        if kw in title or kw in tag_str:
            if visual_style == "标准商业调性" or not visual_style:
                visual_style = style_name
            if len(style_keywords) < 4:
                style_keywords.append(style_name)
    if not style_keywords:
        tag_names = [t.get("name", "") for t in (video.get("tags") or [])[:4]]
        style_keywords = [t for t in tag_names if t] or ["商业广告"]

    # 情绪推断
    mood = "中性专业"
    mood_keywords = []
    for kw, mood_name in _MOOD_KEYWORD_MAP.items():
        if kw in title or kw in tag_str:
            if mood == "中性专业":
                mood = mood_name
            if len(mood_keywords) < 3:
                mood_keywords.append(mood_name)
    if not mood_keywords:
        mood_keywords = ["专业"]

    # 商业类型推断
    commercial_type = "品牌态度片"
    if duration < 30:
        commercial_type = "信息流广告"
    elif duration < 90:
        commercial_type = "产品功能片"
    elif duration > 300:
        commercial_type = "长纪录片/活动记录"

    # 预算级别估算
    if quality >= 4 and team_len >= 15:
        budget_tier = "A级（百万级制作）"
        budget_reasoning = f"{team_len}人完整工业团队+画质{quality}级"
    elif team_len >= 8 or quality >= 3:
        budget_tier = "B级（30-50万标准TVC）"
        budget_reasoning = f"{team_len}人团队+画质{quality}级"
    else:
        budget_tier = "C级（10万内轻量执行）"
        budget_reasoning = f"{team_len}人团队"

    # 目标受众推断
    target_audience = "品牌方/广告主"
    if any(kw in tag_str for kw in ["VLOG", "探店", "测评", "开箱"]):
        target_audience = "C端消费者"
    elif any(kw in tag_str for kw in ["宣传片", "企业", "年会", "招聘"]):
        target_audience = "B端企业客户"
    elif any(kw in tag_str for kw in ["MV", "微电影", "剧情", "短片"]):
        target_audience = "泛娱乐观众"
    elif any(kw in tag_str for kw in ["教程", "知识", "科普"]):
        target_audience = "学习者/从业者"

    # 导演专长推断
    creator_specialty = "商业广告制作"
    if any(kw in tag_str for kw in ["动画", "CG", "三维", "特效"]):
        creator_specialty = "CG/动画/视效"
    elif any(kw in tag_str for kw in ["航拍", "延时", "风光"]):
        creator_specialty = "航拍/延时摄影"
    elif any(kw in tag_str for kw in ["剧情", "微电影", "故事"]):
        creator_specialty = "剧情/故事片"
    elif any(kw in tag_str for kw in ["快剪", "混剪", "节奏"]):
        creator_specialty = "快节奏剪辑/信息流"
    elif any(kw in tag_str for kw in ["产品", "TVC", "广告"]):
        creator_specialty = "产品广告/TVC"

    return {
        "brand": brand,
        "brand_tier": brand_tier,
        "product": product,
        "industry": inferred_industry,
        "commercial_type": commercial_type,
        "visual_style": visual_style,
        "style_keywords": style_keywords[:5],
        "mood": mood,
        "mood_keywords": mood_keywords[:3],
        "budget_tier": budget_tier,
        "budget_reasoning": budget_reasoning,
        "target_audience": target_audience,
        "creator_specialty": creator_specialty,
        "_fallback": True,
    }


# ============================================================
# L2 模块2：ML 模式识别（互动分类/质量预测/异常检测）
# ============================================================

def _classify_engagement(views, likes, collects, shares, comments):
    """互动模式分类（纯规则引擎，无需训练数据）

    返回互动类型、置信度和信号列表。
    """
    if views < 50:
        return {"type": "数据不足", "confidence": 0.0, "signals": ["views_too_low"]}

    like_rate = safe_div(likes, views)
    collect_rate = safe_div(collects, views)
    share_rate = safe_div(shares, views)
    comment_rate = safe_div(comments, views)
    collect_like_ratio = safe_div(collects, max(likes, 1))

    signals = []
    type_name = "普通型"
    confidence = 0.0
    score = 0.0

    # 爆发型：高播放 + 高分享率 + 中等收藏率
    if views > 50000 and share_rate > 0.01:
        score += 3
        signals.append(f"高播放量({views})")
        signals.append(f"高分享率({share_rate:.3f})")

    # 参考型：极高收藏率 + 高收藏/点赞比
    if collect_rate > 0.012 or collect_like_ratio > 0.8:
        score += 4
        signals.append(f"高收藏率({collect_rate:.3f})")
        if collect_like_ratio > 1:
            signals.append(f"收藏>/点赞({collect_like_ratio:.1f}x)")

    # 口碑型：低播放 + 极高点赞率 + 极高评论率
    if views < 20000 and like_rate > 0.04 and comment_rate > 0.005:
        score += 3
        signals.append(f"高互动低播放(like={like_rate:.3f})")

    # 长尾型：低播放 + 持续增长迹象
    if views < 10000 and collect_rate > 0.005:
        score += 2
        signals.append(f"长尾潜力(collect={collect_rate:.3f})")

    if score >= 5:
        type_name = "爆发型" if share_rate > 0.01 else "参考型"
        confidence = min(0.95, score / 7.0)
    elif score >= 3:
        type_name = "口碑型" if comment_rate > 0.005 else "长尾型"
        confidence = score / 7.0
    else:
        type_name = "普通型"
        confidence = 0.5

    return {
        "type": type_name,
        "confidence": round(confidence, 2),
        "signals": signals if signals else ["各项指标接近品类均值"],
    }


def _predict_quality(video):
    """质量层级预测（规则加权，输出 0-10 分）"""
    quality_raw = video.get("quality", 3) or 3
    team = video.get("team", []) or []
    team_len = len(team)
    author = video.get("author", {}) or {}
    ui = author.get("userinfo", {}) or {}
    popularity = ui.get("count_popularity", 0) or 0
    recommend_rate = safe_div(ui.get("count_recommend", 0), max(ui.get("count_article", 1), 1))

    # 基础分从 quality 1-5 映射到 0-10
    base = (quality_raw / 5.0) * 10.0

    # 团队加成（最多 +2）
    team_bonus = min(2.0, team_len / 10.0)

    # 导演信号加成（最多 +1.5）
    director_bonus = 0.0
    if popularity > 1000000:
        director_bonus += 0.8
    elif popularity > 100000:
        director_bonus += 0.4
    if recommend_rate > 0.3:
        director_bonus += 0.7
    elif recommend_rate > 0.15:
        director_bonus += 0.3
    director_bonus = min(director_bonus, 1.5)

    # 金雀奖加成
    jinque_bonus = 0.0
    for member in team:
        if not isinstance(member, dict):
            continue
        jq = (member.get("userinfo") or {}).get("count_jin_que", 0) or 0
        jinque_bonus += min(jq, 5) * 0.1
    jinque_bonus = min(jinque_bonus, 1.0)

    predicted = min(10.0, base + team_bonus + director_bonus + jinque_bonus)

    return {
        "raw_quality": quality_raw,
        "predicted_score": round(predicted, 1),
        "boost_factors": {
            "base_from_quality": round(base, 1),
            "team_bonus": round(team_bonus, 1),
            "director_bonus": round(director_bonus, 1),
            "jinque_bonus": round(jinque_bonus, 1),
        },
    }


def _detect_anomaly(video):
    """异常检测 — 7 个信号综合判断（无监督规则方法）"""
    c = video.get("count", {})
    views = max(c.get("count_view", 0), 1)
    likes = c.get("count_like", 0)
    collects = c.get("count_collect", 0)
    shares = c.get("count_share", 0)
    comments = c.get("count_comment", 0)
    score = c.get("score", 0)

    flags = []
    anomaly_score = 0.0

    # 信号1：收藏率异常高但评论率极低
    collect_rate = safe_div(collects, views)
    comment_rate = safe_div(comments, views)
    if collect_rate > 0.05 and comment_rate < 0.001:
        flags.append("collect_without_comment")
        anomaly_score += 2.0

    # 信号2：播放量高但平台分低
    if views > 50000 and score < 1000:
        flags.append("high_views_low_score")
        anomaly_score += 1.5

    # 信号3：点赞/收藏/分享比例严重失衡
    if likes > 0:
        share_like_ratio = safe_div(shares, likes)
        if share_like_ratio > 2.0:
            flags.append("share_like_imbalance")
            anomaly_score += 1.0

    # 信号4：收藏/点赞比异常（正常 0.1-1.5）
    if likes > 0:
        cl_ratio = safe_div(collects, likes)
        if cl_ratio > 3.0:
            flags.append("collect_like_ratio_high")
            anomaly_score += 1.5
        elif cl_ratio < 0.01 and likes > 100:
            flags.append("collect_like_ratio_low")
            anomaly_score += 0.5

    # 信号5：播放量很高但粉丝极少
    author = video.get("author", {}) or {}
    ui = author.get("userinfo", {}) or {}
    follower_count = ui.get("count_follower", 0) or 0
    if views > 100000 and follower_count < 100:
        flags.append("viral_without_followers")
        anomaly_score += 1.0

    # 信号6：评论率异常高（疑似水军）
    if comment_rate > 0.1 and views > 1000:
        flags.append("comment_rate_high")
        anomaly_score += 1.0

    # 信号7：零互动（疑似爬虫数据）
    if views > 1000 and likes == 0 and collects == 0 and comments == 0:
        flags.append("zero_engagement")
        anomaly_score += 3.0

    anomaly_score = min(anomaly_score, 10.0)
    is_suspicious = anomaly_score >= 3.0

    return {
        "is_suspicious": is_suspicious,
        "anomaly_score": round(anomaly_score, 1),
        "flagged_dimensions": flags,
        "verdict": "异常" if anomaly_score >= 5 else ("可疑" if anomaly_score >= 3 else "正常"),
    }


def run_m2_analysis(video: dict) -> dict:
    """M2: ML 模式识别 — 互动模式分类 + 质量预测 + 异常检测

    使用纯规则引擎，无需外部 ML 依赖。
    返回包含 engagement, quality_prediction, anomaly 的字典。
    """
    c = video.get("count", {})

    engagement = _classify_engagement(
        max(c.get("count_view", 0), 1),
        c.get("count_like", 0),
        c.get("count_collect", 0),
        c.get("count_share", 0),
        c.get("count_comment", 0),
    )

    quality_pred = _predict_quality(video)
    anomaly = _detect_anomaly(video)

    return {
        "engagement": engagement,
        "quality_prediction": quality_pred,
        "anomaly": anomaly,
    }


# ============================================================
# L2 模块4：创作者风格分析
# ============================================================

def analyze_creator_profile(video: dict) -> dict:
    """M4: 创作者深度画像 — 基于多信号评分的导演层级分类

    从粉丝数、人气值、推荐率、获奖、团队规模等信号计算综合评分，
    输出创作者层级、风格标签和质量一致性评估。
    """
    author = video.get("author", {}) or {}
    ui = author.get("userinfo", {}) or {}
    username = ui.get("username", "")
    follower_count = ui.get("count_follower", 0) or 0
    article_count = ui.get("count_article", 0) or 1
    recommend_count = ui.get("count_recommend", 0) or 0
    popularity = ui.get("count_popularity", 0) or 0
    verify_desc = ui.get("verify_description", "")
    vip_flag = ui.get("vip_flag", 0) or 0
    occupation = ui.get("occupation", "") or author.get("occupation", "") or "创作者"

    team = video.get("team", []) or []
    team_len = len(team)

    # ---- 多信号评分 ----
    signals = {}

    # 1. 粉丝影响力（0-3分）
    if follower_count > 50000:
        signals["follower"] = 3.0
    elif follower_count > 10000:
        signals["follower"] = 2.0
    elif follower_count > 1000:
        signals["follower"] = 1.0
    else:
        signals["follower"] = 0.3

    # 2. 人气值（0-3分）
    if popularity > 5000000:
        signals["popularity"] = 3.0
    elif popularity > 1000000:
        signals["popularity"] = 2.0
    elif popularity > 100000:
        signals["popularity"] = 1.0
    else:
        signals["popularity"] = 0.5

    # 3. 推荐率（0-3分）
    recommend_rate = safe_div(recommend_count, article_count)
    if recommend_rate > 0.4:
        signals["recommend"] = 3.0
    elif recommend_rate > 0.2:
        signals["recommend"] = 2.0
    elif recommend_rate > 0.1:
        signals["recommend"] = 1.0
    else:
        signals["recommend"] = 0.3

    # 4. 获奖/认证信号（0-2分）
    award_score = 0.0
    prestige_kw = ["金狮奖", "金雀奖", "获奖", "代表作", "戛纳", "学院", "国际"]
    if any(kw in verify_desc for kw in prestige_kw):
        award_score += 1.0
    for member in team:
        if not isinstance(member, dict):
            continue
        mi = member.get("userinfo", {}) or {}
        jq = mi.get("count_jin_que", 0) or 0
        if jq > 0:
            award_score += min(jq, 3) * 0.3
        md = mi.get("verify_description", "") or ""
        if any(kw in md for kw in prestige_kw):
            award_score += 0.5
    signals["awards"] = min(award_score, 2.0)

    # 5. 团队规模（0-1分）
    signals["team"] = min(1.0, team_len / 15.0)

    # 6. VIP/认证标签（0-1分）
    signals["vip"] = min(1.0, vip_flag / 5.0)

    # 综合评分（满分13，归一化到0-10）
    total_score = sum(signals.values())
    tier_score = round(min(10.0, total_score / 13.0 * 10), 1)

    # 层级判定
    if tier_score >= 8.0:
        tier = "头部创作者"
        tier_label = "头部"
        tier_icon = "diamond"
    elif tier_score >= 5.5:
        tier = "腰部创作者"
        tier_label = "腰部"
        tier_icon = "gold"
    elif tier_score >= 3.0:
        tier = "上升创作者"
        tier_label = "上升"
        tier_icon = "rising"
    else:
        tier = "新兴创作者"
        tier_label = "新兴"
        tier_icon = "new"

    # 风格标签推断
    style_tags = []
    tags = [t.get("name", "") for t in (video.get("tags") or [])]
    tag_str = " ".join(tags)
    if any(kw in tag_str for kw in ["暗黑", "工业", "赛博", "科技"]):
        style_tags.append("科技工业风")
    if any(kw in tag_str for kw in ["暖", "自然", "唯美", "诗意"]):
        style_tags.append("自然诗意")
    if any(kw in tag_str for kw in ["快节奏", "碎切", "混剪", "节奏"]):
        style_tags.append("快剪节奏型")
    if any(kw in tag_str for kw in ["故事", "叙事", "纪录", "采访"]):
        style_tags.append("叙事型")
    if any(kw in tag_str for kw in ["动画", "CG", "三维", "特效"]):
        style_tags.append("数字特效")
    if any(kw in tag_str for kw in ["运动", "燃", "爆发", "力量"]):
        style_tags.append("运动活力")
    if not style_tags:
        cats = video.get("categories") or []
        sub = cats[0].get("sub", {}).get("category_name", "") if cats else ""
        style_tags = [sub] if sub else ["综合型"]

    # 质量一致性
    quality_consistency = round(recommend_rate / 0.5, 2) if recommend_rate > 0 else 0.3
    quality_consistency = min(1.0, quality_consistency)
    if quality_consistency >= 0.8:
        consistency_label = "高"
    elif quality_consistency >= 0.5:
        consistency_label = "中"
    else:
        consistency_label = "低"

    # 品类专精
    cats = video.get("categories") or []
    primary_cat = cats[0].get("category_name", "") if cats else ""
    sub_cat = cats[0].get("sub", {}).get("category_name", "") if cats else ""

    # 合作模式
    if team_len >= 15:
        collaboration_pattern = "大型团队协作"
    elif team_len >= 5:
        collaboration_pattern = "中小团队"
    else:
        collaboration_pattern = "独立/小团队"

    # 定价参考（基于层级和粉丝）
    if tier_label == "头部":
        pricing_reference = "15-50万/条"
    elif tier_label == "腰部":
        pricing_reference = "5-15万/条"
    elif tier_label == "上升":
        pricing_reference = "1-5万/条"
    else:
        pricing_reference = "5千-2万/条"

    # 洞察
    insights = []
    if follower_count > 50000:
        insights.append(f"粉丝基础雄厚 ({follower_count:,})")
    if recommend_rate > 0.3:
        insights.append("作品推荐率远高于平台均值")
    if vip_flag >= 3:
        insights.append("平台认证优质创作者")
    if not insights:
        insights.append("数据积累中，潜力有待观察")

    return {
        "username": username,
        "occupation": occupation,
        "tier": tier,
        "tier_label": tier_label,
        "tier_icon": tier_icon,
        "tier_score": tier_score,
        "style_tags": style_tags[:4],
        "quality_consistency": quality_consistency,
        "consistency_label": consistency_label,
        "stats": {
            "follower_count": follower_count,
            "article_count": article_count,
            "popularity": popularity,
        },
        "specialization": {
            "primary": primary_cat,
            "label": sub_cat,
        },
        "collaboration_pattern": collaboration_pattern,
        "pricing_reference": pricing_reference,
        "insights": insights,
        "signals": {
            "follower_count": follower_count,
            "popularity": popularity,
            "article_count": article_count,
            "recommend_count": recommend_count,
            "recommend_rate": round(recommend_rate, 3),
            "vip_flag": vip_flag,
        },
        "signal_scores": {k: round(v, 1) for k, v in signals.items()},
    }


# ============================================================
# L2 模块5：趋势与机会发现
# ============================================================

def _generate_trend_insight_llm(cat_dist, rising, provider, client):
    """用 LLM 生成更自然的一行趋势洞察"""
    cat_names = [c["category"] for c in cat_dist[:5]]
    rising_names = [r["username"] for r in rising[:5]]
    prompt = f"""你是一位视频平台数据分析师。根据以下搜索聚合数据，用一句话（30字内）概括当前搜索池的趋势特征。

品类分布: {", ".join(cat_names)}
上升创作者: {", ".join(rising_names) if rising_names else "无显著上升创作者"}

要求: 用单引号标记品类名称（如 '广告片'），以便前端加粗渲染。仅输出一句话，无其他文字。"""
    try:
        if provider == "deepseek":
            text = _call_deepseek(client, prompt, L2_SYSTEM_PROMPT)
        else:
            text = _call_anthropic(client, prompt, L2_SYSTEM_PROMPT)
        return text.strip().strip('"').strip("'")
    except Exception:
        return None


def detect_search_trends(items: list, emb_idx=None) -> dict:
    """M5: 从搜索结果中检测品类趋势和上升创作者

    对搜索结果做聚合分析，输出品类分布、潜在上升创作者。
    """
    if not items:
        return {"category_distribution": [], "rising_creators": [], "total_analyzed": 0}

    # 品类分布统计
    cat_counter = {}
    author_stats = {}

    for item in items:
        cats = item.get("categories") or []
        main_cat = cats[0].get("category_name", "其他") if cats else "其他"
        sub_cat = cats[0].get("sub", {}).get("category_name", "") if cats else ""
        cat_key = f"{main_cat}>{sub_cat}" if sub_cat else main_cat
        cat_counter[cat_key] = cat_counter.get(cat_key, 0) + 1

        author = item.get("author", {}) or {}
        ui = author.get("userinfo", {}) or {}
        username = ui.get("username", "")
        if username:
            if username not in author_stats:
                c = item.get("count", {})
                author_stats[username] = {
                    "username": username,
                    "follower_count": ui.get("count_follower", 0) or 0,
                    "article_count": ui.get("count_article", 0) or 0,
                    "recommend_count": ui.get("count_recommend", 0) or 0,
                    "popularity": ui.get("count_popularity", 0) or 0,
                    "appearances": 0,
                    "total_views": 0,
                    "total_collects": 0,
                }
            c = item.get("count", {})
            author_stats[username]["appearances"] += 1
            author_stats[username]["total_views"] += c.get("count_view", 0)
            author_stats[username]["total_collects"] += c.get("count_collect", 0)

    # 品类分布排序
    cat_dist = sorted(
        [{"category": k, "count": v, "pct": round(v / len(items) * 100, 1)} for k, v in cat_counter.items()],
        key=lambda x: x["count"], reverse=True
    )[:8]

    # 上升创作者识别
    rising = []
    for username, stats in author_stats.items():
        if stats["appearances"] < 2:
            continue
        recommend_rate = safe_div(stats["recommend_count"], max(stats["article_count"], 1))
        avg_views = safe_div(stats["total_views"], stats["appearances"])
        rising_score = 0.0
        if recommend_rate > 0.3:
            rising_score += 3
        elif recommend_rate > 0.15:
            rising_score += 1.5
        if avg_views > 50000:
            rising_score += 3
        elif avg_views > 10000:
            rising_score += 1.5
        if stats["follower_count"] < 5000 and stats["popularity"] > 100000:
            rising_score += 2
        if rising_score >= 3:
            rising.append({
                "username": username,
                "rising_score": round(rising_score, 1),
                "recommend_rate": round(recommend_rate, 3),
                "avg_views": int(avg_views),
            })

    rising.sort(key=lambda x: x["rising_score"], reverse=True)

    # 生成洞察文本
    insight_parts = []
    if cat_dist:
        top_names = [f"'{c['category']}'({c['pct']}%)" for c in cat_dist[:3]]
        insight_parts.append(f"品类分布: {', '.join(top_names)}")
    if rising:
        names = '、'.join(r['username'] for r in rising[:3])
        insight_parts.append(f"发现 {len(rising)} 位上升创作者 (如 {names})")
    insight_text = "；".join(insight_parts) if insight_parts else "暂无显著趋势信号"

    # 尝试 LLM 生成更丰富的洞察（失败则用规则文本）
    provider, client = _get_llm_client()
    if provider and client and len(cat_dist) >= 2:
        try:
            insight_text = _generate_trend_insight_llm(cat_dist, rising, provider, client) or insight_text
        except Exception:
            pass

    return {
        "category_distribution": cat_dist,
        "rising_creators": rising[:5],
        "total_analyzed": len(items),
        "insight_text": insight_text,
    }


# ============================================================
# L2 模块6：跨维度综合推理
# ============================================================

def build_m6_synthesis(video: dict, l1_scores: dict = None, l2_metadata: dict = None,
                       m2_results: dict = None, similar_works: list = None,
                       creator_profile: dict = None, trends: dict = None) -> str:
    """M6: 融合所有模块输出，生成自然语言综合洞察

    基于模板引擎将 L1/L2/M2/M4/M5 的所有信号融合为一段可读的
    综合点评，揭示作品的核心竞争力和商业参考价值。
    """
    title = video.get("title", "")
    duration = video.get("duration", 0)

    # 安全取值
    l1 = l1_scores or {}
    l2 = l2_metadata or {}
    m2 = m2_results or {}
    cp = creator_profile or {}

    # ---- 构建各部分陈述 ----

    # 1. 开场定性
    brand = l2.get("brand", "") or "该作品"
    brand_tier = l2.get("brand_tier", "")
    budget_tier = l2.get("budget_tier", "")
    commercial_type = l2.get("commercial_type", "")
    engagement_type = ""
    if m2:
        engagement_type = m2.get("engagement", {}).get("type", "")

    tier_text = f"({brand_tier})" if brand_tier else ""
    opening = f"《{title}》" if title else "该作品"
    opening += f"是一部{commercial_type}" if commercial_type else ""
    opening += f"，属于{engagement_type}" if engagement_type else ""

    # 2. 视觉与情绪
    visual_style = l2.get("visual_style", "")
    mood = l2.get("mood", "")
    style_keywords = l2.get("style_keywords", [])
    vis_text = ""
    if visual_style:
        vis_text = f"其视觉公式为「{visual_style}」"
    if mood:
        vis_text += f"，情绪基调「{mood}」"
    if style_keywords:
        vis_text += f"，关键风格标签: {', '.join(style_keywords[:4])}"

    # 3. L1 数据佐证
    d1 = l1.get("D1_audience_reception", {})
    d2 = l1.get("D2_commercial_value", {})
    d3 = l1.get("D3_team_professionalism", {})
    d1_score = d1.get("score", 0) if isinstance(d1, dict) else 0
    d2_score = d2.get("score", 0) if isinstance(d2, dict) else 0
    d3_score = d3.get("score", 0) if isinstance(d3, dict) else 0

    scores_text = f"L1 综合评分: 观众接受度{d1_score}/专业度{d3_score}/商业价值{d2_score}"

    # 4. M2 信号
    m2_text = ""
    if m2:
        quality_pred = m2.get("quality_prediction", {})
        anomaly = m2.get("anomaly", {})
        if isinstance(quality_pred, dict):
            m2_text = f"ML质量预测{quality_pred.get('predicted_score', 'N/A')}/10"
        if isinstance(anomaly, dict) and anomaly.get("is_suspicious"):
            m2_text += " [异常检测: 可疑]"

    # 5. 创作者
    creator_text = ""
    if cp:
        c_tier = cp.get("tier", "")
        c_name = cp.get("username", "")
        c_styles = cp.get("style_tags", [])
        if c_name and c_tier:
            creator_text = f"创作者 {c_name} 为{c_tier}"
            if c_styles:
                creator_text += f"，擅长 {'/'.join(c_styles[:3])}"

    # 6. 相似作品
    similar_text = ""
    if similar_works:
        top_similar = [f"《{s.get('title', '')}》(相似度{s.get('similarity', 0)})" for s in similar_works[:3]]
        if top_similar:
            similar_text = f"相似作品: {'、'.join(top_similar)}"

    # 7. 趋势
    trend_text = ""
    if trends:
        rising = trends.get("rising_creators", [])
        if rising:
            trend_text = f"搜索池中发现 {len(rising)} 位上升创作者"

    # ---- 组合最终输出 ----
    parts = [opening.strip("。")]
    if vis_text:
        parts.append(vis_text)
    parts.append(scores_text)
    if m2_text:
        parts.append(m2_text)
    if creator_text:
        parts.append(creator_text)
    if budget_tier:
        parts.append(f"制作级别: {budget_tier}")

    if similar_text:
        parts.append(similar_text)
    if trend_text:
        parts.append(trend_text)

    synthesis = "。".join(p for p in parts if p) + "。"

    # 生成多维度洞察
    insights = []
    if engagement_type and engagement_type not in ("普通型", "数据不足"):
        insights.append(f"该作品属于平台定义的「{engagement_type}」作品，对广告商的参考价值高于均值")
    if isinstance(d2.get("score", 0), (int, float)) and d2.get("score", 0) > 8.0 if isinstance(d2, dict) else False:
        insights.append("商业参考价值极高，适合作为品类提案对标案例")
    if cp and cp.get("tier") in ("头部创作者", "腰部创作者"):
        insights.append(f"创作者层级为{cp.get('tier')}，作品质量有持续保障")
    if not insights:
        insights.append("建议上传视频文件以获取深度视听审计")

    return {
        "synthesis": synthesis,
        "insights": insights,
    }


# ============================================================
# LLM 客户端
# ============================================================

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


def _get_llm_client():
    """自动检测 API Key，返回 (provider, client) 或 (None, None)"""
    ds_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if ds_key:
        try:
            from openai import OpenAI
            return ("deepseek", OpenAI(api_key=ds_key, base_url="https://api.deepseek.com"))
        except ImportError:
            pass

    ant_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if ant_key:
        try:
            from anthropic import Anthropic
            return ("anthropic", Anthropic(api_key=ant_key))
        except ImportError:
            pass

    return (None, None)


def _call_deepseek(client, user_prompt: str, system_prompt: str) -> str:
    """调用 DeepSeek API（OpenAI 兼容接口）"""
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.7,
        max_tokens=4096,
    )
    return resp.choices[0].message.content


def _call_anthropic(client, user_prompt: str, system_prompt: str) -> str:
    """调用 Anthropic Claude API"""
    resp = client.messages.create(
        model="claude-sonnet-4-6-20250514",
        max_tokens=4096,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return resp.content[0].text


DEFAULT_SYSTEM_PROMPT = "你是一位拥有20年经验的顶级4A公司创意总监、戛纳广告奖评委。你的点评必须严谨、犀利、专业，使用标准广告行业术语。只输出JSON数组，不输出markdown和任何解释文字。"

L2_SYSTEM_PROMPT = "你是一位广告片分析专家，擅长从元数据中提取结构化信息。你必须严格按JSON格式输出，不输出任何解释文字或markdown。"


# ============================================================
# LLM 批量评价 & 本地摘要回退
# ============================================================

def evaluate_batch_with_llm(videos: list, industry: str, style_preference: str,
                            mode: str = "batch", l2_metadata: dict = None) -> list:
    """调用 LLM 批量评价视频，返回 summary + key_elements 列表。
    mode="ecd" 时返回单条 ECD 审计报告字符串列表。"""
    provider, client = _get_llm_client()
    if not provider or not client:
        return None

    if mode == "ecd" and len(videos) == 1:
        prompt = build_ecd_audit_prompt(videos[0], industry, style_preference, l2_metadata)
        try:
            if provider == "deepseek":
                text = _call_deepseek(client, prompt, DEFAULT_SYSTEM_PROMPT)
            else:
                text = _call_anthropic(client, prompt, DEFAULT_SYSTEM_PROMPT)
            text = text.strip()
            if text.startswith("```"):
                text = re.sub(r'^```\w*\n?', '', text)
                text = re.sub(r'\n?```$', '', text)
            data = json.loads(text)
            if isinstance(data, list) and len(data) > 0:
                return data[0]  # 返回 dict 而非 list
            return data
        except Exception as e:
            print(f"[batch_llm ecd] Failed: {e}", file=sys.stderr)
            return None

    prompt = build_batch_prompt(videos, industry, style_preference)
    try:
        if provider == "deepseek":
            text = _call_deepseek(client, prompt, DEFAULT_SYSTEM_PROMPT)
        else:
            text = _call_anthropic(client, prompt, DEFAULT_SYSTEM_PROMPT)

        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r'^```\w*\n?', '', text)
            text = re.sub(r'\n?```$', '', text)
        results = json.loads(text)
        if isinstance(results, list):
            return results
    except Exception as e:
        print(f"[batch_llm] Failed: {e}", file=sys.stderr)
    return None


def generate_local_summary(video: dict, industry: str = ""):
    """本地规则生成 summary + key_elements，无需 LLM"""
    title = video.get("title", "") or ""
    duration = video.get("duration", 0) or 0
    tags = [t.get("name", "") for t in (video.get("tags") or [])]
    cats = video.get("categories") or []
    cat_name = cats[0].get("category_name", "") if cats else ""

    # 根据标签和分类推断风格词
    style_hints = {
        "快剪": "快节奏剪辑", "延时": "延时摄影", "航拍": "航拍视角",
        "动画": "动画表现", "CG": "CG特效", "剧情": "故事化叙事",
        "产品": "产品展示", "TVC": "TVC质感", "VLOG": "Vlog纪实",
        "混剪": "混剪手法", "黑白": "黑白影调", "慢镜头": "慢镜表现",
        "赛博朋克": "赛博朋克", "复古": "复古胶片", "国风": "国风美学",
    }
    key_elements = []
    for tag in tags:
        for k, v in style_hints.items():
            if k in tag and v not in key_elements:
                key_elements.append(v)

    if not key_elements:
        if cat_name:
            key_elements.append(f"{cat_name}风格")
        if duration < 30:
            key_elements.append("短平快")
        elif duration > 180:
            key_elements.append("深度叙事")
        key_elements.append("商业制作")

    industry_str = f"适用于{industry}行业，" if industry else ""
    cat_str = f"「{cat_name}」品类" if cat_name else "视频"
    summary = f"{industry_str}{cat_str}，{duration}秒{'短片' if duration < 60 else '中长片'}，{'、'.join(key_elements[:2])}。"

    return summary, key_elements[:5]


def evaluate_uploaded_video(video_path: str, metadata: dict = None,
                            industry: str = "", style: str = "") -> dict:
    """Gemini 多模态视频分析 — 直接传视频文件给 Gemini 做真实视听审计。

    DeepSeek / Anthropic 均不支持视频/图片输入，无法做视频分析。
    未配置 GEMINI_API_KEY 时直接返回不可用提示，不做假分析。
    """
    import os as _os
    file_size_mb = _os.path.getsize(video_path) / 1024 / 1024
    file_info = f"{file_size_mb:.1f}MB"

    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    if not gemini_key:
        return {
            "🚨 叙事效率预警": f"视频已上传（{file_info}），升级 VIP 后可使用视频分析功能。",
            "💬 商业提案 PPT 话术直通车": [
                {"针对同品类/硬核性能客户提案": "升级 VIP 后可使用视频分析。"},
                {"针对跨品类平移提案": "升级 VIP 后可使用视频分析。"}
            ],
            "_source": "no_key"
        }

    try:
        import google.generativeai as genai
    except ImportError:
        return {
            "🚨 叙事效率预警": "视频分析功能已上线，升级 VIP 后即可使用。",
            "💬 商业提案 PPT 话术直通车": [
                {"针对同品类/硬核性能客户提案": "升级 VIP 后可使用视频分析。"},
                {"针对跨品类平移提案": "升级 VIP 后可使用视频分析。"}
            ],
            "_source": "sdk_missing"
        }

    detail = (metadata or {}).get("detail", {})
    l2_metadata = (metadata or {}).get("l2_metadata")

    try:
        genai.configure(api_key=gemini_key)
        model = genai.GenerativeModel(
            model_name="gemini-1.5-pro",
            system_instruction=DEFAULT_SYSTEM_PROMPT
        )
        video_file = genai.upload_file(path=video_path)
        while video_file.state.name == "PROCESSING":
            time.sleep(2)
            video_file = genai.get_file(video_file.name)

        title = detail.get("title", "") or "未知视频"
        tags = [t.get("name", "") for t in (detail.get("tags") or [])]

        l2_context = ""
        if l2_metadata:
            l2_context = "\n".join(
                f"- {k}: {v}" for k, v in l2_metadata.items()
                if v and not k.startswith("_") and isinstance(v, str)
            )

        prompt = f"""请对以下上传视频进行商业初筛点评。
视频标题: {title}
视频标签: {', '.join(tags)}
文件信息: {file_info}
行业: {industry or '通用'} / 风格偏好: {style or '不限'}
L2 预分析: {l2_context or '无'}

输出严格 JSON:
[{{
  "🚨 叙事效率预警": "...",
  "💬 商业提案 PPT 话术直通车": [
    {{"针对同品类/硬核性能客户提案": "..."}},
    {{"针对跨品类平移提案": "..."}}
  ]
}}]"""

        response = model.generate_content([video_file, prompt])
        text = response.text.strip()
        if text.startswith("```"):
            text = re.sub(r'^```\w*\n?', '', text)
            text = re.sub(r'\n?```$', '', text)
        data = json.loads(text)
        result = data[0] if isinstance(data, list) and len(data) > 0 else data
        result["_source"] = "gemini_multimodal"
        return result
    except Exception as e:
        print(f"[upload] Gemini analysis failed: {e}", file=sys.stderr)
        return {
            "🚨 叙事效率预警": f"视频已上传（{file_info}），分析服务暂时不可用，请稍后重试。",
            "💬 商业提案 PPT 话术直通车": [
                {"针对同品类/硬核性能客户提案": "升级 VIP 后可使用视频分析。"},
                {"针对跨品类平移提案": "升级 VIP 后可使用视频分析。"}
            ],
            "_source": "error"
        }


# ============================================================
# L3 级视频深度分析 (Qwen3.5-Omni-Plus)
# ============================================================

# C-Score Matrix 评价维度 + C-Eye 商业初筛报告格式
L3_ANALYSIS_PROMPT = '''你是一位拥有20年经验、曾获戛纳金狮奖的顶级4A公司创意总监。你对视觉美学、剪辑节奏、心理博弈和营销转化有着近乎偏执的高标准。

请作为专业拉片工具，对上传的视频进行全方位、量化的多模态分析。你必须同时处理视频的【画面时序】和【音频轨道】。

# C-Eye 商业初筛画像深度报告

请严格按照以下结构输出 Markdown 格式的分析报告：

## 一、C-Score Matrix 评分

请从以下四个维度进行评分（各25分，满分100分）：

### Hook (前3秒) - /25
* 视觉冲击力评估
* 悬念设置分析
* "静音播放"下的吸引力判断

### Rhythm (节奏) - /25
* 转场是否卡点？
* 视觉信息流是否符合人类认知的"心流"？
* 剪辑节奏的"去耐心化"程度

### Brand (品牌) - /25
* 颜色、Logo、Slogan的植入是否自然且强势？
* 品牌产品首次出现时间、总露出次数和时长占比

### Conversion (转化) - /25
* 卖点（USP）是否清晰？
* 是否有强烈的行动导向？

**总分: X/100**

---

## 二、镜头量化统计
* 计算并给出总镜头数、平均镜头时长、镜头时长中位数
* 分析景别分布（特写、中景、大远景占比）与运镜风格
* 品牌产品或核心意象的首次出现时间、总露出次数和总时长占比

## 三、色板量化提取
* 给出主色调板：提取全片占比最高的前 3-4 种核心色值（十六进制 HEX），并说明其视觉功能
* 分析色温变化走向，并推荐适合该片调性的电影 LUT 风格

## 四、声音分析深化
* 评估音频整体响度与动态范围（如 Bass Drop 等关键锚点的时间戳）
* 拆解音频结构（环境音、BGM编曲风格、节奏BPM）
* 精准识别并转录片中的核心语音旁白或歌词宣言

## 五、导演风格与艺术演进
* 结合画面呈现，分析本片在叙事、剪辑上的艺术风格
* 黄金3秒抓手分析：开篇用怎样的视觉策略卡位？

## 六、创作技巧分级拆解
* 🟢 初级技巧（普通人可立即复刻的构图、色彩）
* 🟡 中级技巧（依赖剪辑技巧、Match Cut 转场）
* 🔴 高级技巧（高预算置景、复杂工业级运镜）

## 七、观众心理学与传播预测
* 绘制受众注意力与认知负荷曲线的时间节点变化
* 预测社交媒体上传播的 3 个核心记忆锚点（产品、奇观、情感）
* 平台投放定位：适合哪些渠道（信息流/分众/线下大屏）？

## 八、预算成色与执行避坑
* 预算级别估算（S/A/B/C级）
* 低成本平移方案：如果预算有限，哪些核心技法是必须保住的？

## 九、犀利点评与优化建议
* **critical_review**: 50字以内的犀利点评
* **optimization_plan**: 3条具体的修改建议（例如：将00:12的特写提前到开头）

## 十、结构化数据标签
最后，请输出 JSON 块：
```json
{
  "total_score": 0-100,
  "breakdown": {"hook": X, "rhythm": X, "brand": X, "conversion": X},
  "total_shots": "总镜头数",
  "avg_shot_duration": "平均镜头时长（秒）",
  "shot_distribution": {"特写": "xx%", "中景": "xx%", "远景": "xx%"},
  "color_palette": ["#HEX1", "#HEX2", "#HEX3"],
  "bpm": "节奏BPM",
  "style_tags": ["风格标签1", "风格标签2"],
  "budget_level": "S/A/B/C",
  "critical_review": "犀利点评",
  "optimization_plan": ["建议1", "建议2", "建议3"]
}
```
'''


def analyze_video_with_qwen(video_path: str, industry: str = "", style: str = "",
                             timeout: int = 600) -> dict:
    """使用 Qwen 多模态模型直接分析视频文件。

    通过 DashScope 原生 MultiModalConversation API，用 file:// 协议直传视频，
    无需 Base64 编码，支持最大 500MB 视频。

    Args:
        video_path: 视频文件路径
        industry: 行业背景（可选）
        style: 风格偏好（可选）
        timeout: 超时时间（秒）

    Returns:
        dict: 包含 status, report_json, report_md, error 等字段
    """
    import os as _os

    # 检查 API Key
    dashscope_key = os.environ.get("DASHSCOPE_API_KEY", "")
    if not dashscope_key:
        return {
            "status": "failed",
            "error": "未配置 DASHSCOPE_API_KEY。\n请在 .env 文件中设置: DASHSCOPE_API_KEY=your_key\n\n获取方式: https://www.alibabacloud.com/help/zh/model-studio/get-api-key",
            "report_md": None,
            "report_json": None
        }

    # 检查文件
    if not _os.path.exists(video_path):
        return {
            "status": "failed",
            "error": f"视频文件不存在: {video_path}",
            "report_md": None,
            "report_json": None
        }

    file_size_mb = _os.path.getsize(video_path) / 1024 / 1024
    if file_size_mb > 500:
        return {
            "status": "failed",
            "error": f"视频文件过大 ({file_size_mb:.1f}MB)，最大支持 500MB",
            "report_md": None,
            "report_json": None
        }

    try:
        from dashscope import MultiModalConversation
    except ImportError:
        return {
            "status": "failed",
            "error": "需要安装 dashscope 库: pip install dashscope",
            "report_md": None,
            "report_json": None
        }

    print(f"[L3] Starting video analysis for {video_path} ({file_size_mb:.1f}MB)")

    # 构建提示词
    context_parts = []
    if industry:
        context_parts.append(f"行业背景: {industry}")
    if style:
        context_parts.append(f"风格偏好: {style}")

    full_prompt = L3_ANALYSIS_PROMPT
    if context_parts:
        full_prompt += f"\n\n---\n视频背景信息：\n" + "\n".join(context_parts)

    try:
        local_file_path = f"file://{video_path}"

        messages = [
            {
                "role": "system",
                "content": [{"text": "你是一位拥有20年经验、曾获戛纳金狮奖的顶级4A公司创意总监。你对视觉美学、剪辑节奏、心理博弈和营销转化有着近乎偏执的高标准。只输出分析报告，不输出任何解释文字。"}]
            },
            {
                "role": "user",
                "content": [
                    {"video": local_file_path},
                    {"text": full_prompt}
                ]
            }
        ]

        print(f"[L3] Calling DashScope MultiModalConversation with video...")
        response = MultiModalConversation.call(
            model="qwen3.5-omni-plus",
            messages=messages,
        )

        if response.status_code != 200:
            return {
                "status": "failed",
                "error": f"API 返回错误: code={response.status_code}, message={response.message}",
                "report_md": None,
                "report_json": None
            }

        # 解析 MultiModalConversation 响应
        output = response.output
        if output and output.choices:
            content = output.choices[0].message.content
            if isinstance(content, list):
                report_md = ""
                for part in content:
                    if isinstance(part, dict) and "text" in part:
                        report_md += part["text"]
                    elif isinstance(part, str):
                        report_md += part
            elif isinstance(content, str):
                report_md = content
            else:
                report_md = str(content)
        else:
            return {
                "status": "failed",
                "error": "API 返回内容为空",
                "report_md": None,
                "report_json": None
            }

        report_md = report_md.strip()
        print(f"[L3] API analysis completed ({len(report_md)} chars)")

        report_json = extract_json_from_markdown(report_md)

        return {
            "status": "completed",
            "error": None,
            "report_md": report_md,
            "report_json": report_json,
            "model": "qwen3.5-omni-plus",
            "file_size_mb": round(file_size_mb, 1),
            "method": "dashscope_multimodal"
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "status": "failed",
            "error": f"视频分析失败: {str(e)}",
            "report_md": None,
            "report_json": None
        }


def extract_json_from_markdown(report_md: str) -> dict:
    """从 Markdown 报告中提取 JSON 块"""
    import json as _json

    if not report_md:
        return {}

    # 尝试匹配 ```json 块
    json_match = re.search(r'```json\s*([\s\S]*?)\s*```', report_md)
    if json_match:
        try:
            return _json.loads(json_match.group(1))
        except:
            pass

    # 尝试匹配纯 JSON 对象
    json_match = re.search(r'\{[\s\S]*"total_score"[\s\S]*\}', report_md)
    if json_match:
        try:
            return _json.loads(json_match.group(0))
        except:
            pass

    return {
        "analysis_source": "qwen3.6-plus",
        "note": "JSON extraction failed, please check report_md"
    }
