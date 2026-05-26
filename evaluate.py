# -*- coding: utf-8 -*-
"""
视频评价引擎：多维元数据评分 (D1-D4) + LLM 总结
基于元数据，无需视频文件
"""
import json, math, re, os, sys
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

        visual_style = "标准商业调性"
        tag_str = "".join(tags)
        if any(x in tag_str for x in ["胶片", "复古", "老电影"]):
            visual_style = "胶片复古风"
        elif any(x in tag_str for x in ["赛博", "科技", "硬核"]):
            visual_style = "赛博数字重金属"
        elif any(x in tag_str for x in ["极简", "高级", "冷淡"]):
            visual_style = "高智感都市冷淡风"
        elif any(x in tag_str for x in ["CG", "三维", "特效"]):
            visual_style = "超现实数字资产"

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

        if vip_level >= 3 or team_len > 10:
            budget_class = "A级 (百万级大制作)"
            soul_part = "导演组/美术置景"
        elif team_len > 4:
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


def build_ecd_audit_prompt(video: dict, industry: str, style_preference: str) -> str:
    """构造 4A ECD 风格的商业审计 prompt"""
    tags = [t.get("name", "") for t in video.get("tags", [])]
    content = (video.get("content", "") or "")[:800]

    # 提取基本 L1 属性用于 Context
    l1_data = c_eye_l1_fast_scorer([video], industry)[0] if video else {}

    prompt = f"""你是一个在 4A 广告公司（如奥美、BBDO）摸爬滚打 15 年的顶尖创意总监（ECD）兼商业制片人。你眼光极其毒舌、犀利、一针见血，深谙广告主（甲方）的各种商业痛点，同时对移动端流媒体（抖音、小红书、B站）的流量密码和转化率了如指掌。

请根据以下输入，从纯粹的"商业落地、提案交付"视角，对该影视作品进行多维度的商业初筛点评。

【输入 Context】
- 作品名称: {video.get('title', '')}
- 算法推荐阵列: {l1_data.get('tab_category', '常规商业流')}
- 清洗后核心标签: {', '.join(l1_data.get('dimensions', {}).get('synesthesia', []))}
- 预算估级参考: {l1_data.get('dimensions', {}).get('budget', 'B级')}
- 目标行业/风格: {industry or '通用'} / {style_preference or '不限'}
- 导演/团队创作原述: {content}

【输出规则】
1. 拒绝任何"构图精美"、"演技在线"、"弘扬文化"等毫无商业参考价值的空洞废话。
2. 语言风格：必须使用广告圈、影视圈地道黑话（如：Hook、抓手、心智、调性、平移、提案爆款、痛点、下沉、分镜、起承转合、视觉轰炸）。
3. 态度：保持冷酷、专业、客观，既要一语道破其最大的"可抄资产"，也要毫不留情地指出其"落地转化风险"。
4. 字数限制：整体点评控制在 300 字内。

【响应格式】(必须严格按下述 Markdown 格式输出)

### 🚨 叙事效率预警 (Efficiency Alert)
[此处点评"黄金前3秒"的吸睛抓手（Hook）强不强。明确指出平移到"抖音/小红书/B站信息流"或"分众电梯媒体"投放时的品牌风险与前置转化率预测。]

### 💬 商业提案 PPT 话术直通车 (Pitching Keywords)
1. **针对同品类/硬核性能客户提案：** "本方案将参考本片【此处结合视频标签填入核心视觉资产词】，利用【此处填入视效/剪辑特征】在开篇3秒内剥夺用户注意力……"
2. **针对跨品类平移（如将调性平移给{industry or '跨品类'}）提案：** "我们打破常规，尝试将本片特有的【情绪词】跨界注入到本次品类中，用高级的【调性词】为品牌筑起护城河……"
"""
    return prompt


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


def _call_deepseek(client, prompt: str) -> str:
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
    resp = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=3072,
        system="你是一位拥有20年经验的顶级4A公司创意总监、戛纳广告奖评委。你的点评必须犀利、专业，使用标准广告行业术语。只输出JSON数组，不输出markdown和任何解释文字。",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        timeout=60,
    )
    return resp.content[0].text.strip()


def evaluate_batch_with_llm(videos: list, industry: str = "", style_preference: str = "", mode: str = "batch") -> any:
    """批量调 LLM API 生成评价"""
    provider, client = _get_llm_client()
    if provider is None or client is None:
        return None

    if mode == "ecd" and len(videos) == 1:
        prompt = build_ecd_audit_prompt(videos[0], industry, style_preference)
    else:
        prompt = build_batch_prompt(videos, industry, style_preference)

    try:
        if provider == "deepseek":
            text = _call_deepseek(client, prompt)
        else:
            text = _call_anthropic(client, prompt)
    except Exception as e:
        print(f"[LLM] {provider} API call failed: {e}", file=sys.stderr)
        return None

    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r'^```\w*\n?', '', text)
        text = re.sub(r'\n?```$', '', text)

    if mode == "ecd":
        return text

    try:
        results = json.loads(text)
        if isinstance(results, list) and len(results) == len(videos):
            return results
        return None
    except json.JSONDecodeError:
        return None


# ============================================================
# 简易本地摘要生成（无LLM时的fallback）
# ============================================================

def _extract_brand(title: str) -> str:
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
    collect_rate = safe_div(c.get("count_collect", 0), max(1, views))

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
    if safe_div(likes, max(1, views)) > 0.015:
        tags.append("高赞内容")

    api_tags = [t.get("name", "") for t in video.get("tags", [])]
    for t in api_tags:
        if t not in tags and len(tags) < 6:
            tags.append(t)

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

    parts = []
    if brand:
        parts.append(brand)
    parts.append(category)

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
