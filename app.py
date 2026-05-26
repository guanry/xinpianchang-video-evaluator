# -*- coding: utf-8 -*-
"""
视频创意 AI 评价平台 — Flask Web 应用
"""
import json, time, os, sys
from flask import Flask, request, jsonify, render_template, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from flask_bcrypt import Bcrypt

# Fix Windows console encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except:
        pass

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Change this in production
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

from models import db, User, Favorite, Comment

bcrypt = Bcrypt(app)
db.init_app(app)
login_manager = LoginManager(app)
login_manager.login_view = 'index'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Create database tables
with app.app_context():
    db.create_all()

# 加载筛选器配置
FILTERS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "search_filters.json")
SEARCH_FILTERS = {}
if os.path.exists(FILTERS_FILE):
    with open(FILTERS_FILE, "r", encoding="utf-8") as f:
        SEARCH_FILTERS = json.load(f)

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
    c_eye_l1_fast_scorer,
    build_evaluation_prompt,
    evaluate_batch_with_llm,
    generate_local_summary,
    score_all_dimensions,
    compute_scenario_scores,
    build_full_evaluation,
    extract_l2_metadata,
    run_m2_analysis,
    analyze_creator_profile,
    detect_search_trends,
    build_m6_synthesis,
)
from embedding import get_index
from scrape_video import get_article, format_detail, search_videos as scrape_search
import requests

SEARCH_API = "https://www.xinpianchang.com/api/xpc/v2/search"
DETAIL_API = "https://www.xinpianchang.com/api/xpc/v2/article"
CACHE = {}
CACHE_TTL = 600  # 10 min，对齐上游缓存

HEADERS = {"User-Agent": "Mozilla/5.0"}


def search_videos(keyword: str, page: int = 1, filters: dict = None) -> list:
    """搜索视频，返回list，支持分类/时长/比例等筛选"""
    filter_key = json.dumps(filters, sort_keys=True) if filters else ""
    cache_key = f"search:{keyword}:{page}:{filter_key}"
    now = time.time()
    if cache_key in CACHE and CACHE[cache_key]["ts"] > now - CACHE_TTL:
        return CACHE[cache_key]["data"]

    # 使用 scrape_video 中的新 v2 API 搜索
    cate_id = filters.get("cate_id") if filters else None
    duration = filters.get("duration") if filters else None
    screen_type = filters.get("screen_type") if filters else None

    try:
        items = scrape_search(
            keyword=keyword, page=page,
            cate_id=str(cate_id) if cate_id else None,
            duration=duration,
            screen_type=str(screen_type) if screen_type else None,
        )
    except Exception as e:
        print(f"[search_videos] scrape_search failed: {e}")
        items = []

    # 客户端二次过滤
    if filters and items:
        client_filters = {}
        if filters.get("duration"):
            parts = str(filters["duration"]).split(",")
            if len(parts) == 2:
                client_filters["duration_min"] = int(parts[0])
                client_filters["duration_max"] = int(parts[1])
        if filters.get("screen_type"):
            client_filters["screen_type"] = int(filters["screen_type"])

        if client_filters:
            filtered = []
            for v in items:
                dur = v.get("duration", 0)
                st = v.get("screen_type")
                if "duration_min" in client_filters and dur < client_filters["duration_min"]:
                    continue
                if "duration_max" in client_filters and dur > client_filters["duration_max"]:
                    continue
                if "screen_type" in client_filters and st != client_filters["screen_type"]:
                    continue
                filtered.append(v)
            items = filtered

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

@app.route("/api/register", methods=["POST"])
def api_register():
    data = request.json
    username = data.get("username")
    password = data.get("password")
    
    if not username or not password:
        return jsonify({"error": "请输入用户名和密码"}), 400
    
    if User.query.filter_by(username=username).first():
        return jsonify({"error": "用户名已存在"}), 400
    
    hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
    new_user = User(username=username, password_hash=hashed_password)
    db.session.add(new_user)
    db.session.commit()
    
    return jsonify({"message": "注册成功"})

@app.route("/api/login", methods=["POST"])
def api_login():
    data = request.json
    username = data.get("username")
    password = data.get("password")
    
    user = User.query.filter_by(username=username).first()
    if user and bcrypt.check_password_hash(user.password_hash, password):
        login_user(user)
        return jsonify({"message": "登录成功", "user": {"username": user.username, "is_member": user.is_member}})
    
    return jsonify({"error": "用户名或密码错误"}), 401

@app.route("/api/logout", methods=["POST"])
@login_required
def api_logout():
    logout_user()
    return jsonify({"message": "已退出登录"})

@app.route("/api/me")
def api_me():
    if current_user.is_authenticated:
        return jsonify({
            "authenticated": True,
            "username": current_user.username,
            "is_member": current_user.is_member
        })
    return jsonify({"authenticated": False})

@app.route("/api/upgrade", methods=["POST"])
@login_required
def api_upgrade():
    # Simple membership upgrade simulation
    current_user.is_member = True
    db.session.commit()
    return jsonify({"message": "会员升级成功", "is_member": True})

@app.route("/api/favorites", methods=["GET"])
@login_required
def api_get_favorites():
    favs = Favorite.query.filter_by(user_id=current_user.id).order_by(Favorite.created_at.desc()).all()
    results = []
    for f in favs:
        try:
            data = json.loads(f.video_data)
            data['is_favorite'] = True
            results.append(data)
        except:
            results.append({
                "id": f.video_id,
                "title": f.video_title,
                "cover": f.video_cover,
                "is_favorite": True
            })
    return jsonify({"videos": results})

@app.route("/api/favorites/add", methods=["POST"])
@login_required
def api_add_favorite():
    data = request.json
    video_id = str(data.get("id"))
    
    if Favorite.query.filter_by(user_id=current_user.id, video_id=video_id).first():
        return jsonify({"message": "已在收藏夹中"})
    
    new_fav = Favorite(
        user_id=current_user.id,
        video_id=video_id,
        video_title=data.get("title"),
        video_cover=data.get("cover"),
        video_data=json.dumps(data)
    )
    db.session.add(new_fav)
    db.session.commit()
    return jsonify({"message": "收藏成功"})

@app.route("/api/favorites/remove", methods=["POST"])
@login_required
def api_remove_favorite():
    data = request.json
    video_id = str(data.get("id"))
    fav = Favorite.query.filter_by(user_id=current_user.id, video_id=video_id).first()
    if fav:
        db.session.delete(fav)
        db.session.commit()
        return jsonify({"message": "已取消收藏"})
    return jsonify({"error": "未找到该收藏"}), 404

@app.route("/api/comments", methods=["GET"])
def api_get_comments():
    video_id = request.args.get("video_id")
    if not video_id:
        return jsonify({"error": "缺失 video_id"}), 400
    
    comments = Comment.query.filter_by(video_id=video_id).order_by(Comment.created_at.desc()).all()
    results = [{
        "username": c.user.username,
        "content": c.content,
        "created_at": c.created_at.strftime("%Y-%m-%d %H:%M")
    } for c in comments]
    return jsonify({"comments": results})

@app.route("/api/comments/add", methods=["POST"])
@login_required
def api_add_comment():
    data = request.json
    video_id = str(data.get("video_id"))
    content = data.get("content")
    
    if not video_id or not content:
        return jsonify({"error": "内容不能为空"}), 400
    
    new_comment = Comment(
        user_id=current_user.id,
        video_id=video_id,
        content=content
    )
    db.session.add(new_comment)
    db.session.commit()
    
    return jsonify({
        "message": "评论成功",
        "comment": {
            "username": current_user.username,
            "content": content,
            "created_at": new_comment.created_at.strftime("%Y-%m-%d %H:%M")
        }
    })

@app.route("/api/filters")
def api_filters():
    """返回筛选器配置"""
    return jsonify(SEARCH_FILTERS)


@app.route("/api/search")
def api_search():
    keyword = request.args.get("kw", "").strip()
    industry = request.args.get("industry", "").strip()
    style = request.args.get("style", "").strip()
    page = request.args.get("page", 1, type=int)

    # 筛选参数
    filters = {
        "cate_id": request.args.get("cate_id", "", type=int) or None,
        "system_tags": request.args.get("system_tags", "").strip() or None,
        "duration": request.args.get("duration", "").strip() or None,
        "screen_type": request.args.get("screen_type", "", type=int) or None,
    }
    # 过滤掉空值
    filters = {k: v for k, v in filters.items() if v is not None}

    if not keyword and not filters:
        return jsonify({"error": "请输入搜索关键词或选择筛选条件"}), 400

    if not current_user.is_authenticated:
        return jsonify({"error": "请先登录系统以执行审计任务", "auth_required": True}), 401

    try:
        items = search_videos(keyword, page, filters)
    except Exception as e:
        return jsonify({"error": f"搜索失败: {str(e)}"}), 500

    if not items:
        return jsonify({"error": f"未找到符合条件的视频", "videos": []})

    # 0. 索引到语义 Embedding 库（模块3）
    try:
        emb_idx = get_index()
        emb_idx.add_batch(items)
        emb_idx.save()
    except Exception as e:
        print(f"[embedding] index failed: {e}")

    # 1. L1 极速评分与分流 (毫秒级，针对搜索列表) — 已内置 D1-D4
    scored_items = c_eye_l1_fast_scorer(items[:24], industry)

    # 2. LLM 批量生成 summary + key_elements (仅限 Pro 用户)
    llm_results = None
    if current_user.is_member:
        llm_results = evaluate_batch_with_llm(items[:12], industry, style)

    # 3. 组装结果
    favorite_ids = []
    if current_user.is_authenticated:
        favorite_ids = [f.video_id for f in Favorite.query.filter_by(user_id=current_user.id).all()]

    for i, item in enumerate(scored_items):
        item["is_favorite"] = item["id"] in favorite_ids

        # 注入 LLM 深度审计结果 (Pro 专属)
        if llm_results and i < len(llm_results):
            item["summary"] = llm_results[i].get("summary", "")
            item["key_elements"] = llm_results[i].get("key_elements", [])
            item["l2_audit"] = True
        else:
            original_v = next((v for v in items if str(v.get("id")) == item["id"]), None)
            if original_v:
                summary, key_elements = generate_local_summary(original_v, industry)
                item["summary"] = summary
                item["key_elements"] = key_elements
            item["l2_audit"] = False

    # 提取纯分数字段（前端兼容）
    for item in scored_items:
        sd = item.get("scores", {})
        item["D1"] = sd.get("D1", 0)
        item["D2"] = sd.get("D2", 0)
        item["D3"] = sd.get("D3", 0)
        item["D4"] = sd.get("D4", 0)
        ss = item.get("scenario_scores", {})
        item["overall"] = ss.get("default", item.get("reference_score", 0))

    # 4. 趋势检测（模块5）
    trend_signals = None
    try:
        emb_idx = get_index()
        trend_signals = detect_search_trends(items, emb_idx)
    except Exception as e:
        print(f"[api_search] trend detection failed: {e}")

    return jsonify({
        "keyword": keyword,
        "industry": industry,
        "style": style,
        "total": len(items),
        "page": page,
        "count": len(scored_items),
        "llm_enabled": llm_results is not None,
        "videos": scored_items,
        "trends": trend_signals,
        "indexed_count": get_index().size if trend_signals else 0,
    })


@app.route("/api/evaluate")
def api_evaluate():
    """获取单条视频的算法评分 + 4A ECD 深度审计（L2 级）"""
    try:
        article_id = request.args.get("id", "").strip()
        industry = request.args.get("industry", "").strip()
        style = request.args.get("style", "").strip()

        if not article_id:
            return jsonify({"error": "请提供视频ID"}), 400

        if not current_user.is_authenticated:
            return jsonify({"error": "请登录后查看深度审计"}), 401
            
        if not current_user.is_member:
            return jsonify({"error": "深度审计报告 (L2级) 为 Pro 专属功能", "upgrade_required": True}), 403

        # 尝试清理 ID (防止包含非数字字符)
        clean_id = "".join(filter(str.isdigit, article_id))
        if not clean_id:
            return jsonify({"error": "视频 ID 格式错误"}), 400

        try:
            detail = get_video_detail(int(clean_id))
        except Exception as e:
            print(f"[api_evaluate] get_video_detail failed for {clean_id}: {e}")
            return jsonify({"error": f"获取视频详情失败: {str(e)}"}), 500

        if not detail:
            return jsonify({"error": "未能获取到该视频的详细信息"}), 404

        # 1. 多维评分 D1-D4
        try:
            evaluation = build_full_evaluation(detail)
            scores = evaluation["scores"]
        except Exception as e:
            print(f"[api_evaluate] build_full_evaluation failed: {e}")
            evaluation = None
            scores = {"D1_audience_reception": {"score": 5.0}, "D2_commercial_value": {"score": 5.0},
                      "D3_team_professionalism": {"score": 5.0}, "D4_freshness": {"score": 5.0}}

        # 2. M2 ML 模式识别（互动分类 / 质量预测 / 异常检测）
        m2_results = None
        try:
            m2_results = run_m2_analysis(detail)
        except Exception as e:
            print(f"[api_evaluate] run_m2_analysis failed: {e}")

        # 3. L2 AI 元数据提取（模块1：LLM 推理）
        l2_metadata = None
        try:
            l2_metadata = extract_l2_metadata(detail, industry, style)
            # 注入 M2 互动分类结果到 L2 metadata（覆盖 LLM/规则推断）
            if l2_metadata and m2_results:
                l2_metadata["engagement_type"] = m2_results["engagement"]["type"]
                l2_metadata["engagement_confidence"] = m2_results["engagement"]["confidence"]
                l2_metadata["engagement_signals"] = m2_results["engagement"]["signals"]
        except Exception as e:
            print(f"[api_evaluate] extract_l2_metadata failed: {e}")

        # 4. 调用 LLM 进行 ECD 模式审计（注入 L2 元数据）
        try:
            ecd_report = evaluate_batch_with_llm([detail], industry, style, mode="ecd", l2_metadata=l2_metadata)
        except Exception as e:
            print(f"[api_evaluate] evaluate_batch_with_llm failed: {e}")
            ecd_report = None

        if not ecd_report:
            summary, key_elements = generate_local_summary(detail, industry)
            ecd_report = f"### 审计中断\n系统繁忙或 AI 配置未就绪，请稍后再试。\n\n**本地初筛摘要：**\n{summary}"

        # 提取扁平化分数用于前端
        flat_scores = {}
        for dim_key, dim_data in scores.items():
            if isinstance(dim_data, dict):
                flat_scores[dim_key] = dim_data.get("score", 0)
            else:
                flat_scores[dim_key] = dim_data

        # 5. 相似作品推荐（模块3：Embedding 语义相似度）
        similar_works = []
        try:
            emb_idx = get_index()
            # 确保当前视频已被索引
            emb_idx.add(detail)
            similar_works = emb_idx.find_similar(str(detail.get("id")), top_k=5)
        except Exception as e:
            print(f"[api_evaluate] similar_works failed: {e}")

        # 6. 创作者深度画像（模块4）
        creator_profile = None
        try:
            creator_profile = analyze_creator_profile(detail)
        except Exception as e:
            print(f"[api_evaluate] creator_profile failed: {e}")

        # 7. 跨维度综合推理（模块6）
        synthesis = None
        try:
            synthesis = build_m6_synthesis(detail, l1_scores=scores, l2_metadata=l2_metadata,
                                           m2_results=m2_results, similar_works=similar_works,
                                           creator_profile=creator_profile)
        except Exception as e:
            print(f"[api_evaluate] build_m6_synthesis failed: {e}")

        return jsonify({
            "id": detail.get("id"),
            "title": detail.get("title"),
            "scores": flat_scores,
            "score_details": scores,
            "scenarios": evaluation.get("scenarios", {}) if evaluation else {},
            "pool_info": evaluation.get("pool_info", {}) if evaluation else {},
            "ecd_report": ecd_report,
            "l2_metadata": l2_metadata,
            "m2_results": m2_results,
            "similar_works": similar_works,
            "creator_profile": creator_profile,
            "synthesis": synthesis,
            "web_url": detail.get("web_url", "")
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"系统内部错误: {str(e)}"}), 500


if __name__ == "__main__":
    from evaluate import _get_llm_client
    provider, _ = _get_llm_client()
    print("=" * 50)
    print("视频创意 AI 评价平台")
    if provider:
        print(f"LLM 评价: 已接入 ({provider})")
    else:
        print("LLM 评价: 未配置 — 使用本地摘要")
        print("  设置 DEEPSEEK_API_KEY 或 ANTHROPIC_API_KEY 以启用 LLM 评价")
    print("访问 http://localhost:5001")
    print("=" * 50)
    app.run(debug=True, use_reloader=False, host="0.0.0.0", port=5001)
