# -*- coding: utf-8 -*-
"""
视频信息抓取器
通过 apis.netstart.cn/xpc 代理 API 获取视频详情
文档: https://apis.netstart.cn/xpc/
"""
import json, os, sys, re
import requests

# Fix Windows console encoding for Chinese output
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except:
        pass

API_BASE = "https://www.xinpianchang.com/api/xpc/v2"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": "https://www.xinpianchang.com/",
}


def extract_article_id(url_or_id):
    """从URL或纯数字ID中提取article_id"""
    if isinstance(url_or_id, int):
        return str(url_or_id)
    if isinstance(url_or_id, str) and url_or_id.isdigit():
        return url_or_id
    # 从URL提取: https://www.xinpianchang.com/a13266271 -> 13266271
    m = re.search(r'/a(\d+)', url_or_id)
    if m:
        return m.group(1)
    m = re.search(r'article/(\d+)', url_or_id)
    if m:
        return m.group(1)
    raise ValueError(f"无法从 '{url_or_id}' 提取视频ID")


def get_article(article_id, from_pc=False):
    """获取视频文章详情"""
    url = f"{API_BASE}/article/{article_id}"
    params = {}
    if from_pc:
        params["from"] = "pc"
    resp = requests.get(url, params=params, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    # v2 API 响应格式：直接返回 data 或包裹在 {status, data} 中
    if "status" in data and "data" in data:
        if data.get("status") != 0:
            raise Exception(f"API Error: code={data.get('code')} message={data.get('message')}")
        return data["data"]
    # 直接返回的 data
    return data


def search_videos(keyword: str = "", page: int = 1, cate_id: str = None,
                  duration: str = None, screen_type: str = None,
                  sort: str = "hot") -> list:
    """搜索视频，返回 list"""
    url = f"{API_BASE}/search"
    params = {"type": "article", "sort": sort, "page": page}
    if keyword:
        params["kw"] = keyword
    if cate_id:
        params["cate_id"] = cate_id
    if duration:
        params["duration"] = duration
    if screen_type:
        params["screen_type"] = screen_type

    resp = requests.get(url, params=params, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    # v2 API 响应格式
    if "status" in data and "data" in data:
        if data.get("status") != 0:
            return []
        return data["data"].get("list", [])
    # 直接格式
    return data.get("list", data if isinstance(data, list) else [])


def get_comments(article_id, page=1):
    """获取视频评论"""
    url = f"{API_BASE}/article/{article_id}/comments"
    params = {"page": page}
    resp = requests.get(url, params=params, headers=HEADERS, timeout=30)
    data = resp.json()
    if "status" in data and "data" in data:
        return data["data"]
    return data


def get_related(article_id):
    """获取相关视频"""
    url = f"{API_BASE}/article/{article_id}/next"
    resp = requests.get(url, headers=HEADERS, timeout=30)
    return resp.json()


def format_detail(detail):
    """将API返回数据格式化为可读的摘要"""
    c = detail.get("count", {})
    a = detail.get("author", {})
    u = a.get("userinfo", {})

    info = {
        "id": detail["id"],
        "title": detail["title"],
        "duration_sec": detail.get("duration"),
        "duration_display": f"{detail['duration']//60}分{detail['duration']%60}秒" if detail.get("duration") else None,
        "cover_url": detail.get("cover"),
        "quality": detail.get("quality"),
        "badge": detail.get("badge"),
        "ip_location": detail.get("ip_location"),
        "web_url": detail.get("web_url"),

        "stats": {
            "view_count": c.get("count_view"),
            "like_count": c.get("count_like"),
            "collect_count": c.get("count_collect"),
            "share_count": c.get("count_share"),
            "comment_count": c.get("count_comment"),
            "score": c.get("score"),
        },

        "categories": [
            {
                "main": cat.get("category_name"),
                "sub": cat.get("sub", {}).get("category_name"),
            }
            for cat in detail.get("categories", [])
        ],

        "author": {
            "username": u.get("username"),
            "role": a.get("role"),
            "bio": u.get("about"),
            "verify": u.get("verify_description"),
            "gender": "男" if u.get("sex") == 1 else "女" if u.get("sex") == 2 else "未知",
            "vip_level": u.get("vip_flag"),
            "avatar_url": u.get("avatar"),
            "profile_url": u.get("web_url"),
        },

        "content": detail.get("content", ""),
    }
    return info


def scrape(url_or_id):
    """一站式抓取视频信息"""
    article_id = extract_article_id(url_or_id)
    print(f"抓取视频 ID: {article_id}")

    detail = get_article(article_id)
    info = format_detail(detail)

    # 也获取PC版内容（可能有不同的content字段）
    try:
        detail_pc = get_article(article_id, from_pc=True)
        pc_content = detail_pc.get("content", "")
        if pc_content and pc_content != info["content"]:
            info["content_pc"] = pc_content
    except:
        pass

    return info


# ===== CLI =====
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="视频信息抓取器")
    parser.add_argument("url_or_id", nargs="?", default="13615873", help="视频URL或ID")
    parser.add_argument("--json", "-j", action="store_true", help="输出原始JSON")
    parser.add_argument("--save", "-s", type=str, help="保存到指定文件")
    args = parser.parse_args()

    info = scrape(args.url_or_id)

    if args.json:
        print(json.dumps(info, ensure_ascii=False, indent=2))
    else:
        print(f"\n{'='*60}")
        print(f"【{info['title']}】")
        print(f"{'='*60}")
        print(f"时长: {info['duration_display']} | 画质等级: {info['quality']} | 标签: {info['badge']}")
        print(f"IP属地: {info['ip_location']}")
        print(f"链接: {info['web_url']}")
        print(f"封面: {info['cover_url']}")

        print(f"\n--- 数据表现 ---")
        s = info['stats']
        print(f"播放: {s['view_count']:,} | 点赞: {s['like_count']:,} | 收藏: {s['collect_count']:,} | 分享: {s['share_count']:,} | 评论: {s['comment_count']} | 评分: {s['score']}")

        print(f"\n--- 分类 ---")
        for cat in info['categories']:
            print(f"  {cat['main']} > {cat['sub']}")

        print(f"\n--- 创作人 ---")
        a = info['author']
        print(f"  {a['username']} | {a['role']} | {a['gender']} | VIP{a['vip_level']}")
        print(f"  简介: {a['bio']}")
        print(f"  认证: {a['verify']}")
        print(f"  主页: {a['profile_url']}")

        print(f"\n--- 内容 ---")
        content = info.get('content') or info.get('content_pc', '')
        print(content[:2000] if content else "(无内容)")

    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
        print(f"\n已保存至: {args.save}")
