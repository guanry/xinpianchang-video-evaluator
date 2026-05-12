# -*- coding: utf-8 -*-
import json
from evaluate import evaluate_video, generate_local_summary

# 模拟 scrape_video 返回的数据
video_data = {
    "id": 13266271,
    "title": "HUAWEI｜Nice To Hear You",
    "duration": 315,
    "quality": 4,
    "badge": "recommend",
    "count": {
        "count_view": 97448,
        "count_like": 755,
        "count_collect": 1534,
        "count_share": 306,
        "score": 9684
    },
    "categories": [
        {"category_name": "广告片", "sub": {"category_name": "互联网服务"}},
        {"category_name": "广告片", "sub": {"category_name": "剧情广告"}}
    ],
    "tags": [{"name": "感人"}, {"name": "品牌"}, {"name": "华为"}],
    "author": {
        "userinfo": {
            "vip_flag": 3,
            "verify_description": "导演/编剧"
        }
    },
    "content": "华为手语视频服务 导演版..."
}

# 测试评分
print("--- 算法评分测试 ---")
scores = evaluate_video(video_data, industry="3C科技", style_preference="感人")
print(json.dumps(scores, indent=2, ensure_ascii=False))

# 测试本地摘要（降级方案）
print("\n--- 本地摘要测试 ---")
summary, tags = generate_local_summary(video_data, industry="3C科技")
print(f"Summary: {summary}")
print(f"Tags: {tags}")
