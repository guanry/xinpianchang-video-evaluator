# -*- coding: utf-8 -*-
"""
测试 L3 深度拉片分析 — Qwen3.5-Omni-Plus 视频直接分析
使用 evaluate.py 中的 analyze_video_with_qwen 函数
"""
import os, sys, time, json

# 加载 .env
env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(env_file):
    with open(env_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip()
                if k and v:
                    os.environ[k] = v

from evaluate import analyze_video_with_qwen

# 使用已有的测试视频
VIDEO_PATH = r"D:\视频评价平台\uploads\l3_20260527_111016_e2dfd333.mp4"

if not os.path.exists(VIDEO_PATH):
    alt = r"D:\视频评价平台\uploads\l3_20260527_105507_e2dfd333.mp4"
    if os.path.exists(alt):
        VIDEO_PATH = alt
    else:
        print("没有找到测试视频")
        sys.exit(1)

file_size_mb = os.path.getsize(VIDEO_PATH) / 1024 / 1024
print(f"视频: {os.path.basename(VIDEO_PATH)} ({file_size_mb:.1f}MB)")
print(f"DASHSCOPE_API_KEY: {'已配置' if os.environ.get('DASHSCOPE_API_KEY') else '未配置!'}")
print()

start = time.time()
result = analyze_video_with_qwen(VIDEO_PATH, industry="数码产品", style="科技感")
elapsed = time.time() - start

print(f"\n耗时: {elapsed:.0f}s  状态: {result.get('status')}  模型: {result.get('model', 'N/A')}")

if result.get("error"):
    print(f"错误: {result['error']}")
    sys.exit(1)

report_md = result.get("report_md", "")

# 保存报告
output_dir = r"D:\视频评价平台\L3分析"
os.makedirs(output_dir, exist_ok=True)
timestamp = time.strftime("%Y%m%d_%H%M%S")

md_path = os.path.join(output_dir, f"l3_report_{timestamp}.md")
with open(md_path, "w", encoding="utf-8") as f:
    f.write(report_md)
print(f"报告已保存: {md_path}")

json_path = os.path.join(output_dir, f"l3_report_{timestamp}.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(result.get("report_json", {}), f, ensure_ascii=False, indent=2)
print(f"JSON 已保存: {json_path}")
print(f"报告长度: {len(report_md)} 字符")
