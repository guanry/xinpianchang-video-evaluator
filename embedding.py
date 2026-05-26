# -*- coding: utf-8 -*-
"""
语义 Embedding 索引 — 模块3：作品语义向量 + 相似推荐
基于字符 bigram + 标签/分类加权，无需额外依赖
"""
import json, os, math
from typing import Optional


class EmbeddingIndex:
    """轻量级作品语义索引，基于字符 bigram + 元数据加权特征

    使用方式：
        idx = EmbeddingIndex("cache/embeddings.json")
        idx.add_batch(search_results)       # 批量索引
        similar = idx.find_similar("13615873", top_k=5)  # 查相似
    """

    def __init__(self, cache_file: str = None):
        self._videos = {}       # video_id -> lightweight metadata
        self._vectors = {}      # video_id -> {feature: weight}
        self._cache_file = cache_file
        if cache_file and os.path.exists(cache_file):
            self._load()

    # ── public API ──

    @property
    def size(self) -> int:
        return len(self._vectors)

    def add(self, video: dict):
        """索引单条视频"""
        vid = str(video.get("id", ""))
        if not vid or vid in self._vectors:
            return
        self._videos[vid] = self._strip_meta(video)
        self._vectors[vid] = self._build_vector(video)

    def add_batch(self, videos: list):
        """批量索引"""
        for v in (videos or []):
            self.add(v)

    def find_similar(self, video_id, top_k: int = 5,
                     exclude_same_author: bool = True) -> list:
        """查找与给定视频最相似的 top_k 个作品

        返回: [{"id": str, "title": str, "similarity": float, "cover": str, ...}, ...]
        """
        vid = str(video_id)
        if vid not in self._vectors:
            return []

        source_vec = self._vectors[vid]
        source_author = self._videos.get(vid, {}).get("author", "")

        scored = []
        for other_id, other_vec in self._vectors.items():
            if other_id == vid:
                continue
            if exclude_same_author:
                other_author = self._videos.get(other_id, {}).get("author", "")
                if source_author and other_author and source_author == other_author:
                    continue

            sim = self._cosine_similarity(source_vec, other_vec)
            if sim > 0.05:  # 最小相似度阈值
                meta = self._videos.get(other_id, {})
                scored.append({
                    "id": other_id,
                    "title": meta.get("title", ""),
                    "cover": meta.get("cover", ""),
                    "duration": meta.get("duration", 0),
                    "author": other_author,
                    "similarity": round(sim, 3),
                })

        scored.sort(key=lambda x: x["similarity"], reverse=True)
        return scored[:top_k]

    def save(self):
        """持久化到缓存文件"""
        if not self._cache_file:
            return
        os.makedirs(os.path.dirname(self._cache_file), exist_ok=True)
        data = {
            "videos": self._videos,
            "vectors": {k: v for k, v in self._vectors.items()},
        }
        with open(self._cache_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        return self._cache_file

    # ── internals ──

    def _load(self):
        try:
            with open(self._cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._videos = data.get("videos", {})
            self._vectors = data.get("vectors", {})
        except Exception:
            self._videos = {}
            self._vectors = {}

    @staticmethod
    def _strip_meta(video: dict) -> dict:
        """提取轻量元数据用于展示"""
        author = video.get("author", {}) or {}
        ui = author.get("userinfo", {}) or {}
        return {
            "id": str(video.get("id", "")),
            "title": video.get("title", ""),
            "cover": video.get("cover", ""),
            "duration": video.get("duration", 0),
            "author": ui.get("username", ""),
            "web_url": video.get("web_url", ""),
        }

    @staticmethod
    def _build_vector(video: dict) -> dict:
        """从元数据构建稀疏特征向量

        特征权重：
          - 标题 bigram：×1
          - 标签：×2.5
          - 主分类：×3
          - 子分类：×3.5（品类一致性最强信号）
          - 作者名：×1
          - 内容 bigram：×1
        """
        vec = {}

        # 标题：字符 bigram
        title = video.get("title", "") or ""
        for i in range(len(title) - 1):
            bg = "t:" + title[i:i+2]
            vec[bg] = vec.get(bg, 0) + 1

        # 标签（高权重）
        for tag in video.get("tags", []) or []:
            name = (tag.get("name", "") if isinstance(tag, dict) else str(tag)).strip()
            if name:
                vec["tag:" + name] = 2.5

        # 分类（中高权重 — 品类一致性是强信号）
        for cat in video.get("categories", []) or []:
            if isinstance(cat, dict):
                main = (cat.get("category_name") or "").strip()
                sub = (cat.get("sub") or {}).get("category_name", "").strip()
                if main:
                    vec["cat:" + main] = 3.0
                if sub:
                    vec["cat:" + sub] = 3.5

        # 导演（低权重，用于区分同一作者）
        author = video.get("author", {}) or {}
        ui = author.get("userinfo", {}) or {}
        username = (ui.get("username") or "").strip()
        if username:
            vec["author:" + username] = 1.0

        # 内容片段（补充语义）
        content = (video.get("content", "") or "")[:300]
        for i in range(len(content) - 1):
            ch = content[i:i+2]
            if ch.strip() and not ch[0].isspace() and not ch[1].isspace():
                vec["c:" + ch] = vec.get("c:" + ch, 0) + 1

        return vec

    @staticmethod
    def _cosine_similarity(vec_a: dict, vec_b: dict) -> float:
        """计算两个稀疏向量的余弦相似度"""
        if not vec_a or not vec_b:
            return 0.0

        # 点积
        dot = 0.0
        for k, va in vec_a.items():
            vb = vec_b.get(k, 0)
            if vb:
                dot += va * vb

        if dot == 0:
            return 0.0

        # L2 范数
        norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
        norm_b = math.sqrt(sum(v * v for v in vec_b.values()))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot / (norm_a * norm_b)


# ── 全局单例 ──

_global_index: Optional[EmbeddingIndex] = None


def get_index(cache_file: str = "instance/embedding_cache.json") -> EmbeddingIndex:
    """获取全局 EmbeddingIndex 单例"""
    global _global_index
    if _global_index is None:
        _global_index = EmbeddingIndex(cache_file)
    return _global_index
