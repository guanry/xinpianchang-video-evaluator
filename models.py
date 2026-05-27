from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime

db = SQLAlchemy()

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    is_member = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    favorites = db.relationship('Favorite', backref='user', lazy=True)
    comments = db.relationship('Comment', backref='user', lazy=True)

class Favorite(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    video_id = db.Column(db.String(50), nullable=False)
    video_title = db.Column(db.String(200))
    video_cover = db.Column(db.String(500))
    video_data = db.Column(db.Text)  # Store JSON of video details for easy retrieval
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Unique constraint to prevent duplicate favorites per user
    __table_args__ = (db.UniqueConstraint('user_id', 'video_id', name='_user_video_uc'),)

class Comment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    video_id = db.Column(db.String(50), nullable=False)
    content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class VideoRecord(db.Model):
    """缓存视频完整数据与 AI 评价结果"""
    id = db.Column(db.String(50), primary_key=True)  # article_id
    title = db.Column(db.String(200))
    cover = db.Column(db.String(500))
    duration = db.Column(db.Integer)
    # 存储完整原始 JSON 数据
    raw_data = db.Column(db.Text) 
    # 关键指标快照
    views = db.Column(db.Integer, default=0)
    likes = db.Column(db.Integer, default=0)
    collects = db.Column(db.Integer, default=0)
    shares = db.Column(db.Integer, default=0)
    
    # --- AI 评价持久化 ---
    ecd_report = db.Column(db.Text)      # ECD 审计报告 (Markdown)
    l2_metadata = db.Column(db.Text)     # L2 结构化元数据 (JSON)
    synthesis = db.Column(db.Text)       # 综合推理结果 (JSON)
    creator_profile = db.Column(db.Text)  # 创作者画像 (JSON)
    
    # --- Gemini L3 视听审计 (New) ---
    l3_gemini_report = db.Column(db.Text)     # Gemini 详细视听审计报告 (Markdown)
    l3_structured_data = db.Column(db.Text)   # Gemini 提取的结构化量化数据 (JSON)
    
    last_updated = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    ai_evaluated_at = db.Column(db.DateTime)  # AI 评价生成的具体时间

    def is_stale(self, hours=24):
        """检查数据是否超过指定小时未更新（默认24小时）"""
        delta = datetime.utcnow() - self.last_updated
        return delta.total_seconds() > hours * 3600


class SearchCache(db.Model):
    """缓存搜索结果，避免重复调 API + 重算 L1"""
    id = db.Column(db.Integer, primary_key=True)
    query_hash = db.Column(db.String(64), unique=True, nullable=False)  # SHA256 of kw+page+filters
    results_json = db.Column(db.Text)  # 缓存完整搜索结果 JSON
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def is_stale(self, hours=24):
        delta = datetime.utcnow() - self.created_at
        return delta.total_seconds() > hours * 3600
