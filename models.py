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


class UserCredits(db.Model):
    """用户积分/余额系统"""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), unique=True, nullable=False)
    balance = db.Column(db.Integer, default=0)  # 余额，单位：分
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = db.relationship('User', backref=db.backref('credits', uselist=False))


class CreditTransaction(db.Model):
    """积分交易记录"""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    amount = db.Column(db.Integer, nullable=False)  # 正数充值，负数消费
    type = db.Column(db.String(20), nullable=False)  # recharge/consume/refund
    reference_id = db.Column(db.String(50))  # 关联ID (如 l3_analysis.id)
    description = db.Column(db.String(200))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    user = db.relationship('User', backref='transactions')


class L3Analysis(db.Model):
    """L3 级单视频深度分析记录"""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    
    # 视频信息
    video_filename = db.Column(db.String(200))  # 上传的文件名
    video_path = db.Column(db.String(500))  # 服务器存储路径
    video_size_mb = db.Column(db.Float)  # 文件大小 MB
    
    # 分析状态
    status = db.Column(db.String(20), default='pending')  # pending/processing/completed/failed
    error_message = db.Column(db.Text)
    
    # 分析结果
    report_json = db.Column(db.Text)  # 结构化分析结果 JSON
    report_md = db.Column(db.Text)  # Markdown 格式报告
    
    # 关联搜索结果视频（可选）
    reference_video_id = db.Column(db.String(50), nullable=True)
    reference_video_title = db.Column(db.String(200), nullable=True)

    # 扣费信息
    credits_used = db.Column(db.Integer, default=99)  # 消耗积分，默认99分=9.9元

    # 时间戳
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    started_at = db.Column(db.DateTime)  # 开始处理时间
    completed_at = db.Column(db.DateTime)  # 完成时间

    user = db.relationship('User', backref='l3_analyses')
