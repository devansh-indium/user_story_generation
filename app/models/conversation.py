from app import db
from datetime import datetime

class Conversation(db.Model):
    __tablename__ = "conversations"

    id         = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(100), nullable=False, index=True)
    role       = db.Column(db.String(20),  nullable=False)  # 'user' or 'assistant'
    content    = db.Column(db.Text,        nullable=False)
    created_at = db.Column(db.DateTime,    default=datetime.utcnow)

    def to_dict(self):
        return {
            "role":    self.role,
            "content": self.content
        }