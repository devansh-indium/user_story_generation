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
    
class ConversationEmbedding(db.Model):
    __tablename__ = "conversation_embeddings"

    id            = db.Column(db.Integer, primary_key=True)
    session_id    = db.Column(db.String(36), nullable=False, index=True)
    conversation_id = db.Column(db.Integer, db.ForeignKey("conversations.id"))
    content       = db.Column(db.Text, nullable=False)  
    ticket_id     = db.Column(db.String(50))             
    ticket_type   = db.Column(db.String(20))             
    platform      = db.Column(db.String(10))             
    embedding     = db.Column(db.Text, nullable=False)   
    created_at    = db.Column(db.DateTime, default=datetime.utcnow)