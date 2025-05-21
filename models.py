from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class History(db.Model):
    id: int = db.Column(db.Integer, primary_key=True)
    title: str = db.Column(db.String(200), nullable=False)


class HistoryMessage(db.Model):
    id: int = db.Column(db.Integer, primary_key=True)
    history_id: int = db.Column(db.Integer, db.ForeignKey("history.id"), nullable=False)
    message: str = db.Column(db.Text, nullable=False)
    is_user: bool = db.Column(db.Boolean, default=True)
    is_system: bool = db.Column(db.Boolean, default=False)
    is_pending: bool = db.Column(db.Boolean, default=False, nullable=False)
    history = db.relationship("History", backref=db.backref("messages", lazy=True))
