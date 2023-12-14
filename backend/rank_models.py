from init import db

class Rank(db.Model):
    __tablename__ = 'rank'
    
    id = db.Column(db.Integer, primary_key=True)
    position = db.Column(db.Integer, unique=True)
    topic_id = db.Column(db.Integer, db.ForeignKey('topic.id'))
    options = db.relationship('Option', order_by='Option.id', back_populates='rank') 
    topic = db.relationship('Topic', back_populates='ranks')

class Ranking:
    @staticmethod
    def rank_options(options):
        return sorted(options, key=lambda option: option.get_score(), reverse=True)
