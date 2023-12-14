from init import db
import json
        
class Vote(db.Model):
    __tablename__ = 'vote'
    id = db.Column(db.Integer, primary_key=True)
    member_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    topic_id = db.Column(db.Integer, db.ForeignKey('topic.id'))
    member_weight = db.Column(db.Float)  # the vote value might be weighted by the member's weight when calculating results
    values = db.Column(db.JSON)
    
    @classmethod
    def cast_vote(cls, member_id, topic_id, values):
        existing_vote = cls.query.filter_by(member_id=member_id, topic_id=topic_id).first()
        if existing_vote:
            existing_vote.values = values
        else:
            vote = cls(member_id=member_id, topic_id=topic_id, values=values)
            db.session.add(vote)

        db.session.commit()

    @classmethod
    def are_all_votes_in(cls, topic_id, team_size):
        votes_count = cls.query.filter_by(topic_id=topic_id).count()
        return votes_count >= team_size