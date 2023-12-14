from init import db, bcrypt
from flask import request, jsonify
from decision_criteria_models import Option, DecisionCriteria, OptionDecisionCriteriaAssociation
from rank_models import Rank
from vote_models import Vote
import hashlib

team_member_association = db.Table('team_member_association',
    db.Column('team_id', db.Integer, db.ForeignKey('team.id'), primary_key=True),
    db.Column('user_id', db.Integer, db.ForeignKey('user.id'), primary_key=True),
    db.Column('weight', db.Float, default=1.0)
)

topic_participants_association = db.Table('topic_participants_association',
    db.Column('topic_id', db.Integer, db.ForeignKey('topic.id'), primary_key=True),
    db.Column('user_id', db.Integer, db.ForeignKey('user.id'), primary_key=True),
    db.Column('weight', db.Float)
)

class User(db.Model):
    __tablename__ = 'user'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    host_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    hosted_teams = db.relationship('Team', back_populates='host')
    teams = db.relationship('Team', secondary=team_member_association, back_populates='members')
    teams_as_member = db.relationship('Team', secondary=team_member_association, back_populates='members')
    type = db.Column(db.String(50))
    
    __mapper_args__ = {
        'polymorphic_identity': 'user',
        'polymorphic_on': type
    }
    
    def create_team(self, name, description=None):
        if self.type != 'host':
            self.type = 'host'
            db.session.add(self)
        team = Team(name=name, description=description, host=self)
        db.session.add(team)
        try:
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            raise e
        return team
    
    def set_password(self, plain_password):
        self.password_hash = hashlib.sha256(plain_password.encode('utf-8')).hexdigest()

    def check_password(self, plain_password):
        return self.password_hash == hashlib.sha256(plain_password.encode('utf-8')).hexdigest()
    
    @property
    def is_authenticated(self):
        return True
    
    @property
    def is_active(self):
        return True
        
    @property
    def is_anonymous(self):
        return False
    
    def get_id(self):
        return str(self.id)

    def to_dict(self):
        return {
            'id' : self.id,
            'username': self.username,
            'email': self.email,
            'host_id': self.host_id,
            'user_type': self.type
        }
    
    def register(self, plain_password):
        existing_user = User.query.filter_by(email=self.email).first()
        if not existing_user:
            self.set_password(plain_password)
            db.session.add(self)
            try:
                db.session.commit()
                print(f"User {self.username} registered successfully!")  # For debugging
                return {"message": f"User {self.username} registered successfully!"}, 201 
            except Exception as e:
                db.session.rollback()
                print(f"An error occured during registration: {str(e)}") # Log the actual error message
                return {"error": f"An error occurred during registration: {e}. Please try again later."}, 500
        else:
            print(f"Attempt to register with an already registered email: {self.email}")  # For debugging
            return {"error": "Email already registered"}, 409
    
    @classmethod
    def log_in(cls, email, plain_password):
        user = cls.query.filter_by(email=email).first()
        if user and user.check_password(plain_password):
            return user
        return None
    
    def choose_topic(self, topic):
        if topic not in self.topics:
            self.topics.append(topic)
            db.session.commit()
            return f"Joined topic: {topic.name}"
        else:
            return "Already Joined this topic!"
    
class Host(User):
    __mapper_args__ = {
        'polymorphic_identity': 'host',
    }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.type = 'host'
        
    def create_topic(self, topic_name):
        topic = Topic(name=topic_name, host=self)
        db.session.add(topic)
        db.session.commit()
        return topic

    def assign_members(self, member, topic):
        if member not in topic.members:
            topic.members.append(member)
            db.session.commit()
            return f"{member.username} assigned to {topic.name}"
        else:
            return f"{member.username} is already assigned to {topic.name}"
    
    def add_option_to_topic(self, topic, description):
        if topic.host_id == self.id:
            new_option = Option(description=description, topic=topic)
            db.session.add(new_option)
            db.session.commit()
            return f"Option '{description} added to topic '{topic.name}'"
        else:
            return "Unauthorized action. This topic does not belong to the host"
    
    def remove_option_from_topic(self, topic, option):
        if topic.host_id == self.id and option in topic.options:
            db.session.delete(option)
            db.session.commit()
            return f"Option '{option.description}' removed from topic '{topic.name}'"
        else:
            return "Unauthorized action or option not found"        
    
    def get_all_options_for_topic(self, topic):
        if topic.host_id == self.id:
            return topic.options
        else:
            return "Unauthorized action. This topic does not belong to the host"
    
    def create_decision_criteria(self, topic, name, description, potential_answers=None):
        criteria = DecisionCriteria(description, topic, potential_answers)
        db.session.add(criteria)
        db.session.commit()
        return criteria
    
    def update_decision_criteria(self, criteria, name=None, description=None, potential_answers=None):
        if name:
            criteria.name = name
        if description:
            criteria.description = description
        if potential_answers:
            criteria.potential_answers = potential_answers
        db.session.add(criteria)
        db.session.commit()
    
    def post_poll(self, topic):
        new_topic = Topic(name=topic, host=self)
        db.session.add(new_topic)
        db.session.commit()
    
    def set_member_weight(self, member, weight):
        member.weight = weight
        db.session.add(member)
        db.session.commit()
        
class Member(User):
    __mapper_args__ = {
        'polymorphic_identity': 'member',
    }
    weight = db.Column(db.Float, default=1.0)
    votes = db.relationship('Vote', backref="member", foreign_keys=[Vote.member_id])
                            
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.type = 'member'
    
    @property
    def weight(self):
        return self._weight
    
    @weight.setter
    def weight(self, value):
        self._weight = value
        db.session.commit()

class Team(db.Model):
    __tablename__ = 'team'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), nullable=False)
    description = db.Column(db.String(255))
    topics = db.relationship('Topic', backref='team', lazy='dynamic')
    host_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    host = db.relationship('User', back_populates='hosted_teams')
    members = db.relationship('User', secondary=team_member_association, back_populates='teams_as_member')

    def add_member(self, member):
        if member not in self.members:
            self.members.append(member)
            db.session.commit()

    def remove_member(self, member):
        if member in self.members:
            self.members.remove(member)
            db.session.commit()
    
    def to_dict(self):
        return {
            'id' : self.id,
            'name': self.name,
            'description': self.description,
            'host_id': self.host_id
        }

class Topic(db.Model):
    __tablename__ = 'topic'

    id = db.Column(db.Integer, primary_key=True)
    host_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    title = db.Column(db.String(255), nullable=False)
    description = db.Column(db.String(255))
    decision_criteria = db.relationship('decision_criteria_models.DecisionCriteria', backref='topic', lazy='dynamic')
    team_id = db.Column(db.Integer, db.ForeignKey('team.id'))
    participants = db.relationship('User', secondary='topic_participants_association')
    ranks = db.relationship('Rank', back_populates='topic')
    
    
    def __init__(self, title, description, team_id=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.title = title
        self.description = description
        self.team_id = team_id
        self.participants = []
        self._decision_criteria = []
        
    def get_assigned_members(self):
        return self.members
    
    def get_member_count(self):
        return len(self.participants)
    
    def to_dict(self):
        return {
            'id': self.id,
            'title': self.title,
            'description': self.description,
            'team_id': self.team_id
        }
    

