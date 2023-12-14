from init import db

class OptionDecisionCriteriaAssociation(db.Model):
    __tablename__ = 'option_decision_association'
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    option_id = db.Column(db.Integer, db.ForeignKey('option.id', ondelete='CASCADE'), primary_key=True)
    decision_criteria_id = db.Column(db.Integer, db.ForeignKey('decision_criteria.id', ondelete='CASCADE'), primary_key=True)
    user = db.relationship('User', backref='option_decision_criteria_associations')
    input_value = db.Column(db.String)
    
    def to_dict(self):
        return {
            'option_id': self.option_id,
            'decision_criteria_id': self.decision_criteria_id,
            'input_value': self.input_value
        }

    # option = db.relationship('Option', back_populates='decision_criteria_associations', cascade="all, delete-orphan")
    # decision_criteria = db.relationship('DecisionCriteria', back_populates='option_associations', cascade="all, delete-orphan")

class UserDecisionCriteriaWeight(db.Model):
    __tablename__ = 'user_decision_criteria_weight'
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), primary_key=True)
    topic_id = db.Column(db.Integer, db.ForeignKey('topic.id'), primary_key=True)
    decision_criteria_id = db.Column(db.Integer, db.ForeignKey('decision_criteria.id'), primary_key=True)
    weight = db.Column(db.Float)

    user = db.relationship('User', backref='decision_criteria_weights')
    decision_criteria = db.relationship('DecisionCriteria', backref='user_weights')


class Option(db.Model):
    __tablename__ = 'option'
    
    id = db.Column(db.Integer, primary_key=True)
    topic_id = db.Column(db.Integer, db.ForeignKey('topic.id'))
    description = db.Column(db.String(255))
    rank_id = db.Column(db.Integer, db.ForeignKey('rank.id'))
    rank = db.relationship('Rank', back_populates='options')
    is_base_model = db.Column(db.Boolean, default=False)

    # New columns for TOPSIS details
    normalized_decision_matrix = db.Column(db.PickleType())  # Stores the normalized decision matrix
    weighted_normalized_decision_matrix = db.Column(db.PickleType())  # Stores the weighted normalized matrix
    ideal_solution_distance = db.Column(db.Float)  # Distance to the ideal solution
    anti_ideal_solution_distance = db.Column(db.Float)  # Distance to the anti-ideal solution
    value = db.Column(db.Integer)
    # votes = db.relationship('Vote', backref='option')
    decision_criteria_associations = db.relationship('OptionDecisionCriteriaAssociation', cascade="all, delete", backref='option')
    
    def to_dict(self):
        associations_dict = [assoc.to_dict() for assoc in self.decision_criteria_associations]
        return {
            'id': self.id,
            'description': self.description,
            'is_base_model': self.is_base_model,
            'rank_position': self.rank.position if self.rank else None,
            'normalized_decision_matrix': self.normalized_decision_matrix,
            'weighted_normalized_decision_matrix': self.weighted_normalized_decision_matrix,
            'ideal_solution_distance': self.ideal_solution_distance,
            'anti_ideal_solution_distance': self.anti_ideal_solution_distance,
            'final_score': self.value,
            'associations': associations_dict
    }
        
class DecisionCriteria(db.Model):
    __tablename__ = 'decision_criteria'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), nullable=False)
    topic_id = db.Column(db.Integer, db.ForeignKey('topic.id'))
    description = db.Column(db.String(255))
    potential_answers = db.Column(db.PickleType())  # store a list of potential answers
    ideal_solution = db.Column(db.Float)
    anti_ideal_solution = db.Column(db.Float)
    weight = db.Column(db.Float, default=1.0)
    
    # Relationship with Option using the association table
    option_associations = db.relationship('OptionDecisionCriteriaAssociation', cascade="all, delete", backref='decision_criteria')

    def __init__(self, name, description, topic_id, potential_answers=None):
        self.name = name
        self.description = description
        self.topic_id = topic_id
        self.potential_answers = potential_answers if potential_answers else []

    @property
    def potential_answers(self):
        if self.potential_answers:
            return self.potential_answers.split(',')
        return []

    @potential_answers.setter
    def potential_answers(self, value):
        if value:
            self.potential_answers = ','.join(value)
    
    def validate_input(self, input_value):
        if input_value in self.potential_answers:
            return True
        return False
    
    def to_dict(self):
        return{
            'id': self.id,
            'title': self.name,
            'description': self.description,
            'weight': self.weight,
            'ideal_solution': self.ideal_solution,
            'anti_solution': self.anti_ideal_solution,
            'options': [assoc.option.to_dict() for assoc in self.option_associations]
        }