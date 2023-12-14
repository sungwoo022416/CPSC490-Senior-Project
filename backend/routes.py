from flask import Blueprint, request, jsonify, session, abort
from flask_login import login_user, current_user, login_required,logout_user
from user_models import User, Team, Topic, team_member_association, topic_participants_association
from decision_criteria_models import DecisionCriteria, Option, OptionDecisionCriteriaAssociation, UserDecisionCriteriaWeight
from rank_models import Rank
from vote_models import Vote
from init import db, login_manager, bcrypt
from sqlalchemy.orm import joinedload
import numpy as np
import json

main = Blueprint('main', __name__)

@main.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    user = User.log_in(email, password)
    
    if user:
        login_user(user)
        return jsonify({"message": "Logged in successfully!"}), 200
    else:
        return jsonify({"error": "Login Unsuccessful. Please check email and password"}), 401

@main.route('/logout')
def logout():
    logout_user()
    return jsonify({"message": "logged out sucessfully! "}), 200

@main.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username')
    email = data.get('email')
    plain_password = data.get('password')
    
    user = User(username=username, email=email)
    registration_response, status_code = user.register(plain_password)

    return jsonify(registration_response), status_code

@main.route('/current_user', methods=['GET'])
@login_required
def get_current_user():
    return jsonify(current_user.to_dict())

@main.route('/teams', methods=['POST'])
@login_required
def create_team_route():
    if not current_user.is_authenticated:
        return jsonify({"error": "Authentication required"}), 401
    
    data = request.get_json()
    team_name = data.get('name')
    description = data.get('description')
    
    new_team = current_user.create_team(name=team_name, description=description)
    new_team.host = current_user
    try:
        db.session.add(new_team)
        db.session.commit()
        return jsonify(new_team.to_dict()), 201    
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500

@main.route('/teams', methods=['GET'])
@login_required
def get_teams():

    hosted_teams = Team.query.filter_by(host_id=current_user.id).all()
    hosted_teams_data = [{"id": team.id, "name": team.name, "description": team.description, "role": "Host"} for team in hosted_teams]

    member_of_teams = current_user.teams_as_member
    member_of_teams_data = [{"id": team.id, "name": team.name, "description": team.description, "role": "Member"} for team in member_of_teams]

    all_teams_data = hosted_teams_data + member_of_teams_data
    return jsonify(all_teams_data), 200

@main.route('/teams/<int:team_id>/topics', methods=['POST'])
@login_required
def create_topic_route(team_id):
    if not current_user.is_authenticated:
        return jsonify({"error": "Authentication required"}), 401
    
    data = request.get_json()
    title = data.get('title')
    description = data.get('description')
    
    if not title or not description:
        return jsonify({"error": "Title and description are required. "}), 400
    
    if not team_id:
        return jsonify({"error": "Team ID is required."}), 400
    
    if Topic.query.filter_by(title=title, team_id=team_id).first():
        return jsonify({"error": "Topic with the given title already exists"}), 400
    
    if current_user.type != 'host':
        return jsonify({"error": "Only hosts can create topics"}), 403

    
    new_topic = Topic(title=title, description=description, team_id=team_id) 
    new_topic.host_id = current_user.id
    
    db.session.add(new_topic)
    try:
        db.session.commit()
        return jsonify({"message": "Topic created successfully.", "topic": new_topic.to_dict()}), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500
    
@main.route('/teams/<int:team_id>/topics', methods=['GET'])
@login_required
def get_topics_for_team(team_id):
    if not current_user.is_authenticated:
        return jsonify({"error": "Authentication required"}), 401

    team = Team.query.get_or_404(team_id)
    
    if not team:
        return jsonify({"error": "Team not found"}), 404
    
    if current_user.id != team.host_id:

        topics = Topic.query.filter(Topic.team_id == team_id, Topic.participants.any(id=current_user.id)).all()
        if not topics:
         
            return jsonify({"error": "Access denied"}), 403
    else:
   
        topics = Topic.query.filter_by(team_id=team_id).all()
        
    topics_data = [topic.to_dict() for topic in topics]
    return jsonify(topics_data), 200

@main.route('/topics/<int:topic_id>/decision-criterias', methods=['POST'])
@login_required
def create_decision_criteria(topic_id):
    if not current_user.is_authenticated:
        return jsonify({"error": "Authentication required"}), 401
    
    data = request.get_json()
    name = data.get('title')
    description = data.get('description')
   
    if not name:
        return jsonify({"error": "Name is required."}), 400
    if not description:
        return jsonify({"error": "Description is required."}), 400

    if DecisionCriteria.query.filter_by(name=name, topic_id=topic_id).first():
        return jsonify({"error": "Decision criteria with the given name already exists for this topic"}), 400

    new_criteria = DecisionCriteria(name=name, description=description, topic_id=topic_id)
    db.session.add(new_criteria)
    
    existing_options = Option.query.filter_by(topic_id=topic_id).all()
    
    for option in existing_options:
        association = OptionDecisionCriteriaAssociation(option_id=option.id, decision_criteria_id=new_criteria.id)
        db.session.add(association)
        
    try:
        db.session.commit()
        return jsonify({"message": "Decision criteria created successfully.", "decisionCriteria": new_criteria.to_dict()}), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500
    

@main.route('/save-weights', methods=['POST'])
@login_required
def decision_criteria_weights_save():
    data = request.get_json()
    user_id = data['userId']
    topic_id = data['topicId']
    weights = data['weights']

    for criteria_id, weight in weights.items():
        user_criteria_weight = UserDecisionCriteriaWeight.query.filter_by(
            topic_id=topic_id,
            decision_criteria_id=criteria_id,
            user_id=user_id
        ).first()

        if user_criteria_weight:
            user_criteria_weight.weight = float(weight)
        else:
            new_weight = UserDecisionCriteriaWeight(
                user_id=user_id,
                topic_id=topic_id,
                decision_criteria_id=criteria_id,
                weight=float(weight)
            )
            db.session.add(new_weight)

    try:
        db.session.commit()
        return jsonify({"message": "User-specific weights updated successfully"}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500

@main.route('/topics/<int:topic_id>/decision-criterias', methods=['GET'])
@login_required
def get_decision_criterias_with_options(topic_id):
    topic = Topic.query.get(topic_id)
    if not topic:
        return jsonify({"error": "Topic not found"}), 404

    decision_criterias = DecisionCriteria.query.filter_by(topic_id=topic_id).all()

    decision_criterias_with_options = []
    for criteria in decision_criterias:
        criteria_dict = criteria.to_dict()
        decision_criterias_with_options.append(criteria_dict)

    return jsonify(decision_criterias_with_options), 200


@main.route('/topics/<int:topic_id>/options', methods=['POST'])
@login_required
def add_option_to_topic(topic_id):
    if not current_user.is_authenticated:
        return jsonify({"error": "Authentication required"}), 401
    
    data = request.get_json()
    description = data.get('description')

    if not description:
        return jsonify({"error": "Description is required"}), 400
    
    new_option = Option(description=description, topic_id=topic_id)
    db.session.add(new_option)

    decision_criterias = DecisionCriteria.query.filter_by(topic_id=topic_id).all()
    if not decision_criterias:
        return jsonify({"error": "No decision criteria found for this topic"}), 404
    
    for criteria in decision_criterias:
        association = OptionDecisionCriteriaAssociation(option=new_option, decision_criteria=criteria)
        db.session.add(association)

    try:
        db.session.commit()
        return jsonify({"message": "Option added successfully", "option": new_option.to_dict()}), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500

@main.route('/topics/<int:topic_id>/add-base-model', methods=['POST'])
@login_required
def add_base_model(topic_id):
    if not current_user.is_authenticated:
        return jsonify({"error": "Authentication required"}), 401

    data = request.get_json()
    model_name = data.get('name')

    if not model_name:
        return jsonify({"error": "Model name is required"}), 400

    existing_base_model = Option.query.filter_by(topic_id=topic_id, is_base_model=True).first()
    if existing_base_model:
        return jsonify({"error": "A base model already exists for this topic"}), 400
    
    new_option = Option(description=model_name, topic_id=topic_id, is_base_model=True)
    
    decision_criterias = DecisionCriteria.query.filter_by(topic_id=topic_id).all()
    if not decision_criterias:
        return jsonify({"error": "No decision criteria found for this topic"}), 404
    
    for criteria in decision_criterias:
        association = OptionDecisionCriteriaAssociation(option=new_option, decision_criteria=criteria)
        db.session.add(association)

    try:
        db.session.add(new_option)
        db.session.commit()
        return jsonify({"message": "Base model added successfully", "option": new_option.to_dict()}), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500

@main.route('/options/<int:option_id>/update-value', methods=['POST'])
@login_required
def update_option_value(option_id):
    if not current_user.is_authenticated:
        return jsonify({"error": "Authentication required"}), 401
    
    option = Option.query.get(option_id)
    if not option:
        return jsonify({'error': 'Option not found'}), 404

    data = request.get_json()
    user_id = data.get('userId')
    criteria_id = data.get('criteriaId')
    input_value = data.get('inputValue')
    
    association = OptionDecisionCriteriaAssociation.query.filter_by(
        option_id=option_id,
        decision_criteria_id=criteria_id
    ).first()
    
    if not association:
        return jsonify({"error": "Association not found"}), 404
    
    association.user_id = user_id
    association.input_value = input_value
    
    try:
        db.session.commit()
        return jsonify({'message': 'Option updated successfully', 'option': option.to_dict()}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

#-----------------------------------------------------------TOPSIS METHOD ----------------------------------------------------------------

# helper function to normalize the decision matrix
def norm(x, norm_type='v'):
    if norm_type =='v': #vector normalization
        norm_factor = np.sqrt(np.sum(np.square(x), axis=0))
        normalized_x = x / norm_factor
    else:  # Linear normalization
        max_values = np.amax(x, axis=0)
        normalized_x = x / max_values
    return normalized_x

# helper function to calculate the weighted normalized decision matrix
def mul_w(weights, matrix):
    return weights * matrix

#helper function to calcuate the ideal and anti-ideal solutions
def pos_neg_solution(matrix, criterion_type='m'):
    if criterion_type == 'm': #min/max criteria
        ideal_solution = np.amax(matrix, axis=0)
        anti_ideal_solution = np.amin(matrix, axis=0)
    else: #absolute criteria
        ideal_solution = np.ones(matrix.shape[1])
        anti_ideal_solution = np.zeros(matrix.shape[1])
    return ideal_solution, anti_ideal_solution

#helper function to calculate the distance to the ideal and anti-ideal solutions
def distance(matrix, ideal_solution, anti_ideal_solution):
    dist_to_ideal = np.sqrt(np.sum(np.square(matrix - ideal_solution), axis= 1))
    dist_to_anti_ideal = np.sqrt(np.sum(np.square(matrix - anti_ideal_solution), axis=1))
    return dist_to_ideal, dist_to_anti_ideal

# main TOPSIS function
def topsis(decision_matrix, weights, norm_type='v', criterion_type='m'):
    normalized_matrix = norm(decision_matrix, norm_type)
    weighted_matrix = mul_w(weights, normalized_matrix)
    ideal_solution, anti_ideal_solution = pos_neg_solution(weighted_matrix, criterion_type)
    dist_to_ideal, dist_to_anti_ideal = distance(weighted_matrix, ideal_solution, anti_ideal_solution)
    performance_score = dist_to_anti_ideal / (dist_to_ideal + dist_to_anti_ideal)
    return performance_score, normalized_matrix, weighted_matrix, ideal_solution, anti_ideal_solution, dist_to_ideal, dist_to_anti_ideal

def update_decision_criteria_weights(topic_id, average_weights):
    decision_criterias = DecisionCriteria.query.filter_by(topic_id=topic_id).all()
    for criteria, avg_weight in zip(decision_criterias, average_weights):
        criteria.weight = avg_weight
    db.session.commit()
    
def compute_weighted_criteria_weights(topic_id):
    decision_criterias = DecisionCriteria.query.filter_by(topic_id=topic_id).all()
    weighted_criteria_weights = []

    for criteria in decision_criterias:
        total_weighted_criteria_weight = 0
        total_member_weights = 0
        for member in Topic.query.get(topic_id).participants:
            member_weight = get_member_weight(member.id, topic_id)
            criterion_user_weight = get_user_criterion_weight(member.id, criteria.id)
            total_weighted_criteria_weight += criterion_user_weight * member_weight
            total_member_weights += member_weight

        weighted_average = total_weighted_criteria_weight / total_member_weights if total_member_weights > 0 else 0
        weighted_criteria_weights.append(weighted_average)

    return weighted_criteria_weights

def aggregate_user_inputs(option_id, criteria_id, topic_id):
    associations = OptionDecisionCriteriaAssociation.query.filter_by(
        option_id=option_id,
        decision_criteria_id=criteria_id
    ).all()

    weighted_input_total = 0
    total_criterion_weights = 0
    for assoc in associations:
        user_criterion_weight = get_user_criterion_weight(assoc.user_id, criteria_id)
        input_value = float(assoc.input_value) if assoc.input_value else 0
        weighted_input_total += input_value * user_criterion_weight
        total_criterion_weights += user_criterion_weight

    return weighted_input_total / total_criterion_weights if total_criterion_weights > 0 else 0

def get_user_criterion_weight(user_id, criteria_id):
    # Retrieve the criterion weight set by the user
    user_weight = UserDecisionCriteriaWeight.query.filter_by(
        user_id=user_id,
        decision_criteria_id=criteria_id
    ).first()
    return user_weight.weight if user_weight else 1  # Default to 1 if not set

def get_member_weight(user_id, topic_id):
    sql = "SELECT weight FROM topic_participants_association WHERE user_id = :user_id AND topic_id = :topic_id"
    result = db.session.execute(sql, {'user_id': user_id, 'topic_id': topic_id}).fetchone()
    return result['weight'] if result else 1

#------------------------------------------------------------------------------------------------------------------
'''
We loop through each DecisionCriteria for the topic_id to create the decision matrix where each row corresponds to an option and each column corresponds to a criteria.
We collect the weights for each DecisionCriteria.
We transpose the matrix so that each row represents an option and each column a criterion's input value.
We apply the TOPSIS method to the decision matrix and weights to get the final scores.
We sort the options by their calculated TOPSIS scores and update their ranks in the database.
'''

@main.route('/compute-rankings', methods=['POST'])
@login_required
def compute_rankings():
    if not current_user.is_authenticated:
        return jsonify({"error": "Authentication required"}), 401

    data = request.get_json()
    topic_id = data.get('topicId')
    
    topic = Topic.query.get(topic_id)
    team_size = topic.get_member_count()
    # Retrieve all options and the associated decision criteria for the topic
    options = Option.query.filter_by(topic_id=topic_id).all()
    decision_criterias = DecisionCriteria.query.filter_by(topic_id=topic_id).all()
    
    # Check if all votes have been cast
    if not Vote.are_all_votes_in(topic_id, team_size):
        return jsonify({"error": "Not all team members have cast their votes"}), 400

    each_criteria_weight = compute_weighted_criteria_weights(topic_id)
    update_decision_criteria_weights(topic_id, each_criteria_weight)
    # print(f"{each_criteria_weight}")
    # Construct the decision matrix and weights array
    decision_matrix = []
    
    for criteria in decision_criterias:
        criteria_column = [aggregate_user_inputs(option.id, criteria.id, topic_id) for option in options]
        decision_matrix.append(criteria_column)
        
    #convert lists to numpy arrays
    decision_matrix = np.array(decision_matrix).T #transpose to get correct shape
    weights = np.array(each_criteria_weight)
    
    # Perform TOPSIS calculation
    final_scores, norm_matrix, weighted_norm_matrix, ideal_sol, anti_ideal_sol, ideal_dist, anti_ideal_dist = topsis(decision_matrix, weights, 'v', 'm')

    for index, criteria in enumerate(decision_criterias):
        criteria.ideal_solution = float(ideal_sol[index])
        criteria.anti_ideal_solution = float(anti_ideal_sol[index])

    try:
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500
    
    # Assign TOPSIS scores to options and sort
    for option, score, norm_vals, weighted_vals, ideal_dist_val, anti_ideal_dist_val in zip(options, final_scores, norm_matrix, weighted_norm_matrix, ideal_dist, anti_ideal_dist):
        option.value = score
        # Convert NumPy arrays to JSON strings for storage
        option.normalized_decision_matrix = norm_vals.tolist()
        option.weighted_normalized_decision_matrix = weighted_vals.tolist()
        option.ideal_solution_distance = ideal_dist_val
        option.anti_ideal_solution_distance = anti_ideal_dist_val

    sorted_options = sorted(options, key=lambda option: option.value, reverse=True)
    
    try:
        for position, option in enumerate(sorted_options, start=1):
            rank = Rank.query.filter_by(position=position).first()
            if not rank:
                rank = Rank(position=position)
                db.session.add(rank)
        
            option.rank = rank    
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500
    
    ranked_options = [{
        'option_id': option.id,
        'description': option.description,
        'score': option.value,
        'position': option.rank.position
    } for option in sorted_options]
    return jsonify([option.to_dict() for option in sorted_options]), 200
  
    return jsonify(ranked_options), 200

@main.route('/topics/<int:topic_id>/ranked-options', methods=['GET'])
@login_required
def get_ranked_options(topic_id):
    options = Option.query.filter_by(topic_id=topic_id).all()
   
    decision_criterias = DecisionCriteria.query.filter_by(topic_id=topic_id).all()
    
    decision_criterias_dicts = [criteria.to_dict() for criteria in decision_criterias]
    
    ranked_options = sorted(options, key=lambda option: option.value, reverse=True)
    
    ranked_options_dicts = [option.to_dict() for option in ranked_options]
    
    response = {
        'ranked_options': ranked_options_dicts,
        'decision_criterias': decision_criterias_dicts
    }

    return jsonify(response)

@main.route('/topics/<int:topic_id>/user-inputs', methods=['GET'])
@login_required
def get_user_inputs(topic_id):
    votes = Vote.query.filter_by(topic_id=topic_id).all()
    user_inputs = []

    # Create matrix for each user input
    for vote in votes:
        user = User.query.get(vote.member_id)
        user_input_matrix = create_user_input_matrix(vote, topic_id)

        user_data = {
            'user_id': vote.member_id,
            'username': user.username,
            'matrix': user_input_matrix
        }
        user_inputs.append(user_data)

    return jsonify(user_inputs)

def create_user_input_matrix(vote, topic_id):
    # Fetch criteria weights and names for the user and topic
    criteria_weights = UserDecisionCriteriaWeight.query.filter_by(
        user_id=vote.member_id,
        topic_id=topic_id
    ).all()

    # Create a dictionary for criteria names and weights with criteria id as keys
    criteria_names = {criteria.decision_criteria_id: criteria.decision_criteria.name for criteria in criteria_weights}
    criteria_weight_values = {criteria.decision_criteria_id: criteria.weight for criteria in criteria_weights}

    # Create the header with criteria names. We use the dictionary to get the names by id
    header = ['Options/Criteria'] + [criteria_names[criteria_id] for criteria_id in criteria_names]
    matrix = [header]

    # Create a row for criteria weights
    weights_row = ['Criteria Weights'] + [criteria_weight_values.get(criteria_id, 'N/A') for criteria_id in criteria_names]
    matrix.append(weights_row)

    # Populate the matrix with option scores for each criteria
    for criteria_id, options in vote.values.items():
        for option in options:
            option_name = Option.query.get(option['option_id']).description
            row = next((r for r in matrix if r[0] == option_name), None)
            if not row:
                row = [option_name] + ['N/A'] * (len(header) - 1)
                matrix.append(row)
            # Ensure that the criteria_id is in the header
            criteria_index = header.index(criteria_names.get(int(criteria_id), 'N/A'))
            if criteria_index >= 0:
                row[criteria_index] = option['input_value']

    return matrix



# -----------------------------------------------------------------------User network------------------------------------------------------
@main.route('/users', methods=['GET'])
@login_required
def get_users():
    users = User.query.all()
    return jsonify([user.to_dict() for user in users]), 200

def update_topic_participant_weights(team_id):
    team_members = db.session.query(
        team_member_association.c.user_id,
        team_member_association.c.weight
    ).filter_by(team_id=team_id).all()

    for user_id, weight in team_members:
        db.session.query(topic_participants_association).filter(
            topic_participants_association.c.user_id == user_id
        ).update({'weight': weight})

    db.session.commit()

@main.route('/teams/<int:team_id>/add-member', methods=['POST'])
@login_required
def add_user_to_team(team_id):
    if not current_user.is_authenticated:
        return jsonify({"error": "Authentication required"}), 401
    # Extract the user ID from the request payload
    data = request.get_json()
    user_id = data.get('userId')

    if user_id is None:
        return jsonify({"error": "User ID is required"}), 400

    team = Team.query.get(team_id)
    user_to_add = User.query.get(user_id)
    # Check if the current user has permission to add members to the team
    if not team or not user_to_add:
        abort(404, description="Team or User not found")

    if current_user.id != team.host_id:
        abort(403, description="You do not have permission to add members to this team")
    
    if user_to_add in team.members:
        abort(400, description="User already a member of the team")

    
    try:
        team.members.append(user_to_add)
        db.session.commit()
        return jsonify({"message": "User added to team successfully"}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500
    
@main.route('/teams/<int:team_id>/details', methods=['GET'])
@login_required
def get_team_details(team_id):
    team = Team.query.get(team_id)
    if not team:
        return jsonify({"error": "Team not found"}), 404

    # Query the database for members and their weight in the team
    members_query = db.session.query(
        User, team_member_association.c.weight
    ).join(
        team_member_association, User.id == team_member_association.c.user_id
    ).filter(
        team_member_association.c.team_id == team_id
    )
    members_data = [
        {
            "id": member.id,
            "username": member.username,
            "email": member.email,
            "weight": weight
        } for member, weight in members_query
    ]
    
    team_data = {
        "id": team.id,
        "name": team.name,
        "description": team.description,
        "host": {
            "id": team.host.id,
            "username": team.host.username,
            "email": team.host.email
        },
        "members": members_data
    }

    return jsonify(team_data), 200

@main.route('/teams/<int:team_id>/topics/<int:topic_id>/add-member', methods=['POST'])
@login_required
def add_member_to_topic(team_id, topic_id):
    data = request.get_json()
    user_id = data.get('userId')

    team = Team.query.get(team_id)
    topic = Topic.query.get(topic_id)

    if not team or not topic:
        abort(404, description="Team or Topic not found")

    if topic.team_id != team.id:
        abort(400, description="Topic does not belong to the given Team")

    user_to_add = User.query.get(user_id)

    if current_user.id != team.host_id:
        abort(403, description="You do not have permission to add members to this topic")
        
    if not user_to_add or user_to_add not in team.members:
        abort(400, description="User not found in the team")
    
    existing_member = db.session.query(topic_participants_association).filter_by(
        topic_id=topic_id, user_id=user_id).first()
    if existing_member:
        abort(400, description="User already part of the topic")
    
    member_weight = db.session.query(team_member_association).filter(
        team_member_association.c.user_id == user_id,
        team_member_association.c.team_id == team_id
    ).first()

    if member_weight:
        weight = member_weight.weight
        
    existing_member = db.session.query(topic_participants_association).filter(
        topic_participants_association.c.topic_id == topic_id,
        topic_participants_association.c.user_id == user_id
    ).first()

    if not existing_member:
        db.session.execute(topic_participants_association.insert().values(
            topic_id=topic_id,
            user_id=user_id,
            weight=weight
    ))

    try:
        db.session.commit()
        return jsonify({"message": "User added to topic successfully"}), 200
    except Exception as e:
        db.session.rollback()
        abort(500, description=str(e))

@main.route('/topics/<int:topic_id>/members', methods=['GET'])
@login_required
def get_topic_members(topic_id):
    topic = Topic.query.get(topic_id)
    if not topic:
        return jsonify({"error": "Topic not found"}), 404

    members_data = db.session.query(
        User.id,
        User.username,
        User.email,
        topic_participants_association.c.weight
    ).join(
        topic_participants_association,
        topic_participants_association.c.user_id == User.id
    ).filter(
        topic_participants_association.c.topic_id == topic_id
    ).all()

    members_data = [
        {"id": member_id, "username": username, "email": email, "weight": weight}
        for member_id, username, email, weight in members_data
    ]
    
    return jsonify(members_data), 200

@main.route('/topic/<int:topic_id>/members/<int:user_id>/set-weight', methods=['POST'])
@login_required
def set_topic_member_weight(topic_id, user_id):
    if not current_user.is_authenticated:
        return jsonify({"error": "Authentication required"}), 401

    topic = Topic.query.get_or_404(topic_id)
    user_to_update = User.query.get_or_404(user_id)
    
    if current_user.id != topic.host_id:
        return jsonify({"error": "Only the team host can set member weights"}), 403
    
    data = request.get_json()
    new_weight = data.get('weight')
    # Direct SQL update to the association table
    result = db.session.execute(
        "UPDATE topic_participants_association SET weight = :weight WHERE topic_id = :topic_id AND user_id = :user_id",
        {'weight': new_weight, 'topic_id': topic_id, 'user_id': user_id}
    )

    try:
        db.session.commit()
        return jsonify({"message": "Member weight updated successfully"}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500
    
@main.route('/teams/<int:team_id>/members/<int:user_id>/set-weight', methods=['POST'])
@login_required
def set_member_weight(team_id, user_id):
    if not current_user.is_authenticated:
        return jsonify({"error": "Authentication required"}), 401

    team = Team.query.get_or_404(team_id)
    user_to_update = User.query.get_or_404(user_id)
    
    if current_user.id != team.host_id:
        return jsonify({"error": "Only the team host can set member weights"}), 403

    if user_to_update not in team.members:
        return jsonify({"error": "User is not a member of the team"}), 404
    
    data = request.get_json()
    new_weight = data.get('weight')
    
    result = db.session.execute(
        "UPDATE team_member_association SET weight = :weight WHERE team_id = :team_id AND user_id = :user_id",
        {'weight': new_weight, 'team_id': team_id, 'user_id': user_id}
    )

    try:
        db.session.commit()
        return jsonify({"message": "Member weight updated successfully"}), 200
        update_topic_participant_weights(team_id)
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500

@main.route('/record-vote', methods=['POST'])
@login_required
def record_vote():
    if not current_user.is_authenticated:
        return jsonify({"error": "Authentication required"}), 401

    data = request.get_json()
    topic_id = data.get('topicId')
    user_id = data.get('userId')
    vote_data = data.get('voteData')  # List of {optionId, criteriaId, inputValue}

    vote_values = {}
    
    user_weight = db.session.query(topic_participants_association.c.weight).filter(
        topic_participants_association.c.user_id == user_id,
        topic_participants_association.c.topic_id == topic_id
    ).scalar()

    for vote in vote_data:
        option_id = vote.get('optionId')
        criteria_id = vote.get('criteriaId')
        input_value = vote.get('inputValue')

        criteria_key = str(criteria_id)
        if criteria_key not in vote_values:
            vote_values[criteria_key] = []
            
        vote_values[str(criteria_id)].append({
            'option_id': option_id,
            'input_value': input_value
        })
        
    vote_instance = Vote.query.filter_by(
        member_id=user_id,
        topic_id=topic_id
    ).first()

    if vote_instance:
        vote_instance.values = vote_values
        vote_instance.member_weight = user_weight
        
    else:
        vote_instance = Vote(
            member_id=user_id,
            topic_id=topic_id,
            values=vote_values
        )
        db.session.add(vote_instance)

    try:
        db.session.commit()
        return jsonify({"message": "Votes recorded successfully"}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500


@main.route('/topics/<int:topic_id>/cast-vote', methods=['POST'])
@login_required
def cast_vote(topic_id):
    data = request.get_json()
    values = data.get('values')

    Vote.cast_vote(current_user.id, topic_id, values)

    if Vote.are_all_votes_in(topic_id, team_size):
        pass

    return jsonify({"message": "Vote cast successfully"}), 200


@main.route('/options/<int:option_id>', methods=['DELETE'])
@login_required
def delete_option(option_id):
    if not current_user.is_authenticated:
        return jsonify({"error": "Authentication required"}), 401
    
    option = Option.query.get(option_id)
    if not option:
        return jsonify({"error": "Option not found"}), 404

    try:
        db.session.delete(option)
        db.session.commit()
        return jsonify({"message": "Option deleted successfully"}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500
    
@main.route('/decision-criteria/<int:criteria_id>', methods=['DELETE'])
@login_required
def delete_decision_criteria(criteria_id):
    if not current_user.is_authenticated:
        return jsonify({"error": "Authentication required"}), 401

    criteria = DecisionCriteria.query.get_or_404(criteria_id)

    if criteria.topic.host_id != current_user.id:
        return jsonify({"error": "You do not have permission to delete this criteria"}), 403

    try:
        db.session.delete(criteria)
        db.session.commit()
        return jsonify({"message": "Criteria deleted successfully"}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500

from flask import jsonify, request

@main.route('/delete-topic/<int:topic_id>', methods=['DELETE'])
@login_required
def delete_topic(topic_id):
    try:

        topic = Topic.query.get(topic_id)
        if not topic:
            return jsonify({'error': 'Topic not found'}), 404

        topic.decision_criteria.delete() 
        Rank.query.filter_by(topic_id=topic_id).delete()
        
        db.session.execute(
            topic_participants_association.delete().where(
                topic_participants_association.c.topic_id == topic_id
            )
        )

        db.session.delete(topic)
        db.session.commit()
        return jsonify({'message': 'Topic and related data deleted successfully'}), 200

    except Exception as e:

        db.session.rollback()
        return jsonify({'error': str(e)}), 500
