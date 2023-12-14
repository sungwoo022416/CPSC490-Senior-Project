from flask import Flask, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from sqlalchemy import event, create_engine
from sqlalchemy.engine import Engine
from flask_cors import CORS
from flask_jwt_extended import JWTManager
from init import db, bcrypt, login_manager
from user_models import User
from datetime import timedelta

def create_app(config_name=None):
    app = Flask(__name__)
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///poll_database.db'
    app.config['SECRET_KEY'] = 'very_Secret_Key'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['SESSION_COOKIE_NAME'] = 'poll_user_log_in'
    app.config['SESSION_COOKIE_HTTPONLY'] = True
    app.config['SESSION_COOKIE_SECURE'] = True
    app.config['SESSION_COOKIE_SAMESITE'] = 'None'
    app.config['REMEMBER_COOKIE_DURATION'] = timedelta(days=14)
    
    
    #initialize extensions with the app
    db.init_app(app)
    bcrypt.init_app(app)
    login_manager.init_app(app)
    jwt = JWTManager(app)
    
    #user loader function for Flask-login
    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))
    
    #Enable SQLite foreign key support
    @event.listens_for(Engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()
        
    #Blueprints
    from routes import main as main_blueprint
    app.register_blueprint(main_blueprint)
    
    CORS(app, resources={r"/*": {"origins":"http://localhost:4200"}}, supports_credentials=True)

    return app
#create the flask app
app = create_app()

def initialize_database():
    print("Initializing database...")
    print("Database URI:", app.config['SQLALCHEMY_DATABASE_URI'])
    # Import models
    from user_models import User, Host, Member, Team, Topic
    from decision_criteria_models import DecisionCriteria, Option, OptionDecisionCriteriaAssociation, UserDecisionCriteriaWeight
    from vote_models import Vote
    from rank_models import Rank
    print("Creating database tables...")
    db.create_all()
    print("Database tables created.")
        
if __name__ == '__main__':
    with app.app_context():
        initialize_database() 
    app.run(debug=True)

