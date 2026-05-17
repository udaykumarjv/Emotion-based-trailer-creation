from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify, send_file, Response
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os
from datetime import datetime
import json
import uuid
from video_processor import VideoSummarizer
import threading
import traceback
import mimetypes
from threading import Event
from sqlalchemy.orm import Session

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here-change-in-production'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///app.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max upload

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('summaries', exist_ok=True)
os.makedirs('static/images', exist_ok=True)

db = SQLAlchemy(app)

# Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    analyses = db.relationship('Analysis', backref='user', lazy=True)

class Analysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    video_filename = db.Column(db.String(200), nullable=False)
    target_person = db.Column(db.String(100), nullable=False)
    target_emotion = db.Column(db.String(50), nullable=False)
    summary_video = db.Column(db.String(200))
    frames_directory = db.Column(db.String(200))
    report_file = db.Column(db.String(200))
    total_matches = db.Column(db.Integer, default=0)
    status = db.Column(db.String(20), default='pending')  # pending, processing, completed, failed, no_matches
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    completed_at = db.Column(db.DateTime)

# Global variables
processing_status = {}
stop_events = {}

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password_hash, password):
            session['user_id'] = user.id
            session['username'] = user.username
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password', 'error')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return redirect(url_for('register'))
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists', 'error')
            return redirect(url_for('register'))
        
        if User.query.filter_by(email=email).first():
            flash('Email already registered', 'error')
            return redirect(url_for('register'))
        
        hashed_password = generate_password_hash(password)
        new_user = User(username=username, email=email, password_hash=hashed_password)
        
        db.session.add(new_user)
        db.session.commit()
        
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        flash('Please login first', 'error')
        return redirect(url_for('login'))
    
    user = User.query.get(session['user_id'])
    recent_analyses = Analysis.query.filter_by(user_id=user.id).order_by(Analysis.created_at.desc()).limit(5).all()
    
    return render_template('dashboard.html', user=user, analyses=recent_analyses)

@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    if 'user_id' not in session:
        flash('Please login first', 'error')
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        # Check if video file is uploaded
        if 'video' not in request.files:
            flash('No video file selected', 'error')
            return redirect(request.url)
        
        video_file = request.files['video']
        target_person = request.form.get('target_person')
        target_emotion = request.form.get('target_emotion')
        
        if video_file.filename == '':
            flash('No video file selected', 'error')
            return redirect(request.url)
        
        if not target_person or not target_emotion:
            flash('Please enter target person and select emotion', 'error')
            return redirect(request.url)
        
        # Save uploaded file
        filename = secure_filename(f"{uuid.uuid4()}_{video_file.filename}")
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        video_file.save(video_path)
        
        # Create analysis record
        analysis = Analysis(
            user_id=session['user_id'],
            video_filename=filename,
            target_person=target_person,
            target_emotion=target_emotion,
            status='processing'
        )
        db.session.add(analysis)
        db.session.commit()
        
        # Create stop event for this analysis
        stop_event = Event()
        stop_events[analysis.id] = stop_event
        
        # Start background processing
        try:
            thread = threading.Thread(
                target=process_video_background,
                args=(video_path, target_person, target_emotion, analysis.id, session['user_id'], stop_event)
            )
            thread.daemon = True
            thread.start()
            
            flash('Analysis started! You can stop processing by pressing q in the processing window.', 'success')
            return redirect(url_for('results', analysis_id=analysis.id))
            
        except Exception as e:
            analysis.status = 'failed'
            db.session.commit()
            flash(f'Error starting analysis: {str(e)}', 'error')
            return redirect(url_for('analyze'))
    
    return render_template('analyze.html')

@app.route('/results/<int:analysis_id>')
def results(analysis_id):
    if 'user_id' not in session:
        flash('Please login first', 'error')
        return redirect(url_for('login'))
    
    # Use session.get() instead of query.get() to avoid deprecation warning
    with Session(db.engine) as session_obj:
        analysis = session_obj.get(Analysis, analysis_id)
    
    if not analysis:
        flash('Analysis not found', 'error')
        return redirect(url_for('dashboard'))
    
    # Check if user owns this analysis
    if analysis.user_id != session['user_id']:
        flash('Access denied', 'error')
        return redirect(url_for('dashboard'))
    
    # Load report data if exists
    report_data = None
    if analysis.report_file and os.path.exists(analysis.report_file):
        try:
            with open(analysis.report_file, 'r') as f:
                report_data = json.load(f)
        except:
            report_data = None
    
    # Extract relative video path for serving
    video_path_for_web = None
    if analysis.summary_video:
        # Convert backslashes to forward slashes for web compatibility
        video_path = analysis.summary_video.replace('\\', '/')
        
        # Extract just the filename and the parent directory name
        # The video is stored in: summaries/analysis_X_timestamp/filename.mp4
        # We need to pass the full relative path from summaries/
        if video_path.startswith('summaries/'):
            # Remove 'summaries/' prefix for the video_feed route
            video_path_for_web = video_path[10:]  # Remove 'summaries/'
        else:
            # If it doesn't start with summaries/, use the basename
            video_path_for_web = os.path.basename(video_path)
    
    return render_template('results.html', 
                         analysis=analysis, 
                         report=report_data,
                         video_path=video_path_for_web)  # Changed from video_filename to video_path

@app.route('/history')
def history():
    if 'user_id' not in session:
        flash('Please login first', 'error')
        return redirect(url_for('login'))
    
    user = User.query.get(session['user_id'])
    analyses = Analysis.query.filter_by(user_id=user.id).order_by(Analysis.created_at.desc()).all()
    
    return render_template('history.html', analyses=analyses)
@app.route('/video/<path:filename>')
def video_feed(filename):
    """Serve video files securely"""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    # Try to find the video in summaries directory
    video_path = None
    
    # The filename includes the nested path: analysis_X_timestamp/video.mp4
    full_path = os.path.join('summaries', filename)
    
    if os.path.exists(full_path):
        video_path = full_path
    else:
        # Try to find the file by walking through summaries
        for root, dirs, files in os.walk('summaries'):
            if filename in files:
                video_path = os.path.join(root, filename)
                break
            # Also check if filename is the full path
            basename = os.path.basename(filename)
            if basename in files:
                video_path = os.path.join(root, basename)
                break
    
    if video_path and os.path.exists(video_path):
        # Determine MIME type
        mime_type, _ = mimetypes.guess_type(video_path)
        if not mime_type:
            mime_type = 'video/mp4'
        
        # Get file size
        file_size = os.path.getsize(video_path)
        
        # Create response
        def generate():
            with open(video_path, 'rb') as f:
                while True:
                    chunk = f.read(8192)
                    if not chunk:
                        break
                    yield chunk
        
        headers = {
            'Content-Length': str(file_size),
            'Content-Type': mime_type,
            'Accept-Ranges': 'bytes'
        }
        
        return Response(generate(), 200, headers)
    else:
        print(f"Video not found: {filename}")
        print(f"Full path attempted: {full_path}")
        return "Video not found", 404
@app.route('/download/<path:filename>')
def download(filename):
    if 'user_id' not in session:
        flash('Please login first', 'error')
        return redirect(url_for('login'))
    
    # Try to find the file with nested path
    file_path = None
    
    # Check with full path including summaries/
    summaries_path = os.path.join('summaries', filename)
    if os.path.exists(summaries_path):
        file_path = summaries_path
    else:
        # Walk through summaries to find the file
        for root, dirs, files in os.walk('summaries'):
            # Check if filename matches any file
            for file in files:
                if file == filename or filename.endswith(file):
                    file_path = os.path.join(root, file)
                    break
            if file_path:
                break
    
    if file_path and os.path.exists(file_path):
        # Get just the filename for download
        download_name = os.path.basename(file_path)
        return send_file(file_path, as_attachment=True, download_name=download_name)
    
    flash('File not found', 'error')
    return redirect(url_for('dashboard'))

@app.route('/delete_analysis/<int:analysis_id>')
def delete_analysis(analysis_id):
    if 'user_id' not in session:
        flash('Please login first', 'error')
        return redirect(url_for('login'))
    
    with Session(db.engine) as session_obj:
        analysis = session_obj.get(Analysis, analysis_id)
        
        if not analysis:
            flash('Analysis not found', 'error')
            return redirect(url_for('dashboard'))
        
        # Check if user owns this analysis
        if analysis.user_id != session['user_id']:
            flash('Access denied', 'error')
            return redirect(url_for('dashboard'))
        
        # Delete files
        try:
            if analysis.summary_video and os.path.exists(analysis.summary_video):
                os.remove(analysis.summary_video)
            if analysis.report_file and os.path.exists(analysis.report_file):
                os.remove(analysis.report_file)
            
            # Delete frames directory if exists
            if analysis.frames_directory and os.path.exists(analysis.frames_directory):
                import shutil
                shutil.rmtree(analysis.frames_directory)
        except Exception as e:
            print(f"Error deleting files: {e}")
        
        # Delete from database
        session_obj.delete(analysis)
        session_obj.commit()
    
    flash('Analysis deleted successfully', 'success')
    return redirect(url_for('history'))

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully', 'success')
    return redirect(url_for('index'))

# API endpoint for checking analysis status
@app.route('/api/analysis_status/<int:analysis_id>')
def analysis_status(analysis_id):
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    with Session(db.engine) as session_obj:
        analysis = session_obj.get(Analysis, analysis_id)
        
        if not analysis:
            return jsonify({'error': 'Analysis not found'}), 404
        
        # Check if user owns this analysis
        if analysis.user_id != session['user_id']:
            return jsonify({'error': 'Access denied'}), 403
        
        status = processing_status.get(analysis_id, analysis.status)
        
        return jsonify({
            'status': status,
            'total_matches': analysis.total_matches,
            'created_at': analysis.created_at.isoformat() if analysis.created_at else None,
            'completed_at': analysis.completed_at.isoformat() if analysis.completed_at else None
        })

@app.route('/stop_processing/<int:analysis_id>')
def stop_processing(analysis_id):
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    with Session(db.engine) as session_obj:
        analysis = session_obj.get(Analysis, analysis_id)
        
        if not analysis:
            return jsonify({'error': 'Analysis not found'}), 404
        
        # Check if user owns this analysis
        if analysis.user_id != session['user_id']:
            return jsonify({'error': 'Access denied'}), 403
        
        # Set stop event for this analysis
        if analysis_id in stop_events:
            stop_events[analysis_id].set()
            return jsonify({'status': 'stopping'})
        
        return jsonify({'status': 'not_processing'})

def process_video_background(video_path, target_person, target_emotion, analysis_id, user_id, stop_event=None):
    """Background processing function with stop event support"""
    try:
        # Update status
        processing_status[analysis_id] = 'processing'
        
        summarizer = VideoSummarizer()
        
        # Initialize summarizer
        if not summarizer.initialize_face_recognition():
            raise Exception("Failed to initialize face recognition")
        
        if not summarizer.initialize_emotion_detection():
            print("Warning: Emotion detection not available")
        
        # Process video with stop event support
        summary_video, frames_dir, report_file, total_matches = summarizer.process_video(
            video_path, target_person, target_emotion, analysis_id, stop_event
        )
        
        # Update database with relative paths
        with app.app_context():
            with Session(db.engine) as session_obj:
                analysis = session_obj.get(Analysis, analysis_id)
                if analysis and analysis.user_id == user_id:
                    # Store relative paths for web serving
                    analysis.summary_video = summary_video.replace('\\', '/')
                    analysis.frames_directory = frames_dir.replace('\\', '/')
                    analysis.report_file = report_file.replace('\\', '/')
                    analysis.total_matches = total_matches
                    analysis.status = 'completed' if total_matches > 0 else 'no_matches'
                    analysis.completed_at = datetime.utcnow()
                    session_obj.commit()
        
        processing_status[analysis_id] = 'completed'
        
    except Exception as e:
        print(f"Error in background processing: {e}")
        traceback.print_exc()
        
        with app.app_context():
            with Session(db.engine) as session_obj:
                analysis = session_obj.get(Analysis, analysis_id)
                if analysis and analysis.user_id == user_id:
                    analysis.status = 'failed'
                    session_obj.commit()
        
        processing_status[analysis_id] = 'failed'
        
    finally:
        # Cleanup
        if 'summarizer' in locals():
            summarizer.cleanup()
        
        # Clean up stop event
        if analysis_id in stop_events:
            del stop_events[analysis_id]
        if analysis_id in processing_status:
            del processing_status[analysis_id]

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    
    print("="*60)
    print("FaceEmotion AI Flask Application")
    print("="*60)
    print("\nChecking for required files...")
    
    # Check for required files
    required_files = [
        ('Face Recognition Model', '20170512-110547/20170512-110547.pb'),
        ('Emotion Model JSON', 'model/emotion_model.json'),
        ('Emotion Model Weights', 'model/emotion_model.h5')
    ]
    
    all_files_exist = True
    for file_desc, file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ {file_desc}: {file_path}")
        else:
            print(f"✗ {file_desc}: {file_path} - NOT FOUND")
            all_files_exist = False
    
    if not all_files_exist:
        print("\n⚠️ Warning: Some required files are missing.")
        print("The application may not work correctly.")
        print("\nDirectory structure should be:")
        print("├── 20170512-110547/")
        print("│   ├── 20170512-110547.pb")
        print("│   └── ids/ (optional - for known faces)")
        print("├── model/")
        print("│   ├── emotion_model.json")
        print("│   └── emotion_model.h5")
        print("└── haarcascades/ (optional)")
        print("    └── haarcascade_frontalface_default.xml")
    else:
        print("\n✅ All required files found!")
    
    print("\nStarting Flask application...")
    print("Access the application at: http://localhost:5000")
    print("="*60)
    
    app.run(debug=True, port=5000)