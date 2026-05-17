from sklearn.metrics.pairwise import pairwise_distances
from tensorflow.python.platform import gfile
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from keras.models import model_from_json
import numpy as np
import detect_and_align
import easygui
import cv2
import os
import datetime
import json
import time

class IdData:
    """Keeps track of known identities and calculates id matches"""

    def __init__(self, id_folder, mtcnn, sess, embeddings, images_placeholder, phase_train_placeholder, distance_treshold):
        print("Loading known identities: ", end="")
        self.distance_treshold = distance_treshold
        self.id_folder = id_folder
        self.mtcnn = mtcnn
        self.id_names = []
        self.embeddings = None

        os.makedirs(id_folder, exist_ok=True)
        ids = os.listdir(os.path.expanduser(id_folder))
        if not ids:
            return

        image_paths = []
        for id_name in ids:
            id_dir = os.path.join(id_folder, id_name)
            image_paths += [os.path.join(id_dir, img) for img in os.listdir(id_dir)]

        print("Found %d images in id folder" % len(image_paths))
        aligned_images, id_image_paths = self.detect_id_faces(image_paths)
        feed_dict = {images_placeholder: aligned_images, phase_train_placeholder: False}
        self.embeddings = sess.run(embeddings, feed_dict=feed_dict)

    def detect_id_faces(self, image_paths):
        aligned_images = []
        id_image_paths = []
        for image_path in image_paths:
            image = cv2.imread(os.path.expanduser(image_path), cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            face_patches, _, _ = detect_and_align.detect_faces(image, self.mtcnn)
            if len(face_patches) > 1:
                print("Warning: Found multiple faces in id image:", image_path)
            aligned_images += face_patches
            id_image_paths += [image_path] * len(face_patches)
            path = os.path.dirname(image_path)
            self.id_names += [os.path.basename(path)] * len(face_patches)

        return np.stack(aligned_images), id_image_paths

    def find_matching_ids(self, embs):
        if self.id_names:
            matching_ids = []
            matching_distances = []
            distance_matrix = pairwise_distances(embs, self.embeddings)
            for distance_row in distance_matrix:
                min_index = np.argmin(distance_row)
                if distance_row[min_index] < self.distance_treshold:
                    matching_ids.append(self.id_names[min_index])
                    matching_distances.append(distance_row[min_index])
                else:
                    matching_ids.append(None)
                    matching_distances.append(None)
        else:
            matching_ids = [None] * len(embs)
            matching_distances = [np.inf] * len(embs)
        return matching_ids, matching_distances

def load_model(model):
    model_exp = os.path.expanduser(model)
    if os.path.isfile(model_exp):
        print("Loading model filename:", model_exp)
        with gfile.FastGFile(model_exp, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name="")
    else:
        raise ValueError("Specify model file, not directory!")

class EmotionDetector:
    """Handles emotion detection using pre-trained Keras model"""
    
    def __init__(self, model_json_path, model_weights_path):
        # Load emotion model
        self.emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 
                             3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
        
        # Load json and create model
        with open(model_json_path, 'r') as json_file:
            loaded_model_json = json_file.read()
        self.emotion_model = model_from_json(loaded_model_json)
        
        # Load weights into new model
        self.emotion_model.load_weights(model_weights_path)
        print("✓ Emotion model loaded successfully")
        
    def detect_emotion(self, face_region):
        """Detect emotion from a face region"""
        try:
            # Check if face_region is float and convert to uint8 if needed
            if face_region.dtype != np.uint8:
                if face_region.dtype == np.float32 or face_region.dtype == np.float64:
                    # Assuming the image is in 0-1 range, scale to 0-255
                    if face_region.max() <= 1.0:
                        face_region = (face_region * 255).astype(np.uint8)
                    else:
                        face_region = face_region.astype(np.uint8)
                else:
                    face_region = face_region.astype(np.uint8)
            
            # Convert to grayscale for emotion detection
            if len(face_region.shape) == 3:
                gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            else:
                gray_face = face_region
            
            # Resize to 48x48 as required by emotion model
            cropped_img = cv2.resize(gray_face, (48, 48))
            cropped_img = cropped_img.reshape(1, 48, 48, 1)
            cropped_img = cropped_img.astype('float32') / 255.0
            
            # Predict emotion
            emotion_prediction = self.emotion_model.predict(cropped_img)
            maxindex = int(np.argmax(emotion_prediction))
            confidence = np.max(emotion_prediction)
            
            return self.emotion_dict[maxindex], confidence
        except Exception as e:
            print(f"Error in emotion detection: {e}")
            return "Unknown", 0.0

class VideoSummarizer:
    """Summarizes video based on specific person and emotion"""
    
    def __init__(self, mtcnn, sess, id_data, images_placeholder, embeddings, phase_train_placeholder, emotion_detector):
        self.mtcnn = mtcnn
        self.sess = sess
        self.id_data = id_data
        self.images_placeholder = images_placeholder
        self.embeddings = embeddings
        self.phase_train_placeholder = phase_train_placeholder
        self.emotion_detector = emotion_detector
        self.summary_data = []
        
    def convert_face_patch_to_uint8(self, face_patch):
        """Convert face patch from MTCNN to uint8 format"""
        # MTCNN returns float images in RGB format
        if face_patch.dtype == np.float32 or face_patch.dtype == np.float64:
            # Scale from 0-1 to 0-255
            if face_patch.max() <= 1.0:
                face_patch = (face_patch * 255).astype(np.uint8)
            else:
                face_patch = face_patch.astype(np.uint8)
        else:
            face_patch = face_patch.astype(np.uint8)
        
        # Convert RGB to BGR for OpenCV
        if len(face_patch.shape) == 3:
            face_patch = cv2.cvtColor(face_patch, cv2.COLOR_RGB2BGR)
        
        return face_patch
    
    def process_video_for_summary(self, video_path, target_person, target_emotion):
        """Process video and extract frames where target person shows target emotion"""
        print(f"\n🔍 Looking for '{target_person}' showing '{target_emotion}' emotion...")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Error: Cannot open video file {video_path}")
            return None, None
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📊 Video Info: {frame_width}x{frame_height} @ {fps:.1f} FPS, {total_frames} frames")
        
        # Create summary video
        safe_person = "".join(c for c in target_person if c.isalnum() or c in (' ', '_')).rstrip()
        safe_emotion = "".join(c for c in target_emotion if c.isalnum() or c in (' ', '_')).rstrip()
        summary_video_path = f"{safe_person}_{safe_emotion}_summary.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        summary_writer = cv2.VideoWriter(summary_video_path, fourcc, fps, (frame_width, frame_height))
        
        if not summary_writer.isOpened():
            print(f"❌ Error: Cannot create video writer for {summary_video_path}")
            cap.release()
            return None, None
        
        # Create directory for summary frames
        summary_dir = f"summary_{safe_person}_{safe_emotion}"
        os.makedirs(summary_dir, exist_ok=True)
        
        frame_count = 0
        matched_frames = 0
        start_time = time.time()
        
        print("\n⏳ Processing video... (Press 'q' to stop)")
        print("-" * 50)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            display_frame = frame.copy()
            
            # Convert frame to RGB for MTCNN
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces using MTCNN
            try:
                face_patches, padded_bounding_boxes, landmarks = detect_and_align.detect_faces(frame_rgb, self.mtcnn)
            except Exception as e:
                print(f"Warning: Face detection error on frame {frame_count}: {e}")
                face_patches = []
                padded_bounding_boxes = []
            
            match_found = False
            
            if len(face_patches) > 0:
                face_patches = np.stack(face_patches)
                feed_dict = {
                    self.images_placeholder: face_patches, 
                    self.phase_train_placeholder: False
                }
                embs = self.sess.run(self.embeddings, feed_dict=feed_dict)
                
                # Get matching IDs
                matching_ids, matching_distances = self.id_data.find_matching_ids(embs)
                
                # Process each detected face
                for i, (bb, matching_id, dist, face_patch) in enumerate(zip(
                    padded_bounding_boxes, matching_ids, matching_distances, face_patches)):
                    
                    # Convert face patch to proper format for emotion detection
                    face_patch_bgr = self.convert_face_patch_to_uint8(face_patch)
                    
                    # Detect emotion for this face
                    emotion, confidence = self.emotion_detector.detect_emotion(face_patch_bgr)
                    
                    # Check if this matches our target
                    person_match = False
                    if matching_id:
                        # Case-insensitive comparison
                        person_match = matching_id.lower() == target_person.lower()
                    
                    emotion_match = emotion.lower() == target_emotion.lower()
                    
                    if person_match and emotion_match and confidence > 0.5:
                        match_found = True
                        matched_frames += 1
                        
                        # Get timestamp
                        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                        
                        # Save this frame as image
                        frame_filename = f"frame_{frame_count:06d}_{timestamp:.1f}s.jpg"
                        cv2.imwrite(os.path.join(summary_dir, frame_filename), frame)
                        
                        # Add to summary data
                        self.summary_data.append({
                            'frame_number': frame_count,
                            'timestamp_seconds': round(timestamp, 2),
                            'confidence': round(float(confidence), 3),
                            'person': matching_id,
                            'emotion': emotion,
                            'distance': float(dist) if dist else 0.0,
                            'bounding_box': [int(x) for x in bb]
                        })
                        
                        # Draw highlight on frame
                        cv2.rectangle(display_frame, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0), 4)
                        cv2.putText(display_frame, "MATCH!", (bb[0], max(bb[1]-10, 20)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                        
                        # Draw info box
                        info_text = f"{matching_id} - {emotion} ({confidence:.2f})"
                        cv2.putText(display_frame, info_text, (bb[0], bb[3]+30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Draw regular detection (for display only)
                    else:
                        color = (0, 255, 0) if matching_id else (0, 0, 255)
                        name_text = f"{matching_id if matching_id else 'Unknown'}"
                        if matching_id and dist is not None:
                            name_text += f" ({dist:.2f})"
                        
                        emotion_text = f"{emotion} ({confidence:.2f})"
                        
                        y_offset = bb[3] + 20
                        cv2.putText(display_frame, name_text, (bb[0], y_offset),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        
                        cv2.putText(display_frame, emotion_text, (bb[0], y_offset + 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # If match found, add frame to summary video
            if match_found:
                summary_writer.write(frame)
            
            # Display progress every 50 frames
            if frame_count % 50 == 0:
                elapsed = time.time() - start_time
                fps_processed = frame_count / elapsed if elapsed > 0 else 0
                percent_complete = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                
                progress_bar = "█" * int(percent_complete / 2) + "░" * (50 - int(percent_complete / 2))
                print(f"\rProgress: [{progress_bar}] {percent_complete:.1f}% | "
                      f"Frames: {frame_count}/{total_frames} | "
                      f"Matches: {matched_frames} | "
                      f"Speed: {fps_processed:.1f} FPS", end="")
            
            # Display frame
            cv2.imshow(f"Searching: {target_person} - {target_emotion}", display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n\n⏹️ Processing interrupted by user")
                break
        
        # Release resources
        cap.release()
        summary_writer.release()
        cv2.destroyAllWindows()
        
        print(f"\n\n✅ Processing complete! Processed {frame_count} frames.")
        
        # Generate summary report
        self.generate_summary_report(target_person, target_emotion, video_path, summary_video_path, summary_dir)
        
        return summary_video_path, summary_dir
    
    def generate_summary_report(self, target_person, target_emotion, video_path, summary_video_path, summary_dir):
        """Generate a detailed summary report"""
        print("\n" + "="*60)
        print("📊 SUMMARY REPORT")
        print("="*60)
        
        total_matches = len(self.summary_data)
        
        if total_matches == 0:
            print(f"❌ No matches found for '{target_person}' showing '{target_emotion}' emotion.")
            print("="*60)
            return
        
        print(f"\n🎯 Target: {target_person} showing {target_emotion} emotion")
        print(f"📁 Source video: {os.path.basename(video_path)}")
        print(f"✅ Total matches found: {total_matches}")
        
        # Calculate statistics
        if total_matches > 0:
            confidences = [match['confidence'] for match in self.summary_data]
            avg_confidence = np.mean(confidences)
            max_confidence = max(confidences)
            
            timestamps = [match['timestamp_seconds'] for match in self.summary_data]
            min_time = min(timestamps)
            max_time = max(timestamps)
            
            distances = [match['distance'] for match in self.summary_data if match['distance'] > 0]
            avg_distance = np.mean(distances) if distances else 0
            
            print(f"📈 Average confidence: {avg_confidence:.3f}")
            print(f"🏆 Maximum confidence: {max_confidence:.3f}")
            print(f"⏱️ Time range: {min_time:.1f}s to {max_time:.1f}s")
            print(f"📏 Average face distance: {avg_distance:.3f}")
        
        print(f"🎬 Summary video: {summary_video_path}")
        print(f"🖼️ Frames directory: {summary_dir}/")
        
        # Create timeline
        print("\n⏰ Timeline of matches (first 10):")
        for i, match in enumerate(self.summary_data[:10]):
            mins, secs = divmod(match['timestamp_seconds'], 60)
            print(f"  {i+1:2d}. Frame {match['frame_number']:5d}: "
                  f"{int(mins):02d}:{secs:05.2f} | "
                  f"Conf: {match['confidence']:.3f} | "
                  f"Dist: {match['distance']:.3f}")
        
        if total_matches > 10:
            print(f"  ... and {total_matches - 10} more matches")
        
        # Save detailed report to file
        report_data = {
            'analysis_date': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'target_person': target_person,
            'target_emotion': target_emotion,
            'source_video': os.path.basename(video_path),
            'source_video_path': video_path,
            'summary_video': summary_video_path,
            'frames_directory': summary_dir,
            'total_matches': total_matches,
            'statistics': {
                'average_confidence': float(avg_confidence) if total_matches > 0 else 0,
                'maximum_confidence': float(max_confidence) if total_matches > 0 else 0,
                'average_distance': float(avg_distance) if total_matches > 0 else 0,
                'time_range_seconds': {
                    'start': float(min_time) if total_matches > 0 else 0,
                    'end': float(max_time) if total_matches > 0 else 0,
                    'duration': float(max_time - min_time) if total_matches > 1 else 0
                }
            },
            'matches': self.summary_data
        }
        
        safe_person = "".join(c for c in target_person if c.isalnum() or c in (' ', '_')).rstrip()
        safe_emotion = "".join(c for c in target_emotion if c.isalnum() or c in (' ', '_')).rstrip()
        report_filename = f"report_{safe_person}_{safe_emotion}.json"
        
        with open(report_filename, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\n📄 Detailed report saved as: {report_filename}")
        print("="*60)

def main():
    """Main function with user interaction"""
    
    print("="*60)
    print("🎥 VIDEO SUMMARIZATION SYSTEM")
    print("👤 Face Recognition + 😊 Emotion Detection")
    print("="*60)
    
    # Step 1: Ask for video file
    print("\n📁 Step 1: Select video file")
    video_file = easygui.fileopenbox(title="Select video file", 
                                     filetypes=[["*.mp4", "MP4 files"],
                                                ["*.avi", "AVI files"],
                                                ["*.mov", "MOV files"],
                                                ["*.mkv", "MKV files"]])
    
    if not video_file:
        print("❌ No video file selected. Exiting.")
        return
    
    print(f"✓ Selected: {os.path.basename(video_file)}")
    
    # Step 2: Ask for target person name
    print("\n👤 Step 2: Enter target person name")
    print("(This should match the name in your face recognition database)")
    target_person = easygui.enterbox("Enter the name of person to search for:", 
                                     "Target Person", "")
    
    if not target_person:
        print("❌ No person name entered. Exiting.")
        return
    
    print(f"✓ Searching for: {target_person}")
    
    # Step 3: Ask for target emotion
    print("\n😊 Step 3: Select target emotion")
    emotions = ["Happy", "Sad", "Angry", "Surprised", "Neutral", "Fearful", "Disgusted"]
    target_emotion = easygui.choicebox("Select the emotion to search for:", 
                                       "Target Emotion", emotions)
    
    if not target_emotion:
        print("❌ No emotion selected. Exiting.")
        return
    
    print(f"✓ Emotion to detect: {target_emotion}")
    
    print(f"\n🎯 Target: {target_person} showing {target_emotion} emotion")
    print(f"📹 Video: {os.path.basename(video_file)}")
    print("\n🚀 Initializing systems...")
    
    # Initialize TensorFlow session
    with tf.Graph().as_default():
        with tf.Session() as sess:
            try:
                # Initialize face recognition
                print("🔧 Loading face recognition model...")
                mtcnn = detect_and_align.create_mtcnn(sess, None)
                
                # Load face recognition model
                model_path = '20170512-110547/20170512-110547.pb'
                if not os.path.exists(model_path):
                    print(f"❌ Error: Face recognition model not found at {model_path}")
                    print("Please download the model and place it in the correct directory.")
                    return
                
                load_model(model_path)
                images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
                
                # Initialize ID database
                id_folder = '20170512-110547/ids'
                os.makedirs(id_folder, exist_ok=True)
                id_data = IdData(id_folder, mtcnn, sess, embeddings, 
                                 images_placeholder, phase_train_placeholder, 1.0)
                
                print(f"✓ Loaded {len(id_data.id_names)} known identities")
                
                # Initialize emotion detection
                print("🔧 Loading emotion detection model...")
                emotion_json = 'model/emotion_model.json'
                emotion_weights = 'model/emotion_model.h5'
                
                if not os.path.exists(emotion_json):
                    print(f"❌ Error: Emotion model JSON not found at {emotion_json}")
                    return
                if not os.path.exists(emotion_weights):
                    print(f"❌ Error: Emotion model weights not found at {emotion_weights}")
                    return
                
                emotion_detector = EmotionDetector(emotion_json, emotion_weights)
                
                # Create summarizer
                summarizer = VideoSummarizer(
                    mtcnn=mtcnn,
                    sess=sess,
                    id_data=id_data,
                    images_placeholder=images_placeholder,
                    embeddings=embeddings,
                    phase_train_placeholder=phase_train_placeholder,
                    emotion_detector=emotion_detector
                )
                
                print("\n✅ All systems initialized successfully!")
                print("▶️ Starting video analysis...")
                print("ℹ️ Press 'q' to stop processing early")
                print("-" * 60)
                
                # Process video and create summary
                summary_video, summary_dir = summarizer.process_video_for_summary(
                    video_file, target_person, target_emotion
                )
                
                if summary_video and os.path.exists(summary_video):
                    # Ask if user wants to play the summary
                    play_summary = easygui.ynbox(f"Analysis complete!\n\nFound {len(summarizer.summary_data)} matches.\n\nDo you want to play the summary video?", "Analysis Complete")
                    
                    if play_summary:
                        print("\n▶️ Playing summary video... (Press 'q' to close)")
                        cap = cv2.VideoCapture(summary_video)
                        
                        while True:
                            ret, frame = cap.read()
                            if not ret:
                                break
                            
                            cv2.imshow(f"Summary: {target_person} - {target_emotion}", frame)
                            if cv2.waitKey(30) & 0xFF == ord('q'):
                                break
                        
                        cap.release()
                        cv2.destroyAllWindows()
                    
                    print("\n" + "="*60)
                    print("🎉 ANALYSIS COMPLETE!")
                    print("="*60)
                    print(f"📊 Matches found: {len(summarizer.summary_data)}")
                    print(f"🎬 Summary video: {summary_video}")
                    print(f"📁 Frames saved in: {summary_dir}")
                    print(f"📄 Report saved as: report_{target_person}_{target_emotion}.json")
                    print("="*60)
                else:
                    print("\n❌ No summary video was created. Check for errors above.")
                
            except Exception as e:
                print(f"\n❌ Error during initialization: {e}")
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    # Check for required files
    required_files = [
        '20170512-110547/20170512-110547.pb',
        'model/emotion_model.json',
        'model/emotion_model.h5'
    ]
    
    print("🔍 Checking for required files...")
    missing_files = []
    for f in required_files:
        if os.path.exists(f):
            print(f"✓ {f}")
        else:
            print(f"❌ {f} - NOT FOUND")
            missing_files.append(f)
    
    if missing_files:
        print("\n❌ ERROR: Missing required files!")
        print("Please ensure all model files are in the correct locations.")
        print("\nDirectory structure should be:")
        print("├── 20170512-110547/")
        print("│   ├── 20170512-110547.pb")
        print("│   └── ids/ (optional - for known faces)")
        print("├── model/")
        print("│   ├── emotion_model.json")
        print("│   └── emotion_model.h5")
        print("└── final.py (this file)")
    else:
        print("\n✅ All required files found!")
        main()