import os
import cv2
import json
import numpy as np
from datetime import datetime
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from keras.models import model_from_json
from sklearn.metrics.pairwise import pairwise_distances
import detect_and_align
import time

class IdData:
    """Keeps track of known identities and calculates id matches"""
    
    def __init__(self, id_folder, mtcnn, sess, embeddings, images_placeholder, phase_train_placeholder, distance_threshold=1.0):
        print("Loading known identities...")
        self.distance_threshold = distance_threshold
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

        print(f"Found {len(image_paths)} images in id folder")
        aligned_images, id_image_paths = self.detect_id_faces(image_paths)
        
        if len(aligned_images) > 0:
            feed_dict = {images_placeholder: aligned_images, phase_train_placeholder: False}
            self.embeddings = sess.run(embeddings, feed_dict=feed_dict)
            
            for image_path in id_image_paths:
                path = os.path.dirname(image_path)
                self.id_names.append(os.path.basename(path))

    def detect_id_faces(self, image_paths):
        aligned_images = []
        id_image_paths = []
        for image_path in image_paths:
            try:
                image = cv2.imread(os.path.expanduser(image_path), cv2.IMREAD_COLOR)
                if image is None:
                    continue
                    
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                face_patches, _, _ = detect_and_align.detect_faces(image, self.mtcnn)
                
                if len(face_patches) > 0:
                    aligned_images.append(face_patches[0])  # Use first face only
                    id_image_paths.append(image_path)
                    
                    if len(face_patches) > 1:
                        print(f"Warning: Found multiple faces in id image: {image_path}")
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue

        if aligned_images:
            return np.stack(aligned_images), id_image_paths
        return np.array([]), []

    def find_matching_ids(self, embs):
        if self.embeddings is not None and len(self.embeddings) > 0:
            matching_ids = []
            matching_distances = []
            distance_matrix = pairwise_distances(embs, self.embeddings)
            
            for distance_row in distance_matrix:
                if len(distance_row) > 0:
                    min_index = np.argmin(distance_row)
                    if distance_row[min_index] < self.distance_threshold:
                        matching_ids.append(self.id_names[min_index])
                        matching_distances.append(distance_row[min_index])
                    else:
                        matching_ids.append(None)
                        matching_distances.append(None)
                else:
                    matching_ids.append(None)
                    matching_distances.append(None)
        else:
            matching_ids = [None] * len(embs)
            matching_distances = [np.inf] * len(embs)
        
        return matching_ids, matching_distances

class EmotionDetector:
    """Handles emotion detection using pre-trained Keras model"""
    
    def __init__(self, model_json_path, model_weights_path):
        # Load emotion model
        self.emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 
                             3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
        
        try:
            # Load json and create model
            with open(model_json_path, 'r') as json_file:
                loaded_model_json = json_file.read()
            self.emotion_model = model_from_json(loaded_model_json)
            
            # Load weights into new model
            self.emotion_model.load_weights(model_weights_path)
            print("✓ Emotion model loaded successfully")
        except Exception as e:
            print(f"✗ Error loading emotion model: {e}")
            self.emotion_model = None
        
    def detect_emotion(self, face_region):
        """Detect emotion from a face region"""
        if self.emotion_model is None:
            return "Unknown", 0.0
            
        try:
            # Check if face_region is float and convert to uint8 if needed
            if face_region.dtype != np.uint8:
                if face_region.max() <= 1.0:
                    face_region = (face_region * 255).astype(np.uint8)
                else:
                    face_region = face_region.astype(np.uint8)
            
            # Convert to grayscale for emotion detection
            if len(face_region.shape) == 3 and face_region.shape[2] == 3:
                gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            elif len(face_region.shape) == 3:
                gray_face = cv2.cvtColor(face_region, cv2.COLOR_RGB2GRAY)
            else:
                gray_face = face_region
            
            # Resize to 48x48 as required by emotion model
            cropped_img = cv2.resize(gray_face, (48, 48))
            cropped_img = cropped_img.reshape(1, 48, 48, 1)
            cropped_img = cropped_img.astype('float32') / 255.0
            
            # Predict emotion
            emotion_prediction = self.emotion_model.predict(cropped_img)
            maxindex = int(np.argmax(emotion_prediction))
            confidence = float(np.max(emotion_prediction))
            
            return self.emotion_dict[maxindex], confidence
        except Exception as e:
            print(f"Error in emotion detection: {e}")
            return "Unknown", 0.0

class VideoSummarizer:
    """Summarizes video based on specific person and emotion"""
    
    def __init__(self):
        # We'll create sessions when needed to avoid graph conflicts
        self.face_sess = None
        self.mtcnn = None
        self.images_placeholder = None
        self.embeddings = None
        self.phase_train_placeholder = None
        self.id_data = None
        self.emotion_detector = None
        self.face_graph = None
        
    def initialize_face_recognition(self):
        """Initialize face recognition system with separate graph"""
        try:
            # Create a new graph for face recognition
            self.face_graph = tf.Graph()
            with self.face_graph.as_default():
                # Create session for face recognition
                self.face_sess = tf.Session(graph=self.face_graph)
                
                # Initialize MTCNN
                print("Initializing MTCNN...")
                self.mtcnn = detect_and_align.create_mtcnn(self.face_sess, None)
                
                # Load face recognition model
                model_path = '20170512-110547/20170512-110547.pb'
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"Face recognition model not found at {model_path}")
                
                print(f"Loading face recognition model from {model_path}...")
                with tf.gfile.GFile(model_path, 'rb') as f:
                    graph_def = tf.GraphDef()
                    graph_def.ParseFromString(f.read())
                    tf.import_graph_def(graph_def, name='')
                
                # Get tensors
                self.images_placeholder = self.face_graph.get_tensor_by_name("input:0")
                self.embeddings = self.face_graph.get_tensor_by_name("embeddings:0")
                self.phase_train_placeholder = self.face_graph.get_tensor_by_name("phase_train:0")
                
                # Initialize ID database
                id_folder = '20170512-110547/ids'
                self.id_data = IdData(id_folder, self.mtcnn, self.face_sess, 
                                     self.embeddings, self.images_placeholder, 
                                     self.phase_train_placeholder, 1.0)
                
                print(f"✓ Face recognition initialized. Loaded {len(self.id_data.id_names) if self.id_data.id_names else 0} identities")
                return True
                
        except Exception as e:
            print(f"✗ Error initializing face recognition: {e}")
            return False
    
    def initialize_emotion_detection(self):
        """Initialize emotion detection"""
        try:
            emotion_json = 'model/emotion_model.json'
            emotion_weights = 'model/emotion_model.h5'
            
            if not os.path.exists(emotion_json):
                raise FileNotFoundError(f"Emotion model JSON not found at {emotion_json}")
            if not os.path.exists(emotion_weights):
                raise FileNotFoundError(f"Emotion model weights not found at {emotion_weights}")
            
            self.emotion_detector = EmotionDetector(emotion_json, emotion_weights)
            return True
            
        except Exception as e:
            print(f"✗ Error initializing emotion detection: {e}")
            return False
    
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
        if len(face_patch.shape) == 3 and face_patch.shape[2] == 3:
            face_patch = cv2.cvtColor(face_patch, cv2.COLOR_RGB2BGR)
        
        return face_patch
        
    def process_video(self, video_path, target_person, target_emotion, analysis_id, stop_event=None):
        """Process video and extract frames where target person shows target emotion"""
        print(f"\n🔍 Looking for '{target_person}' showing '{target_emotion}' emotion...")
        
        # Initialize systems if not already done
        if not hasattr(self, 'face_sess') or self.face_sess is None:
            if not self.initialize_face_recognition():
                raise Exception("Failed to initialize face recognition")
        
        if not hasattr(self, 'emotion_detector') or self.emotion_detector is None:
            if not self.initialize_emotion_detection():
                print("Warning: Emotion detection not available, using face recognition only")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 30.0  # Default FPS if not detected
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📊 Video Info: {frame_width}x{frame_height} @ {fps:.1f} FPS, {total_frames} frames")
        
        # Create output directories
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"summaries/analysis_{analysis_id}_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Output video - FIXED: Use forward slashes and proper path
        safe_person = "".join(c for c in target_person if c.isalnum() or c in (' ', '_')).rstrip()
        safe_emotion = "".join(c for c in target_emotion if c.isalnum() or c in (' ', '_')).rstrip()
        summary_video = os.path.join(output_dir, f"{safe_person}_{safe_emotion}_summary.mp4")
        
        # Try different codecs if MP4V doesn't work
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(summary_video, fourcc, fps, (frame_width, frame_height))
        
        # Check if video writer is opened
        if not out.isOpened():
            print("⚠️ MP4V codec not available, trying XVID...")
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(summary_video.replace('.mp4', '.avi'), fourcc, fps, (frame_width, frame_height))
            
            if not out.isOpened():
                print("⚠️ XVID codec not available, trying MJPG...")
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                out = cv2.VideoWriter(summary_video.replace('.mp4', '.avi'), fourcc, fps, (frame_width, frame_height))
                
                if not out.isOpened():
                    raise Exception("Failed to create video writer with any codec")
        
        frames_dir = os.path.join(output_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        
        summary_data = []
        frame_count = 0
        matched_frames = 0
        start_time = time.time()
        
        print("\n⏳ Processing video... (Press 'q' to stop early and show summary)")
        print("-" * 50)
        
        # Create a window for real-time display
        window_name = f"Processing: {target_person} - {target_emotion}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 800, 600)
        
        user_interrupted = False
        
        try:
            while True:
                if stop_event and stop_event.is_set():
                    print("\n⚠️ Processing stopped by user request")
                    user_interrupted = True
                    break
                
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                display_frame = frame.copy()  # Copy for display
                output_frame = frame.copy()   # Separate copy for writing to video
                
                # Convert frame to RGB for MTCNN
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Detect faces
                try:
                    face_patches, bounding_boxes, _ = detect_and_align.detect_faces(frame_rgb, self.mtcnn)
                except Exception as e:
                    print(f"Warning: Face detection error on frame {frame_count}: {e}")
                    face_patches = []
                    bounding_boxes = []
                
                match_found = False
                
                if len(face_patches) > 0:
                    face_patches = np.stack(face_patches)
                    
                    # Get face embeddings
                    with self.face_sess.as_default():
                        with self.face_graph.as_default():
                            feed_dict = {
                                self.images_placeholder: face_patches,
                                self.phase_train_placeholder: False
                            }
                            embs = self.face_sess.run(self.embeddings, feed_dict=feed_dict)
                    
                    # Get matching IDs
                    matching_ids, matching_distances = self.id_data.find_matching_ids(embs)
                    
                    for i, (bb, matching_id, dist, face_patch) in enumerate(zip(
                        bounding_boxes, matching_ids, matching_distances, face_patches)):
                        
                        # Convert face patch for emotion detection
                        face_patch_bgr = self.convert_face_patch_to_uint8(face_patch)
                        
                        # Detect emotion
                        emotion = "Unknown"
                        confidence = 0.0
                        
                        if self.emotion_detector:
                            emotion, confidence = self.emotion_detector.detect_emotion(face_patch_bgr)
                        
                        # Check if this matches our target
                        person_match = False
                        if matching_id:
                            # Case-insensitive comparison
                            person_match = matching_id.lower() == target_person.lower()
                        
                        # If we don't have a specific person in database, we'll match based on emotion only
                        # or if person is "Unknown" but we want any person
                        if target_person.lower() == "unknown" or target_person.lower() == "any":
                            person_match = True
                        
                        emotion_match = emotion.lower() == target_emotion.lower()
                        
                        # For demo purposes, if we don't have emotion detector, accept all as matches
                        if self.emotion_detector is None:
                            emotion_match = True
                            emotion = target_emotion
                            confidence = 0.8  # Default confidence
                        
                        if person_match and emotion_match and confidence > 0.5:
                            match_found = True
                            matched_frames += 1
                            
                            timestamp_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                            
                            # Save frame as image
                            frame_filename = f"frame_{frame_count:06d}_{timestamp_sec:.1f}s.jpg"
                            cv2.imwrite(os.path.join(frames_dir, frame_filename), frame)
                            
                            summary_data.append({
                                'frame_number': frame_count,
                                'timestamp': timestamp_sec,
                                'confidence': confidence,
                                'person': matching_id if matching_id else "Unknown",
                                'emotion': emotion,
                                'distance': float(dist) if dist else 0.0,
                                'bounding_box': [int(x) for x in bb] if hasattr(bb, '__iter__') else []
                            })
                            
                            # Draw on output frame (for summary video)
                            cv2.rectangle(output_frame, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0), 3)
                            label = f"{matching_id if matching_id else 'Unknown'} - {emotion} ({confidence:.2f})"
                            cv2.putText(output_frame, label, 
                                      (bb[0], bb[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 
                                      0.7, (0, 255, 0), 2)
                            
                            # Draw on display frame (for real-time display)
                            cv2.rectangle(display_frame, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0), 3)
                            cv2.putText(display_frame, label, 
                                      (bb[0], bb[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 
                                      0.7, (0, 255, 0), 2)
                            cv2.putText(display_frame, "MATCH!", (bb[0], bb[1]-40),
                                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                        else:
                            # Draw detection even if not a match (for visualization only)
                            color = (0, 255, 255)  # Yellow for non-matching detections
                            cv2.rectangle(display_frame, (bb[0], bb[1]), (bb[2], bb[3]), color, 2)
                            label = f"{matching_id if matching_id else 'Unknown'} - {emotion}"
                            cv2.putText(display_frame, label, 
                                      (bb[0], bb[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 
                                      0.5, color, 1)
                
                # If match found, add output frame to summary video
                if match_found:
                    out.write(output_frame)
                
                # Display progress on display frame
                progress_text = f"Frame: {frame_count}/{total_frames} | Matches: {matched_frames}"
                cv2.putText(display_frame, progress_text, (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Display frame in real-time
                cv2.imshow(window_name, display_frame)
                
                # Check for 'q' key press to stop early
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n⚠️ Processing stopped by user (q key pressed)")
                    user_interrupted = True
                    break
                
                # Display progress in console every 50 frames
                if frame_count % 50 == 0:
                    elapsed = time.time() - start_time
                    fps_processed = frame_count / elapsed if elapsed > 0 else 0
                    percent_complete = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                    
                    progress_bar_length = 50
                    progress = int(percent_complete / 2)
                    progress_bar = "█" * progress + "░" * (progress_bar_length - progress)
                    
                    print(f"\rProgress: [{progress_bar}] {percent_complete:.1f}% | "
                          f"Frames: {frame_count}/{total_frames} | "
                          f"Matches: {matched_frames} | "
                          f"Speed: {fps_processed:.1f} FPS", end="", flush=True)
            
            print(f"\n\n✅ Processing {'interrupted' if user_interrupted else 'complete'}! Processed {frame_count} frames.")
            
            # If user interrupted, still save the partial results
            if user_interrupted:
                print(f"⚠️ Processing interrupted. Saving {matched_frames} matches found so far.")
            
            # Release video writer
            out.release()
            
            # Check if any matches were found
            if matched_frames == 0:
                print("❌ No matches found. Summary video will not be created.")
                # Remove empty video file if created
                if os.path.exists(summary_video):
                    os.remove(summary_video)
                summary_video = None
            else:
                print(f"✅ Summary video created with {matched_frames} matching frames: {summary_video}")
                
        except Exception as e:
            print(f"\n✗ Error during processing: {e}")
            # Release resources in case of error
            if 'out' in locals():
                out.release()
            if 'cap' in locals():
                cap.release()
            cv2.destroyAllWindows()
            raise
            
        finally:
            # Always release resources
            cap.release()
            cv2.destroyAllWindows()
        
        # Generate report
        report_file = self.generate_report(summary_data, video_path, target_person, 
                                          target_emotion, summary_video, frames_dir, 
                                          output_dir)
        summary_video = summary_video.replace('\\', '/')  # Use forward slashes
        frames_dir = frames_dir.replace('\\', '/')
        report_file = report_file.replace('\\', '/')
        # Return relative path for web serving
        return summary_video, frames_dir, report_file, matched_frames

    def play_summary_video(self, video_path, target_person, target_emotion):
        """Play the summary video immediately"""
        if not video_path or not os.path.exists(video_path):
            print(f"✗ Summary video not found: {video_path}")
            return
        
        print(f"\n🎬 Playing summary video: {os.path.basename(video_path)}")
        print("Press 'q' to stop playback, 'p' to pause, 'r' to replay")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Cannot open video: {video_path}")
            return
        
        paused = False
        replay = False
        
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    if replay:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        break
            
            # Display frame info
            fps = cap.get(cv2.CAP_PROP_FPS)
            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            info_text = f"{target_person} - {target_emotion} | Frame: {current_frame}/{total_frames}"
            cv2.putText(frame, info_text, (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            status_text = "PAUSED (Press 'p' to resume)" if paused else "PLAYING (Press 'p' to pause)"
            cv2.putText(frame, status_text, (10, 60), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.imshow(f"Summary: {target_person} - {target_emotion}", frame)
            
            key = cv2.waitKey(25) & 0xFF
            if key == ord('q'):  # Quit
                break
            elif key == ord('p'):  # Pause/Resume
                paused = not paused
            elif key == ord('r'):  # Replay
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                replay = True
                paused = False
            elif key == ord(' '):  # Space for single step when paused
                if paused:
                    ret, frame = cap.read()
                    if not ret:
                        break
        
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Playback finished")
    
    def generate_report(self, summary_data, video_path, target_person, target_emotion,
                       summary_video, frames_dir, output_dir):
        """Generate analysis report"""
        report = {
            'analysis_date': datetime.now().isoformat(),
            'target_person': target_person,
            'target_emotion': target_emotion,
            'source_video': os.path.basename(video_path),
            'summary_video': os.path.basename(summary_video) if summary_video else None,
            'summary_video_path': summary_video,
            'frames_directory': frames_dir,
            'total_matches': len(summary_data),
            'matches': summary_data
        }
        
        if summary_data and len(summary_data) > 0:
            confidences = [m['confidence'] for m in summary_data if 'confidence' in m]
            if confidences:
                report['statistics'] = {
                    'average_confidence': float(np.mean(confidences)),
                    'max_confidence': float(max(confidences)),
                    'min_confidence': float(min(confidences))
                }
        
        report_file = os.path.join(output_dir, "report.json")
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report_file
    
    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'face_sess') and self.face_sess:
            self.face_sess.close()
        if hasattr(self, 'mtcnn'):
            del self.mtcnn
        tf.reset_default_graph()