"""
Track_Safe - Integrated Worker Face Recognition & PPE Monitoring System
This system combines worker registration, face recognition, and PPE compliance monitoring
into one comprehensive safety management solution.
"""

import cv2
import face_recognition
import numpy as np
import os
import csv
import pickle
import sqlite3
import time
from datetime import datetime
from ultralytics import YOLO
import cvzone
import math
import winsound
import pandas as pd
import threading
import queue
from PIL import Image

class TrackSafeSystem:
    def __init__(self):
        # Database configuration
        self.database_folder = ""
        self.database_name = ""
        self.db_path = ""
        self.known_encodings = {}
        self.workers_data = {}
        self.available_jobs = ["manager", "engineer", "technician", "supervisor", "operator"]
        self.recognition_threshold = 0.6
        
        # PPE Detection configuration
        self.ppe_model = None
        self.person_model = None
        self.classNames = ['Boot', 'Face-Protector', 'Gloves', 'Helmet', 'Normal-Glasses', 'Safety-Glasses', 'Vest']
        self.REQUIRED_SAFETY_ITEMS = ['Boot', 'Face-Protector', 'Gloves', 'Helmet', 'Safety-Glasses', 'Vest']
        
        # Logging and tracking
        self.log_columns = [
            'Date', 'Time', 'Name', 'ID', 'Position', 'Salary',
            'Helmet', 'Vest', 'Safety-Glasses', 'Gloves', 'Boot',
            'Face-Protector', 'Non_Safety_Glasses', 'Status'
        ]
        self.log_file = "track_safe_log.csv"
        self.last_logged_state = {}
        self.last_beep_time = 0
        self.beep_interval = 5.0  # Reduced beep frequency
        
        # Performance optimization
        self.frame_skip = 2  # Process every 2nd frame
        self.frame_count = 0
        
    def setup_database(self):
        """Setup and initialize database folder and SQLite database"""
        while True:
            database_name = input("Enter database name: ").strip()
            if database_name:
                self.database_name = database_name
                self.database_folder = database_name
                break
            print("Please enter a valid database name.")
        
        # Create database folder
        if not os.path.exists(self.database_folder):
            os.makedirs(self.database_folder)
            print(f"Created database folder: {self.database_folder}")
        
        # Setup SQLite database
        self.db_path = os.path.join(self.database_folder, f"{database_name}.db")
        self.initialize_sqlite_database()
        
        # Load existing data
        self.load_data()
        
        # Initialize log file
        if not os.path.exists(self.log_file):
            pd.DataFrame(columns=self.log_columns).to_csv(self.log_file, index=False)
            print(f"Initialized log file: {self.log_file}")
    
    def initialize_sqlite_database(self):
        """Initialize SQLite database with proper schema"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create workers table with enhanced schema
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS workers (
                    id TEXT PRIMARY KEY,
                    first_name TEXT NOT NULL,
                    middle_name TEXT,
                    last_name TEXT NOT NULL,
                    full_name TEXT NOT NULL,
                    salary REAL NOT NULL,
                    job TEXT NOT NULL,
                    folder_name TEXT NOT NULL,
                    registration_date TEXT NOT NULL,
                    face_encodings BLOB
                )
            ''')
            
            # Create jobs table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS jobs (
                    job_name TEXT PRIMARY KEY
                )
            ''')
            
            # Insert default jobs
            for job in self.available_jobs:
                cursor.execute('INSERT OR IGNORE INTO jobs (job_name) VALUES (?)', (job,))
            
            conn.commit()
            conn.close()
            print("Database initialized successfully")
            
        except Exception as e:
            print(f"Error initializing database: {e}")
    
    def load_data(self):
        """Load workers data and encodings from SQLite database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM workers')
            workers = cursor.fetchall()
            
            for worker in workers:
                worker_id, first_name, middle_name, last_name, full_name, salary, job, folder_name, reg_date, encodings_blob = worker
                
                self.workers_data[worker_id] = {
                    'first_name': first_name,
                    'middle_name': middle_name or '',
                    'last_name': last_name,
                    'full_name': full_name,
                    'salary': salary,
                    'job': job,
                    'folder': folder_name,
                    'registration_date': reg_date
                }
                
                # Load face encodings
                if encodings_blob:
                    encodings = pickle.loads(encodings_blob)
                    self.known_encodings[worker_id] = encodings
            
            # Load available jobs
            cursor.execute('SELECT job_name FROM jobs')
            jobs = cursor.fetchall()
            self.available_jobs = [job[0] for job in jobs]
            
            conn.close()
            print(f"Loaded {len(self.workers_data)} workers from database")
            
        except Exception as e:
            print(f"Error loading data: {e}")
    
    def save_worker_to_database(self, worker_id, worker_data, encodings):
        """Save worker data to SQLite database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Serialize encodings
            encodings_blob = pickle.dumps(encodings)
            
            cursor.execute('''
                INSERT OR REPLACE INTO workers 
                (id, first_name, middle_name, last_name, full_name, salary, job, folder_name, registration_date, face_encodings)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                worker_id,
                worker_data['first_name'],
                worker_data['middle_name'],
                worker_data['last_name'],
                worker_data['full_name'],
                worker_data['salary'],
                worker_data['job'],
                worker_data['folder'],
                worker_data['registration_date'],
                encodings_blob
            ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            print(f"Error saving worker to database: {e}")
            return False
    
    def generate_worker_id(self):
        """Generate unique 3-digit worker ID"""
        if not self.workers_data:
            return "001"
        
        existing_ids = [int(worker_id) for worker_id in self.workers_data.keys()]
        max_id = max(existing_ids)
        new_id = max_id + 1
        return f"{new_id:03d}"
    
    def validate_name_format(self, name_parts):
        """Validate name format: first_middle_last"""
        if len(name_parts) < 2 or len(name_parts) > 3:
            return False
        
        for part in name_parts:
            if not part.isalpha():
                return False
        
        return True
    
    def collect_worker_info(self):
        """Collect worker information with enhanced name format"""
        while True:
            print("\n=== Worker Registration ===")
            
            # Get name (first_middle_last format)
            while True:
                name_input = input("Enter worker's name (format: first_middle_last or first_last): ").strip()
                name_parts = name_input.split('_')
                
                if self.validate_name_format(name_parts):
                    if len(name_parts) == 2:
                        first_name, last_name = name_parts
                        middle_name = ""
                        full_name = f"{first_name}_{last_name}"
                    else:  # len == 3
                        first_name, middle_name, last_name = name_parts
                        full_name = f"{first_name}_{middle_name}_{last_name}"
                    break
                print("Invalid format. Use first_middle_last or first_last (letters only, separated by underscores)")
            
            # Get salary
            while True:
                salary_input = input("Enter worker's salary: ").strip()
                try:
                    salary = float(salary_input)
                    break
                except ValueError:
                    print("Invalid salary. Please enter a valid number.")
            
            # Get job
            job = self.select_job()
            
            # Show summary
            print(f"\n=== Summary ===")
            print(f"Name: {full_name.replace('_', ' ').title()}")
            print(f"Salary: ${salary:.2f}")
            print(f"Job: {job.title()}")
            
            choice = input("\nConfirm this information? (y/n/edit): ").lower().strip()
            
            if choice == 'y':
                return {
                    'first_name': first_name,
                    'middle_name': middle_name,
                    'last_name': last_name,
                    'full_name': full_name,
                    'salary': salary,
                    'job': job
                }
            elif choice == 'n':
                print("Registration cancelled.")
                return None
            elif choice == 'edit':
                continue
            else:
                print("Please enter 'y', 'n', or 'edit'")
    
    def select_job(self):
        """Job selection with database update"""
        while True:
            print(f"\n=== Select Job ===")
            print("Available jobs:")
            for i, job in enumerate(self.available_jobs, 1):
                print(f"{i}. {job.title()}")
            print(f"{len(self.available_jobs) + 1}. Add new job")
            
            try:
                choice = int(input(f"\nSelect job (1-{len(self.available_jobs) + 1}): ").strip())
                
                if 1 <= choice <= len(self.available_jobs):
                    return self.available_jobs[choice - 1]
                
                elif choice == len(self.available_jobs) + 1:
                    new_job = input("Enter new job title: ").strip().lower()
                    if new_job and new_job.isalpha() and new_job not in self.available_jobs:
                        # Add to database
                        try:
                            conn = sqlite3.connect(self.db_path)
                            cursor = conn.cursor()
                            cursor.execute('INSERT INTO jobs (job_name) VALUES (?)', (new_job,))
                            conn.commit()
                            conn.close()
                            
                            self.available_jobs.append(new_job)
                            print(f"✓ Job '{new_job.title()}' added successfully!")
                            return new_job
                        except Exception as e:
                            print(f"Error adding job: {e}")
                    else:
                        print("Invalid job title or job already exists")
                else:
                    print(f"Invalid choice. Please enter 1-{len(self.available_jobs) + 1}")
                    
            except ValueError:
                print("Invalid input. Please enter a number.")
    
    def check_if_worker_exists(self, test_encoding):
        """Check if a worker with similar face already exists in database"""
        for worker_id, encodings_list in self.known_encodings.items():
            for encoding in encodings_list:
                distance = np.linalg.norm(test_encoding - encoding)
                if distance < self.recognition_threshold:
                    return worker_id, self.workers_data[worker_id]
        return None, None
    
    def capture_verification_image(self):
        """Capture a single image for worker verification before registration"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return None, None
        
        # Optimize camera settings
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("\n=== Worker Verification ===")
        print("Please look at the camera for verification...")
        print("Press SPACE to capture verification image or 'q' to cancel")
        
        verification_image = None
        face_encoding = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            
            # Display instructions
            cv2.putText(frame, "Press SPACE to capture or 'q' to cancel", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            if len(face_locations) == 1:
                face_location = face_locations[0]
                top, right, bottom, left = face_location
                
                # Draw rectangle around face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)
                cv2.putText(frame, "Face detected - Press SPACE", 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Extract face encoding for verification
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                if face_encodings:
                    current_encoding = face_encodings[0]
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord(' '):  # Space key
                        verification_image = frame.copy()
                        face_encoding = current_encoding
                        print("✓ Verification image captured!")
                        break
            else:
                if len(face_locations) == 0:
                    cv2.putText(frame, "No face detected", 
                               (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    cv2.putText(frame, "Multiple faces - show only one", 
                               (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow("Worker Verification", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        return verification_image, face_encoding
    
    def capture_face_images(self, worker_folder, worker_data, worker_id):
        """Enhanced face capture with diversity instructions and angle guidance"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return False, []
        
        # Optimize camera settings for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        captured_encodings = []
        captured_angles = []  # Track captured face angles
        image_count = 0
        max_images = 8
        min_images = 5
        
        print(f"\n=== Capturing Images for {worker_data['full_name'].replace('_', ' ').title()} ===")
        print("The system will capture images from different angles automatically.")
        print("Please follow the instructions shown on screen.")
        print("Press 'q' to cancel registration")
        print("\nStarting capture in 3 seconds...")
        time.sleep(3)
        
        last_capture_time = 0
        capture_interval = 1.5  # Capture every 1.5 seconds
        angle_instructions = [
            "Look straight at the camera",
            "Turn your head slightly to the left",
            "Turn your head slightly to the right", 
            "Tilt your head slightly up",
            "Tilt your head slightly down",
            "Turn head left and look up slightly",
            "Turn head right and look up slightly",
            "Look straight again with a neutral expression"
        ]
        
        current_instruction = 0
        instruction_start_time = time.time()
        instruction_duration = 4.0  # Show each instruction for 4 seconds
        
        while image_count < max_images:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)  # Mirror effect
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            
            current_time = time.time()
            
            # Update instruction
            if current_time - instruction_start_time > instruction_duration:
                current_instruction = (current_instruction + 1) % len(angle_instructions)
                instruction_start_time = current_time
            
            # Display current instruction
            instruction_text = angle_instructions[current_instruction]
            cv2.putText(frame, f"Instruction: {instruction_text}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            cv2.putText(frame, f"Images captured: {image_count}/{max_images}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if len(face_locations) == 1:
                face_location = face_locations[0]
                top, right, bottom, left = face_location
                
                # Draw rectangle around face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                
                if current_time - last_capture_time >= capture_interval:
                    # Extract face encoding
                    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                    if face_encodings:
                        face_encoding = face_encodings[0]
                        
                        # Calculate face angle/pose (simplified)
                        face_center_x = (left + right) / 2
                        face_center_y = (top + bottom) / 2
                        face_width = right - left
                        face_height = bottom - top
                        
                        # Create a simple angle signature
                        angle_signature = (
                            round(face_center_x / frame.shape[1], 2),  # Horizontal position
                            round(face_center_y / frame.shape[0], 2),  # Vertical position
                            round(face_width / face_height, 2)         # Aspect ratio
                        )
                        
                        # Check diversity (both encoding and angle)
                        is_diverse_encoding = True
                        is_diverse_angle = True
                        
                        if captured_encodings:
                            min_distance = min([np.linalg.norm(face_encoding - enc) for enc in captured_encodings])
                            if min_distance < 0.25:  # Stricter threshold for diversity
                                is_diverse_encoding = False
                        
                        if captured_angles:
                            min_angle_diff = min([
                                abs(angle_signature[0] - ang[0]) + 
                                abs(angle_signature[1] - ang[1]) + 
                                abs(angle_signature[2] - ang[2])
                                for ang in captured_angles
                            ])
                            if min_angle_diff < 0.15:  # Minimum angle difference
                                is_diverse_angle = False
                        
                        if is_diverse_encoding and is_diverse_angle:
                            # Save image
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                            image_filename = f"{self.database_name}_{worker_data['full_name']}_{worker_id}_{timestamp}.jpg"
                            image_path = os.path.join(worker_folder, image_filename)
                            cv2.imwrite(image_path, frame)
                            
                            captured_encodings.append(face_encoding)
                            captured_angles.append(angle_signature)
                            image_count += 1
                            last_capture_time = current_time
                            
                            print(f"✓ Captured image {image_count}/{max_images} - Good angle diversity!")
                            
                            # Flash effect
                            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 255, 0), 15)
                        else:
                            if not is_diverse_encoding:
                                cv2.putText(frame, "Similar expression detected - follow instruction", 
                                           (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                            if not is_diverse_angle:
                                cv2.putText(frame, "This angle already captured - try different pose", 
                                           (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                cv2.putText(frame, "Face detected - Follow the instruction above", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                if len(face_locations) == 0:
                    cv2.putText(frame, "No face detected - position yourself in frame", 
                               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                else:
                    cv2.putText(frame, "Multiple faces detected - show only one face", 
                               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            cv2.imshow("Track_Safe - Face Registration", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        if len(captured_encodings) >= min_images:
            # Calculate final diversity score
            diversity_score = self.calculate_encoding_diversity(captured_encodings)
            print(f"✓ Registration successful with {len(captured_encodings)} diverse images")
            print(f"  Diversity score: {diversity_score:.3f}")
            return True, captured_encodings
        else:
            print(f"✗ Registration failed: Only {len(captured_encodings)} images captured (minimum {min_images} required)")
            return False, []
    
    def calculate_encoding_diversity(self, encodings):
        """Calculate diversity score of face encodings"""
        if len(encodings) < 2:
            return 0
        
        total_distance = 0
        comparisons = 0
        
        for i in range(len(encodings)):
            for j in range(i + 1, len(encodings)):
                distance = np.linalg.norm(encodings[i] - encodings[j])
                total_distance += distance
                comparisons += 1
        
        return total_distance / comparisons if comparisons > 0 else 0
    
    def register_new_worker(self):
        """Complete worker registration with pre-verification to prevent duplicates"""
        print("\n=== Track_Safe Worker Registration ===")
        print("Step 1: Worker Verification")
        print("Before registration, we need to verify if this worker already exists.")
        
        # Step 1: Capture verification image
        verification_image, verification_encoding = self.capture_verification_image()
        
        if verification_image is None or verification_encoding is None:
            print("Verification cancelled.")
            return
        
        # Step 2: Check if worker already exists
        print("\n🔍 Checking database for existing worker...")
        existing_worker_id, existing_worker_info = self.check_if_worker_exists(verification_encoding)
        
        if existing_worker_id:
            print(f"\n⚠️  WORKER ALREADY EXISTS!")
            print(f"Name: {existing_worker_info['full_name'].replace('_', ' ').title()}")
            print(f"ID: {existing_worker_id}")
            print(f"Job: {existing_worker_info['job'].title()}")
            print(f"Registration Date: {existing_worker_info['registration_date']}")
            print("Registration cancelled - worker is already in the database.")
            
            # Show verification image with existing worker info
            display_image = verification_image.copy()
            cv2.putText(display_image, "WORKER ALREADY EXISTS", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            cv2.putText(display_image, f"Name: {existing_worker_info['full_name'].replace('_', ' ').title()}", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(display_image, f"ID: {existing_worker_id}", 
                       (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(display_image, "Press any key to continue", 
                       (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow("Worker Already Exists", display_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return
        
        print("✅ Worker not found in database. Proceeding with registration...")
        
        # Step 3: Show verification image to user
        display_image = verification_image.copy()
        cv2.putText(display_image, "NEW WORKER DETECTED", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        cv2.putText(display_image, "Proceed with registration? (y/n)", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow("New Worker Verification", display_image)
        
        print("\nVerification image captured.")
        proceed = input("Do you want to proceed with registration? (y/n): ").lower().strip()
        cv2.destroyAllWindows()
        
        if proceed != 'y':
            print("Registration cancelled by user.")
            return
        
        # Step 4: Collect worker information
        worker_info = self.collect_worker_info()
        if not worker_info:
            return
        
        # Step 5: Generate ID and create folder
        worker_id = self.generate_worker_id()
        folder_name = f"{self.database_name}_{worker_info['full_name']}_{worker_id}"
        worker_folder = os.path.join(self.database_folder, folder_name)
        
        try:
            os.makedirs(worker_folder, exist_ok=True)
            
            # Step 6: Capture multiple face images with guidance
            print(f"\n=== Step 2: Face Image Capture ===")
            print("Now we'll capture multiple images from different angles for better recognition.")
            input("Press Enter when ready to start face capture...")
            
            success, encodings = self.capture_face_images(worker_folder, worker_info, worker_id)
            
            if not success:
                import shutil
                shutil.rmtree(worker_folder, ignore_errors=True)
                print("❌ Registration failed - insufficient quality images")
                return
            
            # Step 7: Save to database
            worker_info['folder'] = folder_name
            worker_info['registration_date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            if self.save_worker_to_database(worker_id, worker_info, encodings):
                # Update local data
                self.workers_data[worker_id] = worker_info
                self.known_encodings[worker_id] = encodings
                
                print(f"\n🎉 === REGISTRATION SUCCESSFUL ===")
                print(f"Worker: {worker_info['full_name'].replace('_', ' ').title()}")
                print(f"ID: {worker_id}")
                print(f"Job: {worker_info['job'].title()}")
                print(f"Salary: ${worker_info['salary']:.2f}")
                print(f"Images captured: {len(encodings)}")
                print(f"Database: {self.database_name}")
                print("✅ Worker successfully registered in Track_Safe database!")
            else:
                import shutil
                shutil.rmtree(worker_folder, ignore_errors=True)
                print("❌ Registration failed - database error")
                
        except Exception as e:
            print(f"❌ Registration error: {e}")
            import shutil
            shutil.rmtree(worker_folder, ignore_errors=True)
    
    def initialize_ppe_models(self):
        """Initialize PPE detection models"""
        try:
            # Load YOLO models (adjust paths as needed)
            self.ppe_model = YOLO("best (1).pt")  # Your PPE model
            self.person_model = YOLO("yolov8n.pt")  # Person detection model
            print("✓ PPE detection models loaded successfully")
            return True
        except Exception as e:
            print(f"✗ Error loading PPE models: {e}")
            return False
    
    def recognize_face(self, face_encoding):
        """Enhanced face recognition with confidence scoring"""
        best_match_id = None
        best_distance = float('inf')
        
        for worker_id, encodings_list in self.known_encodings.items():
            for encoding in encodings_list:
                distance = np.linalg.norm(face_encoding - encoding)
                if distance < best_distance:
                    best_distance = distance
                    best_match_id = worker_id
        
        if best_distance < self.recognition_threshold:
            confidence = 1 - best_distance
            return best_match_id, confidence
        
        return None, 0.0
    
    def should_log_worker(self, worker_id, equipment_status, face_protector_status, non_safety_detected):
        """Determine if worker should be logged based on state changes"""
        # Create state signature
        current_state = {
            'equipment': {item: equipment_status[item]['worn'] for item in self.REQUIRED_SAFETY_ITEMS},
            'face_protector': face_protector_status['worn'],
            'non_safety': non_safety_detected
        }
        
        # Check if state changed
        prev_state = self.last_logged_state.get(worker_id)
        
        if prev_state is None or prev_state != current_state:
            self.last_logged_state[worker_id] = current_state
            return True
        
        return False
    
    def run_integrated_monitoring(self):
        """Main integrated monitoring system"""
        if not self.initialize_ppe_models():
            print("Cannot start monitoring - PPE models not loaded")
            return
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        # Optimize camera settings
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        print(f"\n=== Track_Safe Monitoring Started ===")
        print(f"Database: {self.database_name}")
        print(f"Workers in database: {len(self.workers_data)}")
        print("- Real-time PPE compliance monitoring")
        print("- Automatic worker identification")
        print("- Smart logging system")
        print("- Press 'q' to quit")
        
        unknown_detection_count = {}
        unknown_threshold = 5  # Detect unknown worker 5 times before registration
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue
                
                # Frame skipping for performance
                self.frame_count += 1
                if self.frame_count % self.frame_skip != 0:
                    cv2.imshow("Track_Safe Monitoring", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue
                
                frame = cv2.flip(frame, 1)
                
                # Equipment status tracking
                equipment_status = {item: {'worn': False, 'held': False} for item in self.REQUIRED_SAFETY_ITEMS}
                face_protector_status = {'worn': False, 'held': False}
                non_safety_detected = False
                
                # Detect people
                person_boxes = []
                try:
                    person_results = self.person_model(frame, stream=True, verbose=False)
                    for r in person_results:
                        for box in r.boxes:
                            if int(box.cls[0]) == 0:  # Person class
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                person_boxes.append((x1, y1, x2, y2))
                except:
                    pass
                
                # Detect PPE items
                try:
                    ppe_results = self.ppe_model(frame, stream=True, verbose=False)
                    for r in ppe_results:
                        for box in r.boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            conf = math.ceil((box.conf[0] * 100)) / 100
                            cls = int(box.cls[0])
                            currentClass = self.classNames[cls]
                            
                            if conf > 0.5:
                                is_worn = False
                                
                                # Check if item is worn by person
                                for px1, py1, px2, py2 in person_boxes:
                                    overlap = max(0, min(x2, px2) - max(x1, px1)) * max(0, min(y2, py2) - max(y1, py1))
                                    item_area = (x2-x1) * (y2-y1)
                                    if overlap > item_area * 0.3:  # 30% overlap
                                        is_worn = True
                                        break
                                
                                # Update equipment status
                                if currentClass in equipment_status:
                                    if is_worn:
                                        equipment_status[currentClass]['worn'] = True
                                    else:
                                        equipment_status[currentClass]['held'] = True
                                elif currentClass == 'Face-Protector':
                                    if is_worn:
                                        face_protector_status['worn'] = True
                                    else:
                                        face_protector_status['held'] = True
                                elif currentClass == 'Normal-Glasses':
                                    non_safety_detected = True
                                
                                # Display detection
                                color = (0, 255, 0) if is_worn else (255, 255, 0) if currentClass in self.REQUIRED_SAFETY_ITEMS else (255, 0, 0)
                                cvzone.putTextRect(frame, f'{currentClass} {conf}',
                                                (max(0, x1), max(35, y1)),
                                                colorB=color, colorT=(255, 255, 255),
                                                colorR=color, offset=5)
                                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                except:
                    pass
                
                # Process each detected person
                current_time = time.time()
                for i, (x1, y1, x2, y2) in enumerate(person_boxes):
                    # Face recognition
                    try:
                        face_roi = frame[y1:y2, x1:x2]
                        rgb_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                        face_encodings = face_recognition.face_encodings(rgb_face)
                        
                        worker_id = None
                        worker_info = None
                        
                        if face_encodings:
                            worker_id, confidence = self.recognize_face(face_encodings[0])
                            if worker_id:
                                worker_info = self.workers_data[worker_id]
                                
                                # Display worker info
                                info_text = f"{worker_info['full_name'].replace('_', ' ').title()} ({worker_id})"
                                cvzone.putTextRect(frame, info_text,
                                                (x1, y1 - 30), scale=0.8,
                                                colorB=(0, 200, 0), colorT=(255, 255, 255),
                                                colorR=(0, 200, 0), offset=5)
                            else:
                                # Handle unknown worker
                                person_key = f"{x1}_{y1}_{x2}_{y2}"
                                unknown_detection_count[person_key] = unknown_detection_count.get(person_key, 0) + 1
                                
                                if unknown_detection_count[person_key] >= unknown_threshold:
                                    cvzone.putTextRect(frame, "Unknown Worker - Register?",
                                                    (x1, y1 - 30), scale=0.8,
                                                    colorB=(0, 0, 255), colorT=(255, 255, 255),
                                                    colorR=(0, 0, 255), offset=5)
                                else:
                                    cvzone.putTextRect(frame, f"Identifying... {unknown_detection_count[person_key]}/{unknown_threshold}",
                                                    (x1, y1 - 30), scale=0.8,
                                                    colorB=(255, 165, 0), colorT=(255, 255, 255),
                                                    colorR=(255, 165, 0), offset=5)
                    except:
                        pass
                    
                    # Check compliance
                    all_worn = all(equipment_status[item]['worn'] for item in self.REQUIRED_SAFETY_ITEMS)
                    any_worn = any(equipment_status[item]['worn'] for item in self.REQUIRED_SAFETY_ITEMS)
                    
                    # Status display
                    if all_worn:
                        status_msg = '✓ COMPLIANT: All safety equipment detected'
                        status_color = (0, 255, 0)
                    else:
                        missing_items = [item for item in self.REQUIRED_SAFETY_ITEMS if not equipment_status[item]['worn']]
                        status_msg = f"⚠ MISSING: {', '.join(missing_items)}"
                        status_color = (0, 0, 255)
                    
                    cvzone.putTextRect(frame, status_msg,
                                    (x1, y2 + 10), scale=0.8,
                                    colorB=status_color, colorT=(255, 255, 255),
                                    colorR=status_color, offset=5)
                    
                    # Smart logging
                    if worker_info and self.should_log_worker(worker_id, equipment_status, face_protector_status, non_safety_detected):
                        self.log_worker_status(worker_info, worker_id, equipment_status, face_protector_status, non_safety_detected, all_worn, any_worn)
                    
                    # Alert system - only beep for non-compliant workers
                    if not all_worn and (current_time - self.last_beep_time) > self.beep_interval:
                        try:
                            winsound.Beep(2000, 300)  # Warning beep for non-compliance
                            self.last_beep_time = current_time
                        except:
                            pass  # Ignore beep errors on systems without sound
                
                # Display system info
                cv2.putText(frame, f"Track_Safe - Workers: {len(self.workers_data)}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.imshow("Track_Safe Monitoring", frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        except KeyboardInterrupt:
            print("\nShutting down Track_Safe system...")
        finally:
            cap.release()
            cv2.destroyAllWindows()
    
    def log_worker_status(self, worker_info, worker_id, equipment_status, face_protector_status, non_safety_detected, all_worn, any_worn):
        """Log worker status to CSV file"""
        try:
            now = datetime.now()
            
            log_entry = {
                'Date': now.strftime("%Y-%m-%d"),
                'Time': now.strftime("%H:%M:%S"),
                'Name': worker_info['full_name'].replace('_', ' ').title(),
                'ID': worker_id,
                'Position': worker_info['job'].title(),
                'Salary': f"${worker_info['salary']:.2f}",
                'Helmet': 'Yes' if equipment_status['Helmet']['worn'] else ('Held' if equipment_status['Helmet']['held'] else 'No'),
                'Vest': 'Yes' if equipment_status['Vest']['worn'] else ('Held' if equipment_status['Vest']['held'] else 'No'),
                'Safety-Glasses': 'Yes' if equipment_status['Safety-Glasses']['worn'] else ('Held' if equipment_status['Safety-Glasses']['held'] else 'No'),
                'Gloves': 'Yes' if equipment_status['Gloves']['worn'] else ('Held' if equipment_status['Gloves']['held'] else 'No'),
                'Boot': 'Yes' if equipment_status['Boot']['worn'] else ('Held' if equipment_status['Boot']['held'] else 'No'),
                'Face-Protector': 'Yes' if face_protector_status['worn'] else ('Held' if face_protector_status['held'] else 'No'),
                'Non_Safety_Glasses': 'Yes' if non_safety_detected else 'No',
                'Status': 'Compliant' if all_worn else 'Partial' if any_worn else 'Non-Compliant'
            }
            
            # Create ordered entry
            ordered_entry = {col: log_entry.get(col, 'Unknown') for col in self.log_columns}
            
            # Save to CSV
            pd.DataFrame([ordered_entry])[self.log_columns].to_csv(
                self.log_file, mode='a', header=False, index=False
            )
            
            print(f"Logged: {worker_info['full_name'].replace('_', ' ').title()} - {ordered_entry['Status']}")
            
        except Exception as e:
            print(f"Error logging worker status: {e}")
    
    def list_all_workers(self):
        """List all registered workers with complete information"""
        if not self.workers_data:
            print(f"\nNo workers registered in {self.database_name} database yet.")
            return
        
        print(f"\n=== {self.database_name.upper()} DATABASE - ALL WORKERS ({len(self.workers_data)}) ===")
        print("=" * 100)
        
        for worker_id, data in self.workers_data.items():
            print(f"Worker ID: {worker_id}")
            print(f"Full Name: {data['full_name'].replace('_', ' ').title()}")
            print(f"First Name: {data['first_name'].title()}")
            if data['middle_name']:
                print(f"Middle Name: {data['middle_name'].title()}")
            print(f"Last Name: {data['last_name'].title()}")
            print(f"Job Position: {data['job'].title()}")
            print(f"Salary: ${data['salary']:.2f}")
            print(f"Registration Date: {data['registration_date']}")
            print(f"Face Images: {len(self.known_encodings.get(worker_id, []))}")
            print(f"Data Folder: {data['folder']}")
            print("-" * 100)
        
        print(f"Total Workers in Database: {len(self.workers_data)}")
        print(f"Database Location: {os.path.abspath(self.db_path)}")
    
    def show_main_menu(self):
        """Display main menu options"""
        while True:
            print("\n" + "="*80)
            print("TRACK_SAFE - INTEGRATED WORKER RECOGNITION & PPE MONITORING SYSTEM")
            print("="*80)
            print(f"Database: {self.database_name}")
            print(f"Registered Workers: {len(self.workers_data)}")
            print(f"Available Jobs: {', '.join([job.title() for job in self.available_jobs])}")
            print("\nMenu Options:")
            print("1. Start TrackSafe System")
            print("2. Register New Worker")
            print("3. List All Workers Database")
            print("4. Reload Database (Refresh worker data)")
            print("5. Exit Track_Safe System")
            
            choice = input("\nEnter your choice (1-5): ").strip()
            
            if choice == '1':
                print(f"\n🚀 Starting Track_Safe System...")
                print("📊 PPE compliance monitoring: ACTIVE")
                print("👤 Worker identification: ACTIVE") 
                print("🎯 Person tracking: ACTIVE")
                print("📝 Smart logging system: ACTIVE")
                print("🔊 Audio alerts: ACTIVE")
                print("\n⚠️  IMPORTANT:")
                print("- System will beep ONLY when workers are non-compliant")
                print("- Workers are tracked continuously with fake IDs until face recognition")
                print("- Logging occurs only when worker status changes")
                print("- Press 'q' during monitoring to stop the system")
                
                input("\n✅ Press Enter to start monitoring or Ctrl+C to cancel...")
                
                try:
                    self.run_integrated_monitoring()
                except KeyboardInterrupt:
                    print("\n🛑 Track_Safe monitoring stopped by user")
                except Exception as e:
                    print(f"\n❌ Error during monitoring: {e}")
                
            elif choice == '2':
                self.register_new_worker()
            elif choice == '3':
                self.list_all_workers()
            elif choice == '4':
                print("🔄 Reloading database...")
                self.load_data()
                print(f"✅ Database reloaded - {len(self.workers_data)} workers loaded")
            elif choice == '5':
                print("🔚 Track_Safe system shutdown complete!")
                break
            else:
                print("❌ Invalid choice. Please enter 1-5.")


class PersonTracker:
    """Enhanced person tracking system with fake ID assignment and face recognition integration"""
    
    def __init__(self, track_safe_system):
        self.track_safe = track_safe_system
        self.tracked_persons = {}  # person_id -> person_data
        self.next_fake_id = 1
        self.max_tracking_distance = 100  # Maximum distance to consider same person
        self.max_frames_without_detection = 30  # Frames before considering person lost
        self.iou_threshold = 0.3  # IoU threshold for box matching
        
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union (IoU) between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection area
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union area
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def calculate_distance(self, box1, box2):
        """Calculate center distance between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        center1 = ((x1_1 + x2_1) / 2, (y1_1 + y2_1) / 2)
        center2 = ((x1_2 + x2_2) / 2, (y1_2 + y2_2) / 2)
        
        return math.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    def get_fake_id(self):
        """Generate a new fake ID"""
        fake_id = f"TEMP_{self.next_fake_id:03d}"
        self.next_fake_id += 1
        return fake_id
    
    def update_tracks(self, current_detections, frame):
        """Update person tracking with current detections"""
        current_time = time.time()
        
        # Mark all existing tracks as not updated
        for person_id in self.tracked_persons:
            self.tracked_persons[person_id]['updated_this_frame'] = False
            self.tracked_persons[person_id]['frames_without_detection'] += 1
        
        # Process current detections
        unmatched_detections = []
        
        for detection_box in current_detections:
            best_match_id = None
            best_score = 0
            
            # Try to match with existing tracks
            for person_id, person_data in self.tracked_persons.items():
                if person_data['frames_without_detection'] > self.max_frames_without_detection:
                    continue
                
                # Calculate IoU and distance
                iou = self.calculate_iou(detection_box, person_data['last_box'])
                distance = self.calculate_distance(detection_box, person_data['last_box'])
                
                # Scoring based on IoU and distance
                if iou > self.iou_threshold and distance < self.max_tracking_distance:
                    score = iou * (1 - distance / self.max_tracking_distance)
                    if score > best_score:
                        best_score = score
                        best_match_id = person_id
            
            if best_match_id:
                # Update existing track
                self.tracked_persons[best_match_id].update({
                    'last_box': detection_box,
                    'updated_this_frame': True,
                    'frames_without_detection': 0,
                    'last_seen': current_time,
                    'track_history': self.tracked_persons[best_match_id]['track_history'][-20:] + [detection_box]  # Keep last 20 positions
                })
                
                # Try face recognition for this track
                self.attempt_face_recognition(best_match_id, detection_box, frame, current_time)
            else:
                # New detection - add to unmatched
                unmatched_detections.append(detection_box)
        
        # Create new tracks for unmatched detections
        for detection_box in unmatched_detections:
            fake_id = self.get_fake_id()
            self.tracked_persons[fake_id] = {
                'real_worker_id': None,
                'worker_info': None,
                'confidence': 0.0,
                'is_fake_id': True,
                'last_box': detection_box,
                'first_detection': current_time,
                'last_seen': current_time,
                'updated_this_frame': True,
                'frames_without_detection': 0,
                'track_history': [detection_box],
                'face_recognition_attempts': 0,
                'last_face_attempt': 0
            }
            
            # Try immediate face recognition
            self.attempt_face_recognition(fake_id, detection_box, frame, current_time)
        
        # Remove lost tracks
        lost_tracks = [pid for pid, pdata in self.tracked_persons.items() 
                      if pdata['frames_without_detection'] > self.max_frames_without_detection]
        
        for lost_id in lost_tracks:
            if not self.tracked_persons[lost_id]['is_fake_id']:
                print(f"Lost track of worker: {self.tracked_persons[lost_id]['worker_info']['full_name'] if self.tracked_persons[lost_id]['worker_info'] else lost_id}")
            del self.tracked_persons[lost_id]
    
    def attempt_face_recognition(self, person_id, detection_box, frame, current_time):
        """Attempt face recognition for a tracked person"""
        person_data = self.tracked_persons[person_id]
        
        # Limit face recognition attempts to avoid performance issues
        if (current_time - person_data['last_face_attempt']) < 1.0:  # Try every 1 second
            return
        
        person_data['last_face_attempt'] = current_time
        person_data['face_recognition_attempts'] += 1
        
        try:
            x1, y1, x2, y2 = detection_box
            
            # Expand face region slightly for better recognition
            height = y2 - y1
            width = x2 - x1
            face_expansion = 0.1  # Expand by 10%
            
            fx1 = max(0, int(x1 - width * face_expansion))
            fy1 = max(0, int(y1 - height * face_expansion))
            fx2 = min(frame.shape[1], int(x2 + width * face_expansion))
            fy2 = min(frame.shape[0], int(y2 + height * face_expansion))
            
            face_roi = frame[fy1:fy2, fx1:fx2]
            
            if face_roi.size > 0:
                rgb_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                face_encodings = face_recognition.face_encodings(rgb_face)
                
                if face_encodings:
                    worker_id, confidence = self.track_safe.recognize_face(face_encodings[0])
                    
                    if worker_id and confidence > 0.5:  # Good confidence threshold
                        # Face recognized successfully
                        worker_info = self.track_safe.workers_data[worker_id]
                        
                        # If this was a fake ID, replace it with real worker data
                        if person_data['is_fake_id']:
                            print(f"✓ Face recognized: {worker_info['full_name'].replace('_', ' ').title()} (was {person_id})")
                            
                            # Update person data with real information
                            person_data.update({
                                'real_worker_id': worker_id,
                                'worker_info': worker_info,
                                'confidence': confidence,
                                'is_fake_id': False,
                                'face_confirmed': True
                            })
                        else:
                            # Update confidence for existing recognition
                            person_data['confidence'] = max(person_data['confidence'], confidence)
                    
                    elif person_data['face_recognition_attempts'] > 10 and person_data['is_fake_id']:
                        # Too many failed attempts - might be unknown worker
                        person_data['possibly_unknown'] = True
        
        except Exception as e:
            # Silently handle face recognition errors
            pass
    
    def get_display_info(self, person_id):
        """Get display information for a tracked person"""
        person_data = self.tracked_persons[person_id]
        
        if person_data.get('face_confirmed') and person_data['worker_info']:
            # Real worker with confirmed face
            worker_info = person_data['worker_info']
            display_name = worker_info['full_name'].replace('_', ' ').title()
            display_id = person_data['real_worker_id']
            color = (0, 255, 0)  # Green for confirmed
            status = "CONFIRMED"
        elif person_data['is_fake_id']:
            if person_data.get('possibly_unknown'):
                display_name = "Unknown Worker"
                display_id = person_id
                color = (0, 0, 255)  # Red for unknown
                status = "UNKNOWN"
            else:
                display_name = "Identifying..."
                display_id = person_id
                color = (255, 165, 0)  # Orange for tracking
                status = f"TRACKING ({person_data['face_recognition_attempts']} attempts)"
        else:
            # Recognized but not fully confirmed
            worker_info = person_data['worker_info']
            display_name = worker_info['full_name'].replace('_', ' ').title()
            display_id = person_data['real_worker_id']
            color = (255, 255, 0)  # Yellow for partial
            status = f"PARTIAL ({person_data['confidence']:.2f})"
        
        return {
            'name': display_name,
            'id': display_id,
            'color': color,
            'status': status,
            'worker_info': person_data.get('worker_info'),
            'is_confirmed': person_data.get('face_confirmed', False)
        }


# Enhanced TrackSafeSystem with integrated tracking
class EnhancedTrackSafeSystem(TrackSafeSystem):
    """Enhanced Track_Safe system with advanced person tracking"""
    
    def __init__(self):
        super().__init__()
        self.person_tracker = PersonTracker(self)
        
    def run_integrated_monitoring(self):
        """Enhanced monitoring with person tracking - Fixed performance and stability issues"""
        print("\n🔄 Initializing Track_Safe monitoring system...")
        
        # Initialize PPE models with error handling
        if not self.initialize_ppe_models():
            print("❌ Cannot start monitoring - PPE models not loaded")
            print("Please ensure YOLO model files are available:")
            print("- best (1).pt (PPE detection model)")
            print("- yolov8n.pt (Person detection model)")
            input("Press Enter to return to main menu...")
            return False
        
        print("✅ PPE models loaded successfully")
        
        # Initialize camera with multiple attempts
        cap = None
        for attempt in range(3):
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                break
            print(f"⚠️  Camera initialization attempt {attempt + 1} failed")
            if cap:
                cap.release()
            time.sleep(1)
        
        if not cap or not cap.isOpened():
            print("❌ Error: Could not open camera after 3 attempts")
            print("Please check:")
            print("- Camera is connected and not used by another application")
            print("- Camera drivers are installed")
            input("Press Enter to return to main menu...")
            return False
        
        print("✅ Camera initialized successfully")
        
        # Aggressive camera optimization for reduced delay
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimal buffer to reduce delay
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))  # Use MJPG codec
        
        # Additional optimizations
        self.frame_skip = 1  # Process every frame for better responsiveness
        
        print(f"\n🚀 === Track_Safe Enhanced Monitoring Started ===")
        print(f"📂 Database: {self.database_name}")
        print(f"👥 Workers in database: {len(self.workers_data)}")
        print("🎯 Features Active:")
        print("   - Advanced person tracking with fake IDs")
        print("   - Real-time PPE compliance monitoring")
        print("   - Smart face recognition integration")
        print("   - Intelligent logging system")
        print("   - Audio alerts for non-compliance")
        print("\n⌨️  Controls:")
        print("   - Press 'q' to quit monitoring")
        print("   - Press 'r' to reload database")
        print("   - Press 's' to save current frame")
        print("\n🟢 System is now monitoring... (Camera delay optimized)")
        
        # Performance tracking
        fps_counter = 0
        fps_start_time = time.time()
        
        try:
            monitoring_active = True
            while monitoring_active:
                # Read frame with timeout handling
                ret, frame = cap.read()
                if not ret:
                    print("⚠️  Warning: Could not read frame, retrying...")
                    continue
                
                # FPS optimization - process every frame but with smart detection intervals
                self.frame_count += 1
                fps_counter += 1
                
                # Calculate and display FPS every 30 frames
                if fps_counter % 30 == 0:
                    current_time = time.time()
                    elapsed = current_time - fps_start_time
                    if elapsed > 0:
                        fps = 30 / elapsed
                        if fps < 15:  # If FPS is too low, increase frame skipping
                            self.frame_skip = 2
                        else:
                            self.frame_skip = 1
                    fps_start_time = current_time
                
                # Frame skipping for performance (but reduced)
                if self.frame_count % self.frame_skip != 0:
                    cv2.imshow("Track_Safe Enhanced Monitoring", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        monitoring_active = False
                        break
                    elif key == ord('r'):
                        print("🔄 Reloading database...")
                        self.load_data()
                        print(f"✅ Database reloaded - {len(self.workers_data)} workers")
                    continue
                
                frame = cv2.flip(frame, 1)
                
                # Detect people first
                person_boxes = []
                try:
                    person_results = self.person_model(frame, stream=True, verbose=False)
                    for r in person_results:
                        if hasattr(r, 'boxes') and r.boxes is not None:
                            for box in r.boxes:
                                if int(box.cls[0]) == 0:  # Person class
                                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                                    person_boxes.append((x1, y1, x2, y2))
                except Exception as e:
                    # Silently handle detection errors to prevent crashes
                    pass
                
                # Update person tracking
                self.person_tracker.update_tracks(person_boxes, frame)
                
                # Equipment status tracking
                equipment_status = {item: {'worn': False, 'held': False} for item in self.REQUIRED_SAFETY_ITEMS}
                face_protector_status = {'worn': False, 'held': False}
                non_safety_detected = False
                
                # Detect PPE items with error handling
                try:
                    ppe_results = self.ppe_model(frame, stream=True, verbose=False)
                    for r in ppe_results:
                        if hasattr(r, 'boxes') and r.boxes is not None:
                            for box in r.boxes:
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                conf = math.ceil((box.conf[0] * 100)) / 100
                                cls = int(box.cls[0])
                                
                                if cls < len(self.classNames):  # Safety check
                                    currentClass = self.classNames[cls]
                                    
                                    if conf > 0.5:
                                        is_worn = False
                                        
                                        # Check if item is worn by any tracked person
                                        for person_id, person_data in self.person_tracker.tracked_persons.items():
                                            if not person_data.get('updated_this_frame', False):
                                                continue
                                            
                                            px1, py1, px2, py2 = person_data['last_box']
                                            overlap = max(0, min(x2, px2) - max(x1, px1)) * max(0, min(y2, py2) - max(y1, py1))
                                            item_area = (x2-x1) * (y2-y1)
                                            
                                            if overlap > item_area * 0.3:  # 30% overlap
                                                is_worn = True
                                                break
                                        
                                        # Update equipment status
                                        if currentClass in equipment_status:
                                            if is_worn:
                                                equipment_status[currentClass]['worn'] = True
                                            else:
                                                equipment_status[currentClass]['held'] = True
                                        elif currentClass == 'Face-Protector':
                                            if is_worn:
                                                face_protector_status['worn'] = True
                                            else:
                                                face_protector_status['held'] = True
                                        elif currentClass == 'Normal-Glasses':
                                            non_safety_detected = True
                                        
                                        # Display detection
                                        color = (0, 255, 0) if is_worn else (255, 255, 0) if currentClass in self.REQUIRED_SAFETY_ITEMS else (255, 0, 0)
                                        
                                        # Use simpler text rendering for performance
                                        cv2.putText(frame, f'{currentClass} {conf}', 
                                                  (max(0, x1), max(35, y1)), 
                                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                except Exception as e:
                    # Silently handle PPE detection errors
                    pass
                
                # Draw tracked persons with their information
                current_time = time.time()
                non_compliant_detected = False
                
                for person_id, person_data in list(self.person_tracker.tracked_persons.items()):
                    if not person_data.get('updated_this_frame', False):
                        continue
                    
                    try:
                        x1, y1, x2, y2 = person_data['last_box']
                        display_info = self.person_tracker.get_display_info(person_id)
                        
                        # Draw bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), display_info['color'], 2)
                        
                        # Display person information (simplified for performance)
                        info_text = f"{display_info['name']} ({display_info['id']})"
                        cv2.putText(frame, info_text, (x1, y1 - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, display_info['color'], 2)
                        
                        # Check compliance for this person
                        all_worn = all(equipment_status[item]['worn'] for item in self.REQUIRED_SAFETY_ITEMS)
                        any_worn = any(equipment_status[item]['worn'] for item in self.REQUIRED_SAFETY_ITEMS)
                        
                        if not all_worn:
                            non_compliant_detected = True
                        
                        # Status display (simplified)
                        if all_worn:
                            status_msg = 'COMPLIANT'
                            status_color = (0, 255, 0)
                        else:
                            missing_count = sum(1 for item in self.REQUIRED_SAFETY_ITEMS if not equipment_status[item]['worn'])
                            status_msg = f"MISSING {missing_count} items"
                            status_color = (0, 0, 255)
                        
                        cv2.putText(frame, status_msg, (x1, y2 + 20), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
                        
                        # Smart logging for confirmed workers
                        if display_info['is_confirmed'] and display_info['worker_info']:
                            worker_id = display_info['id']
                            if self.should_log_worker(worker_id, equipment_status, face_protector_status, non_safety_detected):
                                self.log_worker_status(display_info['worker_info'], worker_id, equipment_status, face_protector_status, non_safety_detected, all_worn, any_worn)
                    
                    except Exception as e:
                        # Handle any errors in person processing
                        continue
                
                # Alert system - only beep for non-compliant workers (reduced frequency)
                if non_compliant_detected and (current_time - self.last_beep_time) > self.beep_interval:
                    try:
                        # Use a separate thread for beeping to avoid blocking
                        def beep_async():
                            try:
                                winsound.Beep(2000, 200)  # Shorter beep
                            except:
                                pass
                        
                        import threading
                        beep_thread = threading.Thread(target=beep_async)
                        beep_thread.daemon = True
                        beep_thread.start()
                        
                        self.last_beep_time = current_time
                    except:
                        pass  # Ignore beep errors
                
                # Display system info (minimal)
                tracked_count = len([p for p in self.person_tracker.tracked_persons.values() if p.get('updated_this_frame', False)])
                cv2.putText(frame, f"Track_Safe | Workers: {len(self.workers_data)} | Active: {tracked_count}", 
                           (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                # Show frame
                cv2.imshow("Track_Safe Enhanced Monitoring", frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    monitoring_active = False
                    break
                elif key == ord('r'):
                    print("🔄 Reloading database...")
                    self.load_data()
                    print(f"✅ Database reloaded - {len(self.workers_data)} workers")
                elif key == ord('s'):
                    # Save current frame
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"track_safe_frame_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"📸 Frame saved as {filename}")
        
        except KeyboardInterrupt:
            print("\n🛑 Track_Safe monitoring interrupted by user")
        except Exception as e:
            print(f"\n❌ Unexpected error during monitoring: {e}")
            print("System will attempt to continue. Press 'q' to quit safely.")
        finally:
            # Cleanup
            if cap:
                cap.release()
            cv2.destroyAllWindows()
            print("🔚 Track_Safe monitoring session ended")
            
        return True  # Successfully completed monitoring session


def main():
    """Main function to run Track_Safe system"""
    print("=" * 80)
    print("TRACK_SAFE - INTEGRATED WORKER RECOGNITION & PPE MONITORING SYSTEM")
    print("=" * 80)
    print("Enhanced with:")
    print("- Advanced person tracking with fake ID assignment")
    print("- Real-time face recognition integration")
    print("- Smart PPE compliance monitoring")
    print("- Intelligent logging and database management")
    print("- First_Middle_Last name format support")
    print("- SQLite database with enhanced schema")
    
    # Check required libraries
    try:
        import cv2
        import face_recognition
        import sqlite3
        from ultralytics import YOLO
        print("\n✓ All required libraries are available")
    except ImportError as e:
        print(f"\n✗ Missing required library: {e}")
        print("Please install required packages:")
        print("pip install opencv-python face-recognition ultralytics sqlite3 pandas numpy cvzone winsound")
        return
    
    try:
        # Create and run the enhanced system
        system = EnhancedTrackSafeSystem()
        system.setup_database()
        system.show_main_menu()
        
    except KeyboardInterrupt:
        print("\nTrack_Safe system shutdown by user")
    except Exception as e:
        print(f"Track_Safe system error: {e}")

if __name__ == "__main__":
    main()