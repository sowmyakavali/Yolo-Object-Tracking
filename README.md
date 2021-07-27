# Yolo-Object-Tracking

# Create Virtual Environment
	py -m venv venv
# Activate it
	cd venv\Scripts
	activate.bat

# Install all required libraries
	pip install -r requirements.txt

# Run Tracker
	cd tracker
	py yolo_tracker.py -c {class_name} -v {video path} -ch {chainage csv file} -s {true} 
	
	Ex: py yolo_tracker.py -c bush -v ..\video.mp4 -ch ..\video.csv -s true
