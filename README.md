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
	py yolo_tracker.py -c {class_name} -v {video path} -dr {disapper_rate} -c {confidence}

	-dr = disapper_rate - minimum number of frames to disapper that id and start new one (if dr=10 , if object doesn't appear for 10 frames then it has to ignore the id of that object )
