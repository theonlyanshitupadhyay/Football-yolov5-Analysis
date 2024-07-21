from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8x')

# Perform prediction on the specified video file and save the results
results = model.predict('/teamspace/studios/this_studio/demo_vid_1.mp4', save=True)

# Check the structure of the results object
print("Results object structure:", type(results))

# Ensure results are not empty and have the expected structure
if results and hasattr(results[0], 'boxes'):
    print("First result structure:", type(results[0]))
    print("Detected boxes:", results[0].boxes)

    # Print the results for the first frame
    print(results[0])
    print('================================')

    # Iterate over and print each detected bounding box in the first frame
    for box in results[0].boxes:
        print(box)
else:
    print("No bounding boxes found or unexpected results structure.")
