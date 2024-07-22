# Football Game Analysis using YOLO

This repository contains code and tools for analyzing football games using the YOLO (You Only Look Once) object detection model. The project is designed to track players, referees, and the ball, calculate speeds and distances, and estimate camera movement, providing comprehensive insights into the game.

## Features

- **Object Detection:** Utilizes the YOLO model to detect players, referees, and the ball in each frame of a football game.
- **Object Tracking:** Implements ByteTrack to keep track of detected objects across frames.
- **Position Transformation:** Transforms pixel coordinates to real-world coordinates for accurate tracking.
- **Speed and Distance Estimation:** Calculates the speed and distance covered by players throughout the game.
- **Camera Movement Estimation:** Estimates and adjusts for camera movement to maintain accurate tracking.
- **Ball Possession:** Determines which player is in control of the ball based on proximity.
- **Visualization:** Annotates video frames with detected objects, speeds, distances, and ball possession information.

## Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/theonlyanshitupadhyay/Football-yolov5-Analysis.git
    cd Football-yolov5-Analysis
    ```

2. **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Object Detection and Tracking:**
    ```python
    from tracker import Tracker
    
    tracker = Tracker(model_path='path_to_yolo_model')
    tracks = tracker.get_object_tracks(video_frames)
    ```

2. **Position Transformation:**
    ```python
    from view_transformer import ViewTransformer
    
    transformer = ViewTransformer()
    transformer.add_transformed_position_to_tracks(tracks)
    ```

3. **Speed and Distance Estimation:**
    ```python
    from speed_distance_estimator import SpeedAndDistance_Estimator
    
    estimator = SpeedAndDistance_Estimator()
    estimator.add_speed_and_distance_to_tracks(tracks)
    ```

4. **Camera Movement Estimation:**
    ```python
    from camera_movement_estimator import CameraMovementEstimator
    
    estimator = CameraMovementEstimator(frame=video_frames[0])
    camera_movement = estimator.get_camera_movement(video_frames)
    estimator.add_adjust_positions_to_tracks(tracks, camera_movement)
    ```

5. **Ball Possession Assignment:**
    ```python
    from player_ball_assigner import PlayerBallAssigner
    
    assigner = PlayerBallAssigner()
    for frame_num, frame_tracks in enumerate(tracks['ball']):
        ball_bbox = frame_tracks[1]['bbox']
        player_id = assigner.assign_ball_to_player(tracks['players'][frame_num], ball_bbox)
        if player_id != -1:
            tracks['players'][frame_num][player_id]['has_ball'] = True
    ```

## Visualization

Annotate frames with tracking information:
```python
from tracker import Tracker

tracker = Tracker()
annotated_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)
```

## Output Video

Here is a sample output video demonstrating the analysis results:







https://github.com/user-attachments/assets/0fd53359-4f95-4770-9976-85e7b623d894


## Lightning AI Studio

You can view and interact with the live session of this project on Lightning AI Studio:

[Lightning AI Studio Live Session](https://lightning.ai/live-session/c329b419-e274-4d0d-859f-9148d4c503aa)

## Contributing

If you'd like to contribute to this project, please fork the repository and submit a pull request. For detailed information on how to contribute, refer to the [CONTRIBUTING.md](CONTRIBUTING.md) file.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [YOLOv5](https://github.com/ultralytics/yolov5) for the object detection model.
- [Git LFS](https://git-lfs.github.com) for handling large files.
