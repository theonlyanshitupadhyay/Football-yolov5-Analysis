from utilities.video_utils import read_video, save_video
from tracking_framework.track_object import Tracker
import cv2
import numpy as np
from team_identifier.team_assigner import TeamAssigner
from ball_possession.player_ball_assigner import PlayerBallAssigner
from camera_motion_analysis.camera_movement_estimator import CameraMovementEstimator
from view_transformer.view_transformer import ViewTransformer
from motion_metrics.speed_and_distance_estimator import SpeedAndDistance_Estimator

def main():
    # Path to the stub file containing precomputed tracks
    stub_path = '/teamspace/studios/this_studio/stubs/tracks.pkl'
    
    # Read video frames from a specified video file
    video_frames = read_video('/teamspace/studios/this_studio/demo_vid_1.mp4')

    # Initialize the object tracker with the specified model weights
    tracker = Tracker('/teamspace/studios/this_studio/runs/detect/train/weights/best.pt')

    # Get object tracks from the video frames, optionally using precomputed tracks from a stub file
    tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path='stubs/track_stubs.pkl')
    
    # Save a cropped image of a player from the first frame
    for track_id, player in tracks['players'][0].items():
        bbox = player['bbox']  # Get the bounding box of the player
        frame = video_frames[0]  # Get the first frame of the video

        # Crop the player's bounding box from the frame
        cropped_image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

        # Save the cropped image to a file
        cv2.imwrite(f'output_videos/cropped_image.jpg', cropped_image)
        break

    # Add position information to the tracks
    tracker.add_position_to_tracks(tracks)

    # Initialize the camera movement estimator with the first video frame
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    
    # Get camera movement for each frame, optionally using precomputed data from a stub file
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames, read_from_stub=True, stub_path='stubs/camera_movement_stub.pkl')
    
    # Adjust object positions based on the estimated camera movement
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

    # Initialize the view transformer
    view_transformer = ViewTransformer()
    
    # Add transformed positions to the tracks to account for changes in view
    view_transformer.add_transformed_position_to_tracks(tracks)

    # Interpolate ball positions to fill in missing data points
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # Initialize the speed and distance estimator
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    
    # Add speed and distance information to the tracks
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # Initialize the team assigner
    team_assigner = TeamAssigner()
    
    # Assign team colors to players in the first frame
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])

    # Assign teams to players for all frames
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num], track['bbox'], player_id)
            tracks['players'][frame_num][player_id]['team'] = team  # Assign team ID to the player
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]  # Assign team color to the player

    # Initialize the player ball assigner
    player_assigner = PlayerBallAssigner()
    
    # Track which team has ball control for each frame
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks['players']):
        if not tracks['ball'][frame_num]:
            team_ball_control.append(team_ball_control[-1] if team_ball_control else 0)  # Use the last team's control if no ball data
            continue

        ball_bbox = tracks['ball'][frame_num][1]['bbox']  # Get the ball's bounding box
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)  # Assign the ball to a player

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True  # Mark the player as having the ball
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])  # Record the team with ball control
        else:
            team_ball_control.append(team_ball_control[-1] if team_ball_control else 0)  # Default to the last team if no player is assigned

    team_ball_control = np.array(team_ball_control)  # Convert to a numpy array for easier manipulation
    print(f"Team ball control array: {team_ball_control}")

    # Draw annotations on video frames
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)

    # Draw camera movement on video frames
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)

    # Draw speed and distance information on video frames
    speed_and_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)

    # Save the annotated video to a file
    save_video(output_video_frames, 'output_videos/output_video.avi')

if __name__ == '__main__':
    main()
