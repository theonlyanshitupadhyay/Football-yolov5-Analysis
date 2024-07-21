import numpy as np
import cv2

class ViewTransformer():
    def __init__(self):
        # Define court dimensions
        court_width = 68
        court_length = 23.32

        # Define pixel coordinates of the court's vertices in the image
        self.pixel_vertices = np.array([
            [110, 1035], 
            [265, 275], 
            [910, 260], 
            [1640, 915]
        ])

        # Define real-world coordinates of the court's vertices
        self.target_vertices = np.array([
            [0, court_width],
            [0, 0],
            [court_length, 0],
            [court_length, court_width]
        ])

        # Convert vertices to float32 type
        self.pixel_vertices = self.pixel_vertices.astype(np.float32)
        self.target_vertices = self.target_vertices.astype(np.float32)

        # Compute the perspective transform matrix
        self.persepctive_trasnformer = cv2.getPerspectiveTransform(self.pixel_vertices, self.target_vertices)

    def transform_point(self, point):
        # Convert point to integer tuple
        p = (int(point[0]), int(point[1]))

        # Check if the point is inside the polygon defined by pixel vertices
        is_inside = cv2.pointPolygonTest(self.pixel_vertices, p, False) >= 0
        if not is_inside:
            return None

        # Reshape point for perspective transform
        reshaped_point = point.reshape(-1, 1, 2).astype(np.float32)

        # Transform the point using the perspective transform matrix
        tranform_point = cv2.perspectiveTransform(reshaped_point, self.persepctive_trasnformer)

        # Return the transformed point
        return tranform_point.reshape(-1, 2)

    def add_transformed_position_to_tracks(self, tracks):
        # Iterate over all tracked objects
        for object, object_tracks in tracks.items():
            # Iterate over frames for each object
            for frame_num, track in enumerate(object_tracks):
                # Iterate over individual tracks in each frame
                for track_id, track_info in track.items():
                    # Get the adjusted position of the track
                    position = track_info['position_adjusted']
                    position = np.array(position)

                    # Transform the position
                    position_trasnformed = self.transform_point(position)
                    if position_trasnformed is not None:
                        position_trasnformed = position_trasnformed.squeeze().tolist()

                    # Add the transformed position to the track information
                    tracks[object][frame_num][track_id]['position_transformed'] = position_trasnformed
