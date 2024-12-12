import cv2
import torch
from motpy import Detection, MultiObjectTracker

class YOLOTracker:
    def __init__(self, model_path='ultralytics/yolov5', model_name='yolov5s', fps=30):
        # YOLOv5 model initialization
        self.model = torch.hub.load(model_path, model_name, pretrained=True)
        self.tracker = MultiObjectTracker(dt=1 / fps, tracker_kwargs={'max_staleness': 10})
        self.id_dict = {}
        self.next_id = 0

    def process_frame(self, frame):
        """Process a single frame, perform detection and tracking."""
        results = self.model(frame)
        output = results.pandas().xyxy[0]

        # Filter for "person" label
        objects = output[output['name'] == 'person']

        detections = []
        for _, obj in objects.iterrows():
            coordinates = [int(obj['xmin']), int(obj['ymin']), int(obj['xmax']), int(obj['ymax'])]
            detections.append(Detection(box=coordinates, score=obj['confidence'], class_id=obj['class']))

        # Update tracker with detections
        self.tracker.step(detections=detections)
        track_results = self.tracker.active_tracks()

        # Update ID dictionary
        self.update_id_dict(track_results)

        return track_results

    def update_id_dict(self, track_results):
        for track in track_results:
            if track.id not in self.id_dict:
                self.id_dict[track.id] = self.next_id
                self.next_id += 1

    def draw_boxes(self, frame, track_results):
        for obj in track_results:
            x, y, w, h = obj.box
            x, y, w, h = int(x), int(y), int(w), int(h)
            obj_id = obj.id
            confidence = obj.score
            cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {self.id_dict[obj_id]}: {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        return frame
