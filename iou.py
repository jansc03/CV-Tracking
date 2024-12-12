import numpy as np

def calculate_iou(box1, box2):
    """
    Berechnet die Intersection over Union (IoU) zweier Bounding-Boxes.
    """
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[0] + box1[2], box2[0] + box2[2])
    y2_inter = min(box1[1] + box1[3], box2[1] + box2[3])

    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)

    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]

    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0

def compare_trackers_per_frame(yolo_boxes, custom_boxes):
    """
    Vergleicht die Ergebnisse von YOLO und Custom Tracker für einen Frame.
    """
    iou_values = []

    for yolo_box in yolo_boxes:
        for custom_box in custom_boxes:
            iou = calculate_iou(yolo_box, custom_box)
            iou_values.append(iou)

    return iou_values

def process_frame(yolo_tracker, custom_tracker, frame):
    """
    Berechnet die IoU-Werte für den aktuellen Frame.
    """
    # YOLO Tracker Bounding-Boxes
    yolo_tracks = yolo_tracker.process_frame(frame)
    yolo_boxes = [(track.box[0], track.box[1], track.box[2] - track.box[0], track.box[3] - track.box[1]) for track in yolo_tracks]

    # Custom Tracker Bounding-Boxes
    custom_tracks = custom_tracker.get_active_tracks()
    custom_boxes = [track["bbox"] for track_id, track in custom_tracks.items()]

    # Berechne IoU-Werte für diesen Frame
    return compare_trackers_per_frame(yolo_boxes, custom_boxes)

def aggregate_iou_results(all_iou_values):
    """
    Aggregiert die IoU-Werte über alle Frames hinweg.
    """
    flattened_values = [iou for frame_iou in all_iou_values for iou in frame_iou]

    return {
        "mean_iou": np.mean(flattened_values) if flattened_values else 0,
        "min_iou": np.min(flattened_values) if flattened_values else 0,
        "max_iou": np.max(flattened_values) if flattened_values else 0,
    }
