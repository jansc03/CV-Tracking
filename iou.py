import numpy as np

def calculate_iou(box1, box2):
    """
    Berechnet die Intersection over Union (IoU) zweier Bounding-Boxes.

    Args:
        box1, box2: Bounding-Boxes im Format [x1, y1, w, h].

    Returns:
        IoU-Wert als Float.
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

def compare_trackers(yolo_tracks, custom_tracks):
    """
    Vergleicht die Ergebnisse von YOLO und Custom Tracker basierend auf IoU.

    Args:
        yolo_tracks: Liste der Bounding-Boxes des YOLO-Trackers.
        custom_tracks: Liste der Bounding-Boxes des Custom-Trackers.

    Returns:
        Dictionary mit Mittelwert, Minimum und Maximum der IoU-Werte.
    """
    iou_values = []

    for yolo_box in yolo_tracks:
        for custom_box in custom_tracks:
            iou = calculate_iou(yolo_box, custom_box)
            iou_values.append(iou)

    return {
        "mean_iou": np.mean(iou_values) if iou_values else 0,
        "min_iou": np.min(iou_values) if iou_values else 0,
        "max_iou": np.max(iou_values) if iou_values else 0,
    }

def write_results_to_file(results, filename="tracker_comparison_results.txt"):
    """
    Schreibt die Ergebnisse in eine Datei.

    Args:
        results: Dictionary mit IoU-Ergebnissen.
        filename: Name der Ausgabedatei.
    """
    with open(filename, "w") as file:
        file.write("Tracker Comparison Results\n")
        file.write("==========================\n")
        file.write(f"Mean IoU: {results['mean_iou']:.4f}\n")
        file.write(f"Min IoU: {results['min_iou']:.4f}\n")
        file.write(f"Max IoU: {results['max_iou']:.4f}\n")


# Integration in die Game-Loop (Beispiel für die Nutzung)
def process_frame(yolo_tracker, custom_tracker, frame):
    """
    Verarbeitet einen Frame, berechnet IoU und speichert Ergebnisse.

    Args:
        yolo_tracker: YOLO-Tracker-Instanz.
        custom_tracker: Custom-Tracker-Instanz.
        frame: Aktueller Frame des Videos.
    """
    # YOLO detection and tracking
    yolo_tracks = yolo_tracker.process_frame(frame)
    yolo_boxes = [(track.box[0], track.box[1], track.box[2] - track.box[0], track.box[3] - track.box[1]) for track in yolo_tracks]

    # Custom tracker results
    custom_tracks = custom_tracker.get_active_tracks()
    custom_boxes = [track["bbox"] for track_id, track in custom_tracks.items()]

    # Calculate IoU and write results
    results = compare_trackers(yolo_boxes, custom_boxes)
    write_results_to_file(results)

# Beispiel: Nutzung der Funktionen innerhalb der Game-Loop
# Diesen Abschnitt in die Hauptschleife der Pygame-Integration einfügen:
# process_frame(yolo_tracker, custom_tracker, original_vid)
