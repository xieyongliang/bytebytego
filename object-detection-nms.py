import numpy as np

def nms(boxes, scores, iou_threshold):
    indices = np.argsort(scores)[::-1]  # Sort by confidence scores in descending order
    keep = []

    while len(indices) > 0:
        # Pick the box with the highest score
        current_idx = indices[0]
        keep.append(current_idx)

        # Calculate the IoU for the remaining boxes
        current_box = boxes[current_idx]
        other_boxes = boxes[indices[1:]]

        iou = compute_iou(current_box, other_boxes)

        # Select boxes with IoU less than the threshold
        indices = indices[1:][iou < iou_threshold]

    return keep

def compute_iou(box, other_boxes):
    # Compute the intersection-over-union (IoU) between two sets of boxes
    x1 = np.maximum(box[0], other_boxes[:, 0])
    y1 = np.maximum(box[1], other_boxes[:, 1])
    x2 = np.minimum(box[2], other_boxes[:, 2])
    y2 = np.minimum(box[3], other_boxes[:, 3])

    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    other_areas = (other_boxes[:, 2] - other_boxes[:, 0]) * (other_boxes[:, 3] - other_boxes[:, 1])

    union = box_area + other_areas - intersection

    return intersection / union

# Example usage
boxes = np.array([[100, 100, 200, 200], [110, 110, 210, 210], [50, 50, 150, 150]])
scores = np.array([0.9, 0.75, 0.6])
iou_threshold = 0.5
result = nms(boxes, scores, iou_threshold)
print(result)
