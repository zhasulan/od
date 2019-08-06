



def non_max_suppression(detections, iou_threshold=.5):
    detections = sorted(detections, key=lambda detections: detections[2],
                        reverse=True)

    new_detections = []

    new_detections.append(detections[0])

    del detections[0]

    for index, detection in enumerate(detections):
        for new_detection in new_detections:
            if overlapping_area(detection, new_detection) > iou_threshold:
                del detections[index]
                break
        else:
            new_detections.append(detection)
            del detections[index]
    return new_detections
