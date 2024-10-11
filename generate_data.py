import time
import cv2
from ultralytics import YOLOWorld, YOLO
import argparse
import json
import supervision as sv

"""
Example: generate images with no object (for reducing false positives)
$ python3 generate_data.py --imgdir=img20241010_01 --videoid=sunlight --label=NOTHING --autogen
    
Example: preview frames to get a feel of model performance
$ python3 generate_data.py --videoid=xiaomao1010 --label=xiaomao --view-only

Example: generate frames for a video clip known to be siama
$ python3 generate_data.py --imgdir=img20241010_01 --videoid=siama2 --label=siama --autogen
# Note: currently only support frames where only one object is detected. Frames with multiple objects or zero object are skipped.
"""

parser = argparse.ArgumentParser()
parser.add_argument("--videoid", help="Stem of the video file. Assuming video is in ./video")
parser.add_argument("--imgdir", help="Directory to save output images to. Not needed if --view-only.")
parser.add_argument("--label-model", default="yolov8l-worldv2.pt", help="Model used to produce ground truth box given prompt")
parser.add_argument("--my-model", default="../rpi-code/finetuned_ncnn_model", help="Current model, the performance of which to be evaluated")
parser.add_argument("--label", help="Object in the video. e.g. siama, xiaomao. Must be a class of my-model.")

# Modes: (1) view only (2) manual gen (3) auto gen (4) generate images with no label

parser.add_argument("--view-only", action="store_true", help="If true, iterate through frames annotated with both models' prediction")
parser.add_argument("--autogen", action="store_true", help="If true, generate a image+labeltxt for every frame in which label-model produces a prediction and my-model detects with confidence lower than the threshold below")
parser.add_argument("--confidence-threshold-for-skipping", default=0.7, help="Threshold for counting a my-model's detection as good enough")

args=parser.parse_args()

label = args.label
label_to_category = {
    "xiaomao": "cat",
    "siama": "cat",
    "xiaohua": "cat",
    "tabby": "cat",
    "possum": "possum",
    "raccoon": "raccoon",
    "angelica": "person",
    "beth": "dog",
    "ilan": "person",
    "charles": "person"
}

labels = sorted(label_to_category.keys())
label_idx = labels.index(label) if label != "NOTHING" else -1
# label_to_idx = {l:i for i,l in enumerate(labels)}


videoid = args.videoid
imgdir = args.imgdir

video_path = f"./videos/{videoid}.mp4"
cap = cv2.VideoCapture(video_path)
# generator = sv.get_video_frames_generator(video_path)

if label_idx != -1:
    label_model = YOLOWorld(args.label_model)
    classes = [label_to_category[label]] # ["cat", "possum", "raccoon", "dog"]
    label_model.set_classes(classes)

    test_model = YOLO(args.my_model)

img_idx = 0
WHITE_BOX_ANNOTATOR = sv.BoxAnnotator(thickness=1, color=sv.Color.WHITE)
BOX_ANNOTATOR = sv.BoxAnnotator(thickness=1)  
LEFT_LABEL_ANNOTATOR = sv.LabelAnnotator(text_color=sv.Color.BLACK)
RIGHT_LABEL_ANNOTATOR = sv.LabelAnnotator(text_color=sv.Color.BLACK, text_position=sv.Position.TOP_RIGHT)  

'''
for frame in generator:
    print(frame.shape)
    
    # results = model.infer(frame, confidence=0.002)
    # detections = sv.Detections.from_inference(results).with_nms(threshold=0.1)
    
    # annotated_image = BOX_ANNOTATOR.annotate(frame, detections)
    # annotated_image = LABEL_ANNOTATOR.annotate(annotated_image, detections)
    # print(type(annotated_image),  type(frame))
    
    results = model.predict(frame)
    # results[0].show()
    cv2.imshow("frame", results[0].plot())
    if cv2.waitKey(1) & 0xFF == 'q':
        break
    # sv.plot_image(annotated_image, (10,10))

'''


def convert_box(xyxy):
    img_width = 640
    img_height = 480
    x1, y1, x2, y2 = xyxy.flatten().tolist()
    centerx = (x1+x2)/2 / img_width 
    centery = (y1+y2)/2 / img_height
    width = (x2-x1) / img_width
    height = (y2-y1) / img_height
    return (centerx, centery, width, height)


def generate_annotate_labels(detections, keep_label=False):
    if keep_label:
        return [
        f"{class_name} {confidence:.2f}"
        for class_name, confidence
        in zip(detections['class_name'], detections.confidence)
        ]

    return [f"{confidence:.2f}" for confidence in detections.confidence]  

def save_img_and_label(frame, detections, img_idx):
    if detections is None:
        img_id = f"{videoid}_{str(img_idx).zfill(3)}"
        cv2.imwrite(f"./{imgdir}/{img_id}.jpg", frame)
        return img_idx + 1

    if len(detections) != 1:
        # TODO: support images with multiple detections. For now, skip them.
        return img_idx

    centerx, centery, width, height = convert_box(detections.xyxy)
    img_id = f"{videoid}_{str(img_idx).zfill(3)}"
    cv2.imwrite(f"./{imgdir}/{img_id}.jpg", frame)
    with open(f"./{imgdir}/{img_id}.txt", "w") as f:
        f.write(f"{label_idx} {centerx} {centery} {width} {height}")
    return img_idx + 1 

def detected_target(detections):
    filtered = detections[detections.class_id=label_idx]
    return filtered.confidence.max() >= args.confidence_threshold_for_skipping

frame_cnt = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        if frame_cnt > 0:
            break
        print(f'ret is False and frame is {frame}')
        time.sleep(1)
        continue

    frame_cnt += 1

    if label_idx == -1:
        img_idx = save_img_and_label(frame, None, img_idx)
        continue

    test_model_results = test_model(frame)[0]
    label_model_results = label_model(frame)[0]
    test_model_detections = sv.Detections.from_ultralytics(test_model_results)
    label_model_detections = sv.Detections.from_ultralytics(label_model_results).with_nms(threshold=0.2)
    print(label_model_detections)


    annotated_image = WHITE_BOX_ANNOTATOR.annotate(frame.copy(), label_model_detections)
    annotated_image = BOX_ANNOTATOR.annotate(annotated_image, test_model_detections)

    annotated_image = LEFT_LABEL_ANNOTATOR.annotate(annotated_image, label_model_detections, labels=generate_annotate_labels(label_model_detections))
    annotated_image = RIGHT_LABEL_ANNOTATOR.annotate(annotated_image, test_model_detections, labels=generate_annotate_labels(test_model_detections, keep_label=True))

    cv2.imshow("frame", annotated_image)

    if args.view_only or args.autogen:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if args.autogen and not detected_target(test_model_detections):
            img_idx = save_img_and_label(frame, label_model_detections, img_idx)
        continue

        
    key = cv2.waitKey(0) & 0xFF

    if key == ord('n'): # Next
        continue
    elif key == ord('q'): # Quit
        break
    elif key == ord('s'): # Save
        img_idx = save_img_and_label(frame, label_model_detections, img_idx)

cap.release()
cv2.destroyAllWindows()
