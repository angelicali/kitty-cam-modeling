import time
import cv2
from ultralytics import YOLOWorld, YOLO
# import supervision as sv
# from inference.models.yolo_world.yolo_world import YOLOWorld
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--videoid")
parser.add_argument("--imgdir")
parser.add_argument("--use-world-model", action='store_true')
parser.add_argument("--model")
parser.add_argument("--category")

args=parser.parse_args()

videoid = args.videoid
imgdir = args.imgdir

video_path = f"./videos/{videoid}.mp4"
cap = cv2.VideoCapture(video_path)
# generator = sv.get_video_frames_generator(video_path)


print(f"using model: {args.model}")
if args.use_world_model:
    model = YOLOWorld(args.model)
    classes = [args.category] # ["cat", "possum", "raccoon", "dog"]
    model.set_classes(classes)
else:
    model = YOLO(args.model)

# img_idx = 0
# BOX_ANNOTATOR = sv.BoxAnnotator(thickness=2)
# LABEL_ANNOTATOR = sv.LabelAnnotator(text_thickness=2, text_scale=1, text_color=sv.Color.BLACK)

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
# started = False
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print(f'ret is False and frame is {frame}')
        continue

    # started = True
    results = model(frame)[0]
    cv2.imshow("frame", results.plot())
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break
    '''
    key = cv2.waitKey(0) & 0xFF

    if key == ord('n'):
        continue
    elif key == ord('q'):
        break
    elif key == ord('s'):
        cv2.imwrite(f"./{imgdir}/{videoid}_{str(img_idx).zfill(3)}.jpg", frame)
        img_idx += 1 
#    if cv2.waitKey(25) & 0xFF == ord('q'):
#        break
'''
cap.release()
cv2.destroyAllWindows()
