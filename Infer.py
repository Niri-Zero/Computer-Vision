import numpy as np
import torch
import cv2
import copy
from torchvision.ops import nms
import onnxruntime as ort
import time
import cProfile
import pstats
import io
from pstats import SortKey


conf_thresh = 0.4
iou_thresh = 0.4


sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
ort_sess = ort.InferenceSession(
    "/Users/leo/Work/Machine Learning/Projects/Drone/Weights/best.quant-opt-shape.onnx", sess_options, providers=["CPUExecutionProvider"])


def xywhn2xyxy(box, padw=0, padh=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    xyxy = box.clone() if isinstance(box, torch.Tensor) else np.copy(box)
    w = xyxy[:, 2]
    h = xyxy[:, 3]

    xyxy[:, 0] = (xyxy[:, 0] - w / 2) + padw  # top left x
    xyxy[:, 1] = (xyxy[:, 1] - h / 2) + padh  # top left y
    xyxy[:, 2] = xyxy[:, 0] + w + padw  # bottom right x
    xyxy[:, 3] = xyxy[:, 1] + h + padh  # bottom right y

    return xyxy


def drawRects(boxes, img):
    for box in boxes:
        x1, y1, x2, y2, box_conf, _ = box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        rect = cv2.rectangle(img, (x1, y1), (x2, y2),
                             color=(0, 0, 255), thickness=2)
        cv2.putText(rect, f'Empty Shelf {round(box_conf.item(), 2)}', (
            int(x1), int(y1 - 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)


cam = cv2.VideoCapture(0)
cv2.namedWindow("image")
while True:
    # pr = cProfile.Profile()
    # pr.enable()

    ret, frame = cam.read()
    if not ret:
        print("failed to grab")
        break
    start = time.time()

    frame = cv2.resize(frame, (160, 128), interpolation=cv2.INTER_AREA)

    inp = ort.OrtValue.ortvalue_from_numpy(np.array(torch.tensor(frame[None, ...]).permute(
        0, 3, 1, 2).contiguous() / 255.0), "cpu")
    outputs = ort_sess.run(None, {'images': inp})

    outputs = outputs[0][0]
    selected = outputs[outputs[:, 4] >= conf_thresh]
    selected = torch.tensor(xywhn2xyxy(selected))
    final_boxes = nms(
        boxes=selected[:, :4], scores=selected[:, 4], iou_threshold=iou_thresh)
    final_boxes = torch.index_select(selected, 0, final_boxes)

    drawRects(final_boxes, frame)
    cv2.imshow("image", frame)

    # pr.disable()
    # s = io.StringIO()
    # sortby = SortKey.CUMULATIVE
    # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    # ps.print_stats()
    # print(s.getvalue())

    k = cv2.waitKey(1)
    if k % 256 == 27:
        break

cam.release()
cv2.destroyAllWindows()
