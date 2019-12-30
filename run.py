import os
import cv2
import time
import numpy as np

# import some common detectron2 utilities
from detectron2.config import get_cfg
# from detectron2.engine import DefaultTrainer
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode
# from detectron2.utils.visualizer import Visualizer
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances


cfg = get_cfg()
# cfg.merge_from_file("./detectron2_repo/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.merge_from_file("configs/mask_rcnn/mask_rcnn_R_50_FPN_3x.yaml")
#cfg.DATASETS.TRAIN = ("Clutterpics", "Coffee Cup Resized","Plastic Bottle Resized 1","Plastic Bottle Resized 2","Plastic Bottle Resized 3","Plastic Bottle Resized 4","Plastic Bottle Resized 5","Plastic Bottle Resized VD")
cfg.DATASETS.TEST = ()   # no metrics implemented for this dataset
cfg.DATALOADER.NUM_WORKERS = 2
#cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"  # initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.02
cfg.SOLVER.MAX_ITER =  10000  # 300 iterations seems good enough, but you can certainly train longer
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 300   # faster, and good enough for this toy dataset
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2   # 3 classes (data, fig, hazelnut)
cfg.MODEL.WEIGHTS = os.path.join("model_final_final_good.pth")
# cfg.MODEL.WEIGHTS = os.path.join("/content/drive/My Drive/THINGSWENEED/model_final_final_good.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4   # set the testing threshold for this model
cfg.DATASETS.TEST = ("test")
predictor = DefaultPredictor(cfg)

# we don't know why this is needed
# register_coco_instances("Clutterpics", {}, "clutterpics.json" ,"Clutterpics")
# register_coco_instances("Clutterpics", {},"clutterpics-lesslite.json","Clutterpics")
register_coco_instances("Clutterpics", {},"clutterpics-lite.json","Clutterpics")
plastic_metadata = MetadataCatalog.get("Clutterpics")
# whut, somehow this line needs to be here for the classes to show in the visualiser
DatasetCatalog.get("Clutterpics")

"""
import random
from detectron2.utils.visualizer import Visualizer

for i, d in enumerate(random.sample(dataset_dicts, 3)):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=plastic_metadata, scale=0.5)
    vis = visualizer.draw_dataset_dict(d)
    cv2.imwrite('{}.png'.format(i),vis.get_image()[:, :, ::-1])
    # cv2_imshow(vis.get_image()[:, :, ::-1])
# """

# exit()

cap = cv2.VideoCapture(2)
# cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture('video.mp4')

win_name = 'JAL'
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

viz = VideoVisualizer(plastic_metadata, instance_mode=ColorMode.IMAGE_BW)
if cap.isOpened():

    inference_time_cma = 0
    drawing_time_cma = 0
    n = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        tic = time.time()
        res = predictor(frame)
        toc = time.time()

        curr_inference_time = toc - tic
        inference_time_cma = (n * inference_time_cma + curr_inference_time) / (n+1)


        print('cma inference time: {:0.3} sec'.format(inference_time_cma))

        tic2 = time.time()

        drawned_frame = frame.copy() # make a copy of the original frame
        
        # draw on the frame with the res
        # v = Visualizer(drawned_frame[:, :, ::-1],
        #             metadata=plastic_metadata, 
        #             scale=0.8, 
        #             instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
        # )
        v_out = viz.draw_instance_predictions(drawned_frame, res["instances"].to("cpu"))
        # v_out = viz.draw_instance_predictions(drawned_frame[:, :, ::-1], res["instances"].to("cpu"))
        drawned_frame = v_out.get_image()
        
        cv2.imshow(win_name, drawned_frame)
        toc2 = time.time()

        curr_drawing_time = toc2 - tic2
        drawing_time_cma = (n * drawing_time_cma + curr_drawing_time) / (n+1)
        
        print('cma draw time: {:0.3} sec'.format(drawing_time_cma))

        if cv2.waitKey(1) & 0xff == ord('q'):
            break

        n += 1

cap.release()
print('Done.')
