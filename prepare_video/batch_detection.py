'''
use detectron2 library to detecated the person centered video

'''
# %% 
import torch
import numpy as np

import detectron2
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor

from torchvision.transforms.functional import crop, pad, resize

# %%

class Batch_Detection():
    def __init__(self, img_size: int) -> None:

        # set for detection
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.90  # set threshold for this model
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
        self.predictor = DefaultPredictor(cfg)

        self.img_size = img_size

    def get_person_bboxes(self, inp_img:torch.tensor, predictor):
        '''
        based the detectron2 api to get the location.

        Args:
            inp_img (torch.tensor): frame of video
            predictor (_type_): a predictor function

        Returns:
            list: bbox with predictions
        '''
        predictions = predictor(inp_img.cpu().detach().numpy())['instances'].to('cpu')
        boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
        scores = predictions.scores if predictions.has("scores") else None
        classes = np.array(predictions.pred_classes.tolist() if predictions.has("pred_classes") else None)
        predicted_boxes = boxes[np.logical_and(classes==0, scores>0.80 )].tensor.cpu() # only person
        return predicted_boxes, predictions

    def get_center_point(self, box:torch.tensor):
        '''
        from the bbox to get the center point, for the after calc.

        Args:
            box (torch.tensor): (x1, y1, x2, y2)

        Returns:
            tensor: (new_x, new_y) of center point, (x1, y1, x2, y2) of bbox.
        '''
        x1, y1, x2, y2 = box

        new_x = (x2 - x1) / 2 + x1 
        new_y = (y2 - y1) / 2 + y1

        return (new_x, new_y), (x1, y1, x2, y2)

    def get_frame_box(self, inp_imgs:list):
        '''
        get the predict bbox from the inp_imgs

        Args:
            inp_imgs (list): (t, h, w, c)

        Returns:
            list: frame_list (h, w, c), box_list, pred_list
        '''

        frame_list = []
        box_list = []
        pred_list = []
        CENTER_POINT = 0

        frames, h, w, c = inp_imgs.size()
        
        # 1 batch different frame
        for frame in range(frames):

            inp_img = inp_imgs[frame, :, :, :] # (h, w, c)

            predicted_boxes, pred = self.get_person_bboxes(inp_img, self.predictor)

            # determin which is the person and which is the doctor
            if predicted_boxes.shape == (2, 4): # two boxes, patient and doctor
            
                center_point_1, coord_list_1 = self.get_center_point(predicted_boxes[0])
                center_point_2, coord_list_2 = self.get_center_point(predicted_boxes[1])

                if CENTER_POINT == 0:

                    x1_1, y1_1, x2_1, y2_1 = coord_list_1
                    x1_2, y1_2, x2_2, y2_2 = coord_list_2

                    box_1_height = y2_1 - y1_1
                    box_2_height = y2_2 - y1_2

                    if box_1_height > box_2_height:
                        predicted_boxes = predicted_boxes[1]
                        CENTER_POINT = center_point_2
                    else:
                        predicted_boxes = predicted_boxes[0]
                        CENTER_POINT = center_point_1

                else:

                    distance_1 = torch.abs(center_point_1[0] - CENTER_POINT[0])
                    distance_2 = torch.abs(center_point_2[0] - CENTER_POINT[0])

                    if distance_1 < distance_2:
                        predicted_boxes = predicted_boxes[0]
                        CENTER_POINT = center_point_1
                    else:
                        predicted_boxes = predicted_boxes[1]
                        CENTER_POINT = center_point_2

                frame_list.append(inp_img)
                box_list.append(predicted_boxes.unsqueeze(dim=0))
                pred_list.append(pred)

            elif predicted_boxes.shape == (1, 4): # one box, only one person
                
                center_point, _ = self.get_center_point(predicted_boxes[0])

                if CENTER_POINT != 0:

                    distance = torch.abs(center_point[0] - CENTER_POINT[0])

                    if distance < 100:

                        frame_list.append(inp_img)
                        box_list.append(predicted_boxes)
                        pred_list.append(pred)

                        CENTER_POINT = center_point

                else:

                    frame_list.append(inp_img)
                    box_list.append(predicted_boxes)
                    pred_list.append(pred)

                    CENTER_POINT = center_point
            
        return frame_list, box_list, pred_list, CENTER_POINT

    def clip_pad_with_bbox(self, imgs: list, boxes: list, img_size: int = 256, bias:int = 10):
        '''
        based torchvision function to crop, pad, resize img.

        clip with the bbox, (x1-bias, y1) and padd with the (gap-bais) in left and right.

        Args:
            imgs (list): imgs with (h, w, c)
            boxes (list): (x1, y1, x2, y2)
            img_size (int, optional): croped img size. Defaults to 256.
            bias (int, optional): the bias of bbox, with the (x1-bias) and (x2+bias). Defaults to 5.

        Returns:
            tensor: (c, t, h, w)
        ''' 

        frame_list = []

        for num in range(len(imgs)):

            x1, y1, x2, y2 = boxes[num].int().squeeze() # dtype must int for resize, crop function

            box_width = x2 - x1
            box_height = y2 - y1 

            width_gap = ((box_height - box_width) / 2).int() # keep int type

            img = imgs[num].permute(2, 0, 1) # (h, w, c) to (c, h, w), for pytorch function

            # give a bias for the left and right crop bbox.
            croped_img = crop(img, top=y1, left=(x1 - bias), height=box_height, width=(box_width + 2 * bias))

            pad_img = pad(croped_img, padding=(width_gap - bias, 0), fill=0)

            resized_img = resize(pad_img, size=(img_size, img_size))

            frame_list.append(resized_img)

        return torch.stack(frame_list, dim=1) # c, t, h, w

    def clip_with_bbox(self, imgs: list, boxes: list, img_size: int = 256):
        
        frame_list = []

        for num in range(len(imgs)):

            x1, y1, x2, y2 = boxes[num].int().squeeze() # dtype must int for resize, crop function

            box_width = x2 - x1
            box_height = y2 - y1 

            width_gap = ((box_height - box_width) / 2 ).int() # keep int type

            img = imgs[num].permute(2, 0, 1) # (h, w, c) to (c, h, w), for pytorch function

            croped_img = crop(img, top=y1, left=(x1 - width_gap), height=box_height, width=box_height)

            resized_img = resize(croped_img, size=(img_size, img_size))

            frame_list.append(resized_img)

        return torch.stack(frame_list, dim=1)

    def handel_batch_imgs(self, video_frame, flag: str = 'pad'):

        t, h, w, c = video_frame.size()

        frame_list, box_list, pred_list, CENTER_POINT = self.get_frame_box(video_frame) # h, w, c

        if flag == 'pad':
            one_batch = self.clip_pad_with_bbox(frame_list, box_list, self.img_size) # c, t, h, w
        else:
            one_batch = self.clip_with_bbox(frame_list, box_list, self.img_size)

        return one_batch
