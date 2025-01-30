# %%
import torch
import numpy as np
from ultralytics import YOLO

from torchvision.transforms.functional import crop, pad, resize

# %% 
class Batch_Detection_YOLOv8():
    def __init__(self, img_size: int, model_path: str = "yolov8n.pt") -> None:
        """
        Constructor for Batch_Detection_YOLOv8.

        Args:
            img_size (int): The size to which cropped images will be resized.
            model_path (str): Path to the YOLOv8 model. Defaults to "yolov8n.pt".
        """
        # Load YOLOv8 model
        self.model = YOLO(model_path)
        self.model.to('cuda')
        self.img_size = img_size

    def get_person_bboxes(self, inp_img: torch.tensor):
        """
        Get the bounding boxes for detected persons.

        Args:
            inp_img (torch.tensor): A single image frame.

        Returns:
            list: Detected bounding boxes with predictions.
        """
        results = self.model.predict(inp_img, conf=0.5, iou=0.5, classes=[0], device="cuda" if torch.cuda.is_available() else "cpu")
        bboxes = results[0].boxes.xyxy if results[0].boxes else torch.empty((0, 4))
        return bboxes

    def get_center_point(self, box: torch.tensor):
        """
        Calculate the center point of a bounding box.

        Args:
            box (torch.tensor): (x1, y1, x2, y2)

        Returns:
            tuple: (new_x, new_y) of the center point, (x1, y1, x2, y2) of bbox.
        """
        x1, y1, x2, y2 = box
        new_x = (x2 - x1) / 2 + x1
        new_y = (y2 - y1) / 2 + y1
        return (new_x, new_y), (x1, y1, x2, y2)

    def get_frame_box(self, inp_imgs: list):
        """
        Get the predicted bounding boxes for all frames.

        Args:
            inp_imgs (list): (t, h, w, c)

        Returns:
            list: frame_list (h, w, c), box_list, CENTER_POINT
        """
        frame_list, box_list = [], []
        CENTER_POINT = 0

        for frame in inp_imgs:
            predicted_boxes = self.get_person_bboxes(frame)

            if len(predicted_boxes) == 2:  # Two persons detected
                center_point_1, coord_list_1 = self.get_center_point(predicted_boxes[0])
                center_point_2, coord_list_2 = self.get_center_point(predicted_boxes[1])

                if CENTER_POINT == 0:
                    height_1 = coord_list_1[3] - coord_list_1[1]
                    height_2 = coord_list_2[3] - coord_list_2[1]
                    if height_1 > height_2:
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

                frame_list.append(frame)
                box_list.append(predicted_boxes.unsqueeze(dim=0))

            elif len(predicted_boxes) == 1:  # One person detected
                center_point, _ = self.get_center_point(predicted_boxes[0])
                if CENTER_POINT != 0 and torch.abs(center_point[0] - CENTER_POINT[0]) < 100:
                    frame_list.append(frame)
                    box_list.append(predicted_boxes)
                    CENTER_POINT = center_point
                else:
                    frame_list.append(frame)
                    box_list.append(predicted_boxes)
                    CENTER_POINT = center_point

        return frame_list, box_list, CENTER_POINT

    def clip_pad_with_bbox(self, imgs: list, boxes: list, img_size: int = 256, bias: int = 10):
        """
        Crop, pad, and resize images based on bounding boxes.

        Args:
            imgs (list): Images with shape (h, w, c).
            boxes (list): Bounding boxes (x1, y1, x2, y2).
            img_size (int, optional): Cropped image size. Defaults to 256.
            bias (int, optional): Padding around bounding boxes. Defaults to 10.

        Returns:
            torch.tensor: Cropped and resized images (c, t, h, w).
        """
        frame_list = []

        for num in range(len(imgs)):
            x1, y1, x2, y2 = boxes[num].int().squeeze()

            box_width = x2 - x1
            box_height = y2 - y1
            width_gap = ((box_height - box_width) / 2).int()

            img = imgs[num].permute(2, 0, 1)  # (h, w, c) -> (c, h, w)
            cropped_img = crop(img, top=y1, left=(x1 - bias), height=box_height, width=(box_width + 2 * bias))
            padded_img = pad(cropped_img, padding=(width_gap - bias, 0), fill=0)
            resized_img = resize(padded_img, size=(img_size, img_size))
            frame_list.append(resized_img)

        return torch.stack(frame_list, dim=1)  # c, t, h, w

    def handle_batch_imgs(self, video_frame, flag: str = 'pad'):
        """
        Process a batch of video frames to extract cropped images.

        Args:
            video_frame (torch.tensor): Video frames (t, h, w, c).
            flag (str, optional): Padding flag. Defaults to 'pad'.

        Returns:
            torch.tensor: Processed frames (c, t, h, w).
        """
        t, h, w, c = video_frame.size()
        frame_list, box_list, CENTER_POINT = self.get_frame_box(video_frame)

        if flag == 'pad':
            one_batch = self.clip_pad_with_bbox(frame_list, box_list, self.img_size)
        else:
            one_batch = self.clip_pad_with_bbox(frame_list, box_list, self.img_size)

        return one_batch
