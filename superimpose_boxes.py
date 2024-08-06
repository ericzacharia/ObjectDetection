import os
import cv2
from multiprocessing import Pool

def load_and_process_image(args):
    image_file, image_dir, bbox_dir, output_dir = args

    image_path = os.path.join(image_dir, image_file)
    bbox_file_path = os.path.join(bbox_dir, os.path.splitext(image_file)[0] + '.txt')

    if os.path.exists(bbox_file_path):
        image = cv2.imread(image_path)
        boxes = load_bounding_boxes(bbox_file_path, image.shape)
        annotated_image = draw_boxes_on_image(image, boxes)

        output_path = os.path.join(output_dir, image_file)
        cv2.imwrite(output_path, annotated_image)

def load_bounding_boxes(file_path, image_shape):
    with open(file_path, 'r') as file:
        boxes = [line.strip().split() for line in file.readlines()]

    normalized_boxes = [(float(cx), float(cy), float(w), float(h)) for _, cx, cy, w, h in boxes]
    scaled_boxes = []
    for cx, cy, w, h in normalized_boxes:
        x = int((cx - w / 2) * image_shape[1])
        y = int((cy - h / 2) * image_shape[0])
        width = int(w * image_shape[1])
        height = int(h * image_shape[0])
        scaled_boxes.append((x, y, width, height))

    return scaled_boxes

def draw_boxes_on_image(image, boxes):
    for x, y, w, h in boxes:
        top_left = (x, y)
        bottom_right = (x + w, y + h)
        image = cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
    return image

def superimpose_boxes(image_dir, bbox_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_files = [file for file in os.listdir(image_dir) if file.endswith(('.png', '.jpg', '.jpeg'))]
    pool_args = [(image_file, image_dir, bbox_dir, output_dir) for image_file in image_files]

    with Pool() as pool:
        pool.map(load_and_process_image, pool_args)

