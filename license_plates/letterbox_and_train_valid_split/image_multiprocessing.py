import cv2
import os
import numpy as np
from multiprocessing import Pool

def letterbox_image(image, padded_image_size=(2000, 2000)):
    image_height, image_width = image.shape[:2]
    padded_image_height, padded_image_width = padded_image_size
    # Check if the image is smaller than the padded size
    if image_width < padded_image_width and image_height < padded_image_height:
        # Image is smaller, place it randomly in the padded image
        delta_w = padded_image_width - image_width
        delta_h = padded_image_height - image_height
        top_pad = np.random.randint(0, delta_h)
        bottom_pad = delta_h - top_pad
        left_pad = np.random.randint(0, delta_w)
        right_pad = delta_w - left_pad
        scale = 1.0  # No scaling for smaller images
    else:
        # Image is larger, scale it down
        scale = min(padded_image_width / image_width, padded_image_height / image_height)
        new_width = int(image_width * scale)
        new_height = int(image_height * scale)
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        delta_w = padded_image_width - new_width
        delta_h = padded_image_height - new_height
        top_pad = delta_h // 2
        bottom_pad = delta_h - top_pad
        left_pad = delta_w // 2
        right_pad = delta_w - left_pad
    # Apply padding
    color = [0, 0, 0]  # Black padding
    padded_image = cv2.copyMakeBorder(image, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=color)

    return padded_image, scale, (left_pad, right_pad, top_pad, bottom_pad)

def adjust_boxes(file_path, image_shape, scale, padding):
    with open(file_path, 'r') as file:
        boxes = [line.strip().split() for line in file.readlines()]
    left_pad, right_pad, top_pad, bottom_pad = padding
    adjusted_boxes = []
    for _, cx, cy, w, h in boxes:
        # Convert from normalized to image coordinates
        cx, cy, w, h = [float(val) for val in [cx, cy, w, h]]
        cx = cx * image_shape[1]  # Convert to original image width
        cy = cy * image_shape[0]  # Convert to original image height
        w = w * image_shape[1]
        h = h * image_shape[0]
        # Scale the boxes
        cx = cx * scale + left_pad
        cy = cy * scale + top_pad
        w = w * scale
        h = h * scale
        x = int(cx - w / 2)
        y = int(cy - h / 2)
        width = int(w)
        height = int(h)
        adjusted_boxes.append((x, y, width, height))
    return adjusted_boxes


def draw_boxes_on_image(image, boxes):
    for x, y, w, h in boxes:
        top_left = (x, y)
        bottom_right = (x + w, y + h)
        image = cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
    return image


def normalize_boxes(boxes, image_shape):
    normalized_boxes = []
    if len(boxes) > 0:
        image_height, image_width, _ = image_shape
        for x, y, w, h in boxes:
            # Convert corner coordinates to center coordinates
            cx = x + w / 2
            cy = y + h / 2
            # Normalize coordinates round to 5 decimal places
            nx = round(cx / image_width, 5)
            ny = round(cy / image_height, 5)
            nw = round(w / image_width, 5)
            nh = round(h / image_height, 5)
            normalized_boxes.append((nx, ny, nw, nh))
    return normalized_boxes

def process_image(filename, input_dir, output_dir, target_size):
    image_dir = os.path.join(input_dir, 'images')
    label_dir = os.path.join(input_dir, 'labels')
    output_images_dir = os.path.join(output_dir, 'images')
    output_labels_dir = os.path.join(output_dir, 'labels')
    annotated_images_dir = os.path.join(output_dir, 'annotated_images')

    image_path = os.path.join(image_dir, filename)
    image = cv2.imread(image_path)

    if filename.startswith('1699'):
        scale_factor = min(np.random.uniform(0.03, 0.10) * target_size[0] / image.shape[1], 1.0)
        image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)

    padded_image, scale, padding = letterbox_image(image, target_size)

    # Save the padded image
    output_image_path = os.path.join(output_images_dir, filename)
    cv2.imwrite(output_image_path, padded_image)

    # Adjust the bounding box labels
    label_filename = os.path.splitext(filename)[0] + '.txt'
    label_path = os.path.join(label_dir, label_filename)
    if os.path.exists(label_path) and label_path.endswith('.txt'):
        try:
            adjusted_boxes = adjust_boxes(label_path, image.shape, scale, padding)
            annotated_image = draw_boxes_on_image(padded_image, adjusted_boxes)

            # Save the annotated image
            annotated_image_path = os.path.join(annotated_images_dir, filename)
            cv2.imwrite(annotated_image_path, annotated_image)

            # Save the adjusted bounding box labels
            output_label_path = os.path.join(output_labels_dir, label_filename)
            normalized_boxes = normalize_boxes(adjusted_boxes, padded_image.shape)
            with open(output_label_path, 'w') as f:
                for normalized_box in normalized_boxes:
                    f.write(f"0 {' '.join([str(x) for x in normalized_box])}" + "\n")
        except UnicodeDecodeError as e:
            print(f"Error reading file {label_path}: {e}")

def parallel_process(input_directory, output_directory, target_size, n_processes):
    # Create directories
    os.makedirs(os.path.join(output_directory, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_directory, 'labels'), exist_ok=True)
    os.makedirs(os.path.join(output_directory, 'annotated_images'), exist_ok=True)

    image_dir = os.path.join(input_directory, 'images')
    file_list = sorted(os.listdir(image_dir))

    # Use multiprocessing
    with Pool(n_processes) as p:
        p.starmap(process_image, [(filename, input_directory, output_directory, target_size) for filename in file_list])
