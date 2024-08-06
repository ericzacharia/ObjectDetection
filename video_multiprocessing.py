# video_processing.py
import supervision as sv
import os
from multiprocessing import Pool

HOME = os.getcwd()
VIDEO_DIR_PATH = f"{HOME}/videos"
IMAGE_DIR_PATH = f"{HOME}/images"
FRAME_STRIDE = 10
VIDEO_DIR_PATHS = sv.list_files_with_extensions(
    directory=VIDEO_DIR_PATH,
    extensions=["mov", "mp4"]
)

def process_video(video_path, image_dir_path, frame_stride):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    image_name_pattern = f"{video_name}-{{:05d}}.png"
    with sv.ImageSink(target_dir_path=image_dir_path, image_name_pattern=image_name_pattern) as sink:
        for image in sv.get_video_frames_generator(source_path=str(video_path), stride=frame_stride):
            sink.save_image(image=image)

def process_videos_in_parallel(video_paths=VIDEO_DIR_PATHS, image_dir_path=IMAGE_DIR_PATH, frame_stride=FRAME_STRIDE, num_processes=None):
    if num_processes is None:
        num_processes = os.cpu_count()
    # Create a list of tuples where each tuple contains the arguments for a single call to process_video
    args_for_process_video = [(video_path, image_dir_path, frame_stride) for video_path in video_paths]
    with Pool(processes=num_processes) as pool:
        pool.starmap(process_video, args_for_process_video)