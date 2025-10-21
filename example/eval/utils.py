import os

import cv2
import t2v_metrics


def video_to_images(video_path, output_folder, format="jpg"):
    """
    将视频转换为图片序列

    参数:
        video_path: 视频文件的路径
        output_folder: 保存图片的文件夹路径
        format: 输出图片的格式，默认为'jpg'
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"视频帧率: {fps} fps")
    print(f"总帧数: {total_frames}")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if ret:
            output_path = os.path.join(
                output_folder, f"frame_{frame_count:06d}.{format}"
            )
            cv2.imwrite(output_path, frame)
            frame_count += 1
            if frame_count % 100 == 0:
                print(f"已处理 {frame_count} 帧")
        else:
            break
    cap.release()


def update_video_list(video_list_path):
    video_list_path = "/data/charles/codes/flash-attn-triton/video"
    video_list = os.listdir(video_list_path)
    for video_name in video_list:
        video_path = os.path.join(video_list_path, video_name)
        output_folder = os.path.join(video_list_path, video_name.split(".")[0])
        video_to_images(video_path, output_folder)


def show_model_list():
    print("VQAScore models:" + str(t2v_metrics.list_all_vqascore_models()))
    print("ITMScore models:" + str(t2v_metrics.list_all_itmscore_models()))
    print("CLIPScore models:" + str(t2v_metrics.list_all_clipscore_models()))
