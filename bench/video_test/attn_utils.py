import cv2
import paddle
from PIL import Image


def get_video_loss(video_frames):
    baseline_video_path = "./video/cogvideo_sage_baseline.mp4"
    transform = paddle.vision.transforms.ToTensor()
    video_tensors = [transform(frame) for frame in video_frames]
    video_tensor = paddle.stack(video_tensors, dim=0).to("cuda")
    cap = cv2.VideoCapture(baseline_video_path)
    baseline_frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frame = transform(frame)
        baseline_frames.append(frame)
    cap.release()
    baseline_tensor = paddle.stack(baseline_frames, dim=0).to("cuda")
    if video_tensor.shape != baseline_tensor.shape:
        raise ValueError("Generated video and baseline video must have the same shape.")
    loss = paddle.nn.functional.mse_loss(input=video_tensor, label=baseline_tensor)
    return loss.item()
