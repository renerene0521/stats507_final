import argparse
from pathlib import Path

import cv2
import mediapipe as mp
from PIL import Image
import PIL

parser = argparse.ArgumentParser(description="Align_face")

parser.add_argument(
    "-unprocessed_dir",
    type=str,
    default="unprocessed",
    help="directory with unprocessed images",
)
parser.add_argument(
    "-output_dir",
    type=str,
    default="input/face",
    help="output directory",
)

parser.add_argument(
    "-output_size",
    type=int,
    default=1024,
    help="size to resize the aligned faces to, must be power of 2",
)
parser.add_argument("-seed", type=int, help="manual seed to use")
parser.add_argument(
    "-cache_dir",
    type=str,
    default="cache",
    help="(unused in Mediapipe version, kept for compatibility)",
)

parser.add_argument(
    "-inter_method",
    type=str,
    default="bicubic",
    help="resize interpolation method: nearest, bilinear, bicubic, lanczos",
)

args = parser.parse_args()

output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

# PIL interpolation mapping
INTERP_MAP = {
    "nearest": PIL.Image.NEAREST,
    "bilinear": PIL.Image.BILINEAR,
    "bicubic": PIL.Image.BICUBIC,
    "lanczos": PIL.Image.LANCZOS,
}
interp = INTERP_MAP.get(args.inter_method.lower(), PIL.Image.BICUBIC)

mp_face_detection = mp.solutions.face_detection

print("Running face alignment with Mediapipe FaceDetection")

# 使用 Mediapipe FaceDetection 取得臉部 bounding box，並裁切儲存
with mp_face_detection.FaceDetection(
    model_selection=1, min_detection_confidence=0.5
) as face_detector:

    for im in Path(args.unprocessed_dir).glob("*.*"):
        img_bgr = cv2.imread(str(im))
        if img_bgr is None:
            print(f"Warning: cannot read image {im}")
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = img_rgb.shape

        results = face_detector.process(img_rgb)

        if not results.detections:
            print(f"No face detected in {im}, skipping.")
            continue

        faces = []

        for det in results.detections:
            bbox = det.location_data.relative_bounding_box

            x_min = int(bbox.xmin * w)
            y_min = int(bbox.ymin * h)
            box_w = int(bbox.width * w)
            box_h = int(bbox.height * h)

            # 加一點 margin，讓臉不會裁得太緊
            margin = 0.3
            x_min_m = int(max(0, x_min - margin * box_w))
            y_min_m = int(max(0, y_min - margin * box_h))
            x_max_m = int(min(w, x_min + box_w + margin * box_w))
            y_max_m = int(min(h, y_min + box_h + margin * box_h))

            face_rgb = img_rgb[y_min_m:y_max_m, x_min_m:x_max_m]
            if face_rgb.size == 0:
                continue

            face_pil = Image.fromarray(face_rgb)
            faces.append(face_pil)

        if not faces:
            print(f"No valid face crops for {im}, skipping.")
            continue

        for i, face in enumerate(faces):
            if args.output_size:
                face = face.resize(
                    (args.output_size, args.output_size), interp
                )

            if len(faces) > 1:
                save_path = output_dir / f"{im.stem}_{i}.png"
            else:
                save_path = output_dir / f"{im.stem}.png"

            face.save(save_path)
            print(f"Saved aligned face to {save_path}")
