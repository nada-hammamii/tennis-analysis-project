# Tennis match video analysis

Computer-vision pipeline that turns a tennis match video into match statistics: player speed, ball shot speed and shot count, drawn on the output video together with a top-down mini-court.

![Output frame](https://github.com/user-attachments/assets/afcb4417-616d-420a-b218-7d271bfa3480)

## How it works

1. **Player detection** — a pretrained YOLOv8x model detects people on each frame; the two players are then selected by their distance to the court keypoints, so spectators and umpires are filtered out.
2. **Ball detection** — a YOLOv5 model fine-tuned on the [Roboflow tennis-ball-detection dataset](https://universe.roboflow.com/viren-dhanwani/tennis-ball-detection/dataset/6) tracks the ball; missing detections are interpolated.
3. **Court keypoints** — a ResNet-50 regressor (PyTorch) predicts the 14 court keypoints from the first frame.
4. **Mini-court projection** — player and ball positions are projected onto a schematic court using the keypoints, which converts pixel distances to metres.
5. **Shot detection** — a shot is detected when the ball's vertical direction changes for enough consecutive frames (see `analysis/ball_analysis.ipynb` for the exploration behind this rule).
6. **Statistics** — for every shot: ball speed (km/h), the player who hit it, and the opponent's running speed. Running totals and averages are drawn on each frame.

```
tennis_analysis/
├── main.py                   # full pipeline, writes output_videos/output_video.avi
├── trackers/                 # PlayerTracker (YOLOv8), BallTracker (fine-tuned YOLOv5)
├── court_line_detector/      # ResNet-50 court keypoint model
├── mini_court/               # top-down court drawing and coordinate conversion
├── utils/                    # video I/O, bbox maths, unit conversions, stats overlay
├── constants/                # real court dimensions (metres)
├── analysis/                 # notebook used to design the shot-detection rule
├── training/                 # notebooks that trained the ball detector and the keypoint model
└── tracker_stubs/            # cached detections so the pipeline runs without a GPU
```

## Run it

```bash
pip install -r requirements.txt
cd tennis_analysis
```

1. Put a match video at `input_videos/input_video.mp4`.
2. Download the trained weights into `models/`:
   - ball detector `yolo5_last.pt` — [Google Drive](https://drive.google.com/file/d/1UZwiG1jkWgce9lNhxJ2L0NVjX1vGM05U/view?usp=sharing)
   - court keypoints `keypoints_model.pth` — [Google Drive](https://drive.google.com/file/d/1QrTOF1ToQ4plsSZbkBs3zOLkVt3MBlta/view?usp=sharing)
3. Run `python main.py`. The annotated video is written to `output_videos/output_video.avi`.

`main.py` reads detections from `tracker_stubs/` by default (`read_from_stub=True`), which lets the pipeline run on CPU for the sample video. Set it to `False` to run the detectors on your own video.

## Training

- `training/tennis_ball_detector_training.ipynb` — fine-tunes YOLOv5 on the Roboflow dataset (`training/tennis-ball-detection-6/`, CC BY 4.0).
- `training/tennis_court_keypoints_training.ipynb` — trains the ResNet-50 keypoint regressor.

## Stack

Python · PyTorch · torchvision · Ultralytics YOLO · OpenCV · pandas · NumPy
