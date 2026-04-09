# YOLO Detection

## Overview
This package uses YOLO to detect **SAM**, **buoys**, and other classes such as **Lolo**, **catamarans**, and **boats**.

The annotation and training pipeline is maintained in a separate repository:  
[alars_labeling_training](https://github.com/moyucrazy12/alars_labeling_training.git)

That repository also documents the trained models currently available for use in this pipeline.

The classes to detect, as well as their corresponding confidence thresholds, can be configured in:

```yaml
config/detection_parameters.yaml
```

---

## Dependencies
- ROS 2 Humble
- Ultralytics `8.3.160`
- NumPy `1.23.5`
- OpenCV `4.11.0`
- PyTorch `2.6.0`

When installing **Ultralytics**, some dependencies such as **PyTorch** and **OpenCV** will usually be installed automatically. See the official documentation for more details:  
[Ultralytics Quickstart](https://docs.ultralytics.com/quickstart/)

> **Note**  
> Running `pip3 install ultralytics` may also install `numpy 2.2.6` (observed as of November 2025). This can cause compatibility issues with the `tf_transformations` library. If that happens, remove the incompatible NumPy version and install the required one manually.

---

## Package Setup
Before using the package, make sure to build the workspace:

```bash
cd [ws_path]
colcon build --symlink-install --packages-select alars_auv_perception
source install/setup.sh
```

The trained models can be downloaded from:  
[alars_labeling_training](https://github.com/moyucrazy12/alars_labeling_training.git)

Place the model file(s) in:

```bash
config/models/
```

---

## Launch YOLO Detector

### 1. Launch only the YOLO detector
```bash
ros2 launch alars_auv_perception alars_yolodetector.launch.py namespace:=M350 device:=cpu use_sim_time:=true model_file:=yolo_model_4cls.pt
```

To open the RViz configuration file:

```bash
rviz2 -d <absolute_path>/perception/alars/auv_yolo_detector/config/M350_yolo.rviz
```

---

### 2. Launch the YOLO detector with a video
```bash
ros2 launch alars_auv_perception alars_video_yolo_detector.launch.py namespace:=M350 device:=cpu use_sim_time:=false
```

For video playback, it is recommended to use:

```bash
use_sim_time:=false
```

If CPU inference is too slow, consider using a GPU instead by setting:

```bash
device:=0
```

or another available GPU device.

To change the input video, edit the path in:

```bash
config/video_publisher_parameters.yaml
```

---

### 3. Launch the YOLO detector with a rosbag
```bash
ros2 launch alars_auv_perception alars_rosbag_yolo_detector.launch.py namespace:=M350 device:=cpu use_sim_time:=false
```

---

### 4. Play only a rosbag
Useful for frame collection or testing:

```bash
ros2 bag play --read-ahead-queue-size 1000 -r 1.0 --clock 100 --start-paused <rosbag_path>
```

It is recommended to create a `datasets/` folder to store videos and rosbags.

---

## Visualization Topics

| Topic | Message Type | Description |
|---|---|---|
| `/namespace/rviz/annotated_image` | `Image` | Camera image with YOLO annotations (bounding boxes and confidence scores). |
| `/namespace/rviz/blurred_sam` | `Image` | Sliced and rotated SAM window in grayscale, blurred before Canny edge detection. |
| `/namespace/rviz/edges` | `Image` | Output of the Canny edge detector. |

The last two topics are especially useful for debugging. In particular, the filter parameter:

```yaml
detection.blur_variance
```

may need adjustment depending on the scene. Ideally, waves should not generate edges, so the variance should be set to the smallest value that still suppresses them.

---

## Published Topics

| Topic | Message Type | Description |
|---|---|---|
| `/namespace/alars_detection/auv` | `PointStamped` | Estimated position of the detected SAM/AUV. |
| `/namespace/alars_detection/auv_obb` | `PolygonStamped` | Oriented bounding box of the detected SAM/AUV. |
| `/namespace/alars_detection/auv_head` | `PolygonStamped` | Estimated head/front region of the detected SAM/AUV. |
| `/namespace/alars_detection/buoy` | `PointStamped` | Estimated position of the detected buoy. |
| `/namespace/alars_detection/buoy_obb` | `PolygonStamped` | Oriented bounding box of the detected buoy. |
| `/namespace/estimated_other_obbs` | `PolygonStamped`* | Oriented bounding boxes for other detected classes. |

\* Adjust the message type if `estimated_other_obbs` uses a different message.

---

## Labeling and Training Pipeline

If you also want to use the labeling and training pipeline, see:  
[alars_labeling_training](https://github.com/moyucrazy12/alars_labeling_training.git)

This pipeline is kept separate from the ROS 2 perception pipeline and is intended only for annotation and training tasks.

It is separated because it depends on **Segment Anything Model 2 and 3**, and typically requires two different Conda environments to avoid dependency conflicts with the main perception pipeline.

You can find its installation instructions in the corresponding repository under the add-ons section.

---

## Maintainer
**Cristhian Mallqui Castro**  
ckmc@kth.se