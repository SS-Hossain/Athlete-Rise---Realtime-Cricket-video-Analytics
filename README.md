# AthleteRise – AI-Powered Cricket Analytics

This project provides a real-time biomechanical analysis of a cricket cover drive from a full video. Using computer vision, it performs pose estimation on the athlete frame-by-frame, calculates key performance metrics, and generates an annotated video with live feedback, along with a final performance evaluation report.

## Core Features

-   **Full Video Processing**: Ingests a video file and processes it sequentially from start to finish.
-   **Per-Frame Pose Estimation**: Utilizes the MediaPipe Pose library to detect 33 body landmarks for each frame.
-   **Live Biomechanical Metrics**: Calculates key metrics in real-time, including elbow angle, spine lean, head-knee alignment, and foot direction.
-   **Visual Overlays**: Generates an output video with the pose skeleton and real-time metric readouts drawn directly onto the frames.
-   **Automated Feedback**: Provides simple, actionable cues on the video when key metric thresholds are breached (e.g., `[FIX] Excessive spine lean`).
-   **Final Shot Evaluation**: Produces a detailed `evaluation.json` file with a summary score (1-10) and written feedback across five categories.

## Advanced Features Implemented (Bonus)

-   **Automatic Phase Segmentation**: The system automatically detects the key phases of the shot (Stance → Downswing → Follow-through) by analyzing the player's wrist velocity.
-   **Contact-Moment Auto-Detection**: By tracking the peak wrist velocity during the downswing, the system identifies the precise moment of impact for a more accurate analysis.
-   **Skill Grade Prediction**: Based on the final scores, the system provides a high-level skill classification ("Beginner", "Intermediate", or "Advanced").
-   **Performance Logging**: The script measures and logs its own performance, achieving an average processing speed of ~20 FPS on a standard CPU.
-   **Configuration Driven**: All key parameters, such as file paths and metric thresholds, are externalized into a `config.json` file for easy tuning without modifying the source code.

## Demo / Output

The script produces two main output files in the `/output` directory:

-   `annotated_video.mp4`: The original video with overlays for the pose skeleton, real-time phase detection (PHASE: DOWNSWING), wrist velocity, and detailed biomechanical metrics.
-   `evaluation.json`: A JSON file containing the phase-aware final scores, feedback, and skill grade.

```json
{
    "footwork": {
        "score": 1,
        "feedback": [
            "Front foot angle could be adjusted for better shot direction."
        ]
    },
    "head_position": {
        "score": 2,
        "feedback": [
            "Work on keeping your head over the front knee at the moment of impact."
        ]
    },
    "swing_control": {
        "score": 6,
        "feedback": [
            "Good elbow elevation during the downswing, providing control and power."
        ]
    },
    "balance": {
        "score": 8,
        "feedback": [
            "Excellent balance with an upright spine during the shot."
        ]
    },
    "follow_through": {
        "score": 0,
        "feedback": [
            "Could not clearly analyze the follow-through phase."
        ]
    },
    "skill_grade": "Beginner"
}
```

## Tech Stack

-   **Python 3.10+**
-   **OpenCV-Python**: For video reading, writing, and drawing operations.
-   **MediaPipe**: For lightweight, high-fidelity pose estimation.
-   **NumPy**: For numerical operations and vector/angle calculations.

## Setup and Installation

Follow these steps to set up the project environment.

1.  **Clone the repository:**
    ```bash
    git clone Athlete-Rise---Realtime-Cricket-video-Analytics
    cd Athlete-Rise---Realtime-Cricket-video-Analytics
    ```

2.  **Create a Python virtual environment:**
    ```bash
    python -m venv venv
    ```

3.  **Activate the virtual environment:**

    *On Windows:*
    ```bash
    .\venv\Scripts\activate
    ```
    

4.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Download the Input Video:**
    Download the required video from [here](https://youtube.com/shorts/vSX3IRxGnNY) and place it in the root of the project directory. Rename the file to `input_video.mp4`.

6.  **Create Configuration File:**
    Create a file named `config.json` in the root directory and paste the following content into it. This file allows you to tune all parameters without changing the code.
    ```json
    {
        "video_paths": {
            "input": "input_video.mp4",
            "output_dir": "output/"
        },
        "pose_estimation": {
            "min_detection_confidence": 0.5,
            "min_tracking_confidence": 0.5
        },
        "thresholds": {
            "visibility": 0.6,
            "elbow_angle_good_min": 90,
            "elbow_angle_good_max": 130,
            "spine_lean_good_max": 20,
            "head_knee_dist_good_max": 50,
            "foot_angle_good_min": 70,
            "foot_angle_good_max": 110,
            "follow_through_min_extension": 150
        },
        "phase_detection": {
            "wrist_velocity_threshold_swing": 8,
            "wrist_velocity_threshold_stance": 2
        }
    }
    ```

## How to Run

With the virtual environment activated and the `input_video.mp4` and `config.json` files in place, run the main script from the terminal:

```bash
python cover_drive_analysis_realtime.py
```

## Assumptions & Limitations
-   **Player Orientation**: The analysis assumes a right-handed batsman. Therefore, the "front" arm and leg are hardcoded to be the player's left side (e.g., `LEFT_ELBOW`, `LEFT_KNEE`).
-   **Camera Angle**: The calculations are based on a 2D projection of a 3D movement. The results are sensitive to the camera angle and may not be perfectly accurate.
-   **Metric Thresholds**: The thresholds used for scoring and live feedback (e.g., ideal elbow angle, max spine lean) are empirical and based on general cricket coaching principles. They are not clinically validated and can be adjusted in the code for different scenarios.
-   **Scope**: This base version does not include bat/ball tracking or automatic shot phase segmentation (e.g., stance, downswing, impact). The analysis is performed over the entire video.
