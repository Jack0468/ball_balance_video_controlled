## vision model
Design a vision model to determine the coordinates of a ball on a platform.
The robots actuators are a 3-degree-of-freedom parallel manipulator.

https://github.com/skulkarni3000/ball-balancing-bot

We aim to supplement the resistive touchpad with a computer vision model.

these are my initial notes.

SIGNAL FLOW
Camera -> microprocessor -> python -> microprocessor -> actuators

ML SIGNAL FLOW.


Identify ball in the frame. 
Identify the platform with respect to the background of the plane (use open cv to remove background / determine ball)
Identify the coloured dots / markers in the frame.
Identify centre of the ball with respect to the plane of the board.
Send that position to the inverse kinematics math.

The first sections can be determined easily using openCV or other multipurpose opensource models.
The bold section is our focus for this section.

Initial benchmarking.
Use different open source ML models to determine best model for our application.
we want to identify the center of the ball (x,y) with respect to the plane of the platform.

Initally we have a video which we can break up into individual frames. The data is currently unlabelled, but we will collect a comprehensive dataset in the future. 
> [!NOTE]
> All future data collection and processing MUST adhere to the Medallion Architecture (Bronze/Silver/Gold) as outlined in the core `ENGINEERING_STANDARDS.md`.

We will need preprocessing pipelines for this data.

normalisation features to consider:
- exposure
- lighting
- camera position
- camera angle
- platform position
- platform angle
- platform size

ball properties for detection:
- Spherical and highly reflective surface.
- Dynamic (in continuous motion during balancing).
- Must be uniquely distinguished from the static, painted target markers on the platform.

output vector:

- (x,y) position of the ball on the plane.

potential outputs which may be useful in the future.
axis of the plane in xyz.

## Target Marker Detection
Alongside tracking the ball, the vision model must identify specific colored markers on the platform in real-time. 
- **Marker Colors**: Blue, Grey, Black, Red.
- **Purpose**: These markers represent target destinations for the ball. They can be placed anywhere on the platform plane.
- **Output**: The system must extract the `(x,y)` centroid of each marker relative to the normalized 2D plane of the platform.
- **Integration**: These `(x,y)` coordinates will be sent as target destinations to the inverse kinematics math, directing the robot to tilt the platform and roll the ball to the selected marker.