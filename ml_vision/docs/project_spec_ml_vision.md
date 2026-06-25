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
- ball position
- ball size
- camera position
- camera angle
- platform position
- platform angle
- platform size
- ball position
- ball size

output vector:

- (x,y) position of the ball on the plane.

potential outputs which may be useful in the future.
axis of the plane in xyz.

FUTURE IMPLEMENTATION.
WE ARE NOT DOING THIS NOW.
will also need to be able to identify markers on the plane. 
the final goal is to move the ball between markers based on a command from the microcontroller.