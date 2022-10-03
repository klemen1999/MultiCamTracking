# Multi camera calibration
This example demonstrates multiple Luxonis OAK cameras tracking objects and showing their position in bird's-eye view. 

![](img/demo.gif)
## Controls
| key 			| action
| :---			| :---			|
| `1` ... `9` 	| select camera |
| `q`			| quit 			|
| `p`			| start pose estimation |
| `d`			| toggle depth view |


## Usage
Run the [`main.py`](main.py) with Python 3.

__Measure the pose of the camera__ \
Press the `p` key to estimate the pose of the camera. An overlay showing the coordinate system will appear and the pose of the camera will be saved to a file. To dismiss the overlay press any key. \
_Repeat for every camera_. 

![pose estimation](img/pose.png)

When a camera's pose is measured, it will appear in the bird's-eye view along with its detected objects.

![bird's-eye view](img/birdseye.png)