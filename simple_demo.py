import poseviz
import tensorflow as tf
import numpy as np
import cameralib


def main():
    # Names of the body joints. Left joint names must start with 'l', right with 'r'.
    image = tf.image.decode_jpeg(tf.io.read_file('2024-01-25-094456.jpg'))
    joint_names = ['pelv', 'lhip', 'rhip', 'spi1', 'lkne', 'rkne', 'spi2', 'lank', 'rank', 'spi3', 'ltoe', 'rtoe',
                   'neck', 'lcla', 'rcla', 'head', 'lsho', 'rsho', 'lelb', 'relb', 'lwri', 'rwri', 'lhan', 'rhan']

    # Joint index pairs specifying which ones should be connected with a line (i.e., the bones of
    # the body, e.g. wrist-elbow, elbow-shoulder)
    joint_edges = [[1, 4],
                   [1, 0],
                   [2, 5],
                   [2, 0],
                   [3, 6],
                   [3, 0],
                   [4, 7],
                   [5, 8],
                   [6, 9],
                   [7, 10],
                   [8, 11],
                   [9, 12],
                   [9, 13],
                   [9, 14],
                   [12, 15],
                   [13, 16],
                   [14, 17],
                   [16, 18],
                   [17, 19],
                   [18, 20],
                   [19, 21],
                   [20, 22],
                   [21, 23]]

    poses = [[[-713.68463, -30.783072, 3753.6438],
              [-655.8688, 62.843517, 3776.9949],
              [-779.35126, 58.8573, 3753.2058],
              [-718.47986, -148.70981, 3782.5464],
              [-609.9741, 426.16083, 3817.8042],
              [-859.98065, 419.23956, 3762.6577],
              [-713.8953, -292.91534, 3766.7026],
              [-652.84973, 824.0831, 3870.423],
              [-860.6004, 818.9723, 3822.6067],
              [-708.6673, -349.73697, 3750.0234],
              [-562.90533, 887.1, 3771.647],
              [-930.159, 877.9912, 3704.1616],
              [-701.197, -569.85693, 3776.0142],
              [-628.3275, -464.6666, 3787.19],
              [-783.8854, -473.26962, 3759.3914],
              [-688.6038, -647.22235, 3716.565],
              [-513.4102, -460.12207, 3814.541],
              [-902.23035, -483.78262, 3748.5542],
              [-424.46988, -233.27115, 3817.8367],
              [-995.72217, -260.05145, 3744.6614],
              [-288.8937, -35.436104, 3710.9114],
              [-1105.2048, -50.187305, 3629.9167],
              [-248.02783, 42.06832, 3695.0813],
              [-1157.6713, 17.41977, 3604.6445]]]
    viz = poseviz.PoseViz(joint_names, joint_edges)
    # Get the current frame
    frame = np.zeros(image.shape, np.uint8)
    # Iterate over the frames of e.g. a video
    for i in range(1):
        # Make predictions here
        # ...

        # Update the visualization
        viz.update(
            frame=frame,
            boxes=np.array([[0, 0, 0, 0]], np.float32),
            poses=np.array(poses, np.float32),
            camera=cameralib.Camera.from_fov(55, frame.shape[:2]))


if __name__ == '__main__':
    main()
