import sys
import urllib.request
import tensorflow_io as tfio
import tensorflow as tf
import tensorflow_hub as tfhub


import cameralib
import poseviz


def main():
    model = tfhub.load('metrabs_eff2l_y4_384px_800k_28ds')
    skeleton = 'smpl_24'
    joint_names = model.per_skeleton_joint_names[skeleton].numpy().astype(str)
    joint_edges = model.per_skeleton_joint_edges[skeleton].numpy()

    video_filepath = 'Basketball.mp4'
    frame_batches = tfio.IODataset.from_ffmpeg(video_filepath, 'v:0').batch(16).prefetch(1)

    camera = cameralib.Camera.from_fov(
        fov_degrees=55, imshape=frame_batches.element_spec.shape[1:3])

    with poseviz.PoseViz(joint_names, joint_edges) as viz:
        while True:
            for frame_batch in frame_batches:
                pred = model.detect_poses_batched(
                    frame_batch, intrinsic_matrix=camera.intrinsic_matrix[tf.newaxis],
                    skeleton=skeleton)

                for frame, boxes, poses in zip(frame_batch, pred['boxes'], pred['poses3d']):
                    viz.update(frame=frame, boxes=boxes, poses=poses, camera=camera)


if __name__ == '__main__':
    main()
