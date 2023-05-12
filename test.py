import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from model import BlazePose
from skimage.measure import regionprops

from config import total_epoch, train_mode, eval_mode, epoch_to_test
from data import test_dataset, label, data


def Eclidian2(a, b):
    # Calculate the square of Eclidian distance
    assert len(a) == len(b)
    summer = 0
    for i in range(len(a)):
        summer += (a[i] - b[i]) ** 2
    return summer


model = BlazePose()
# optimizer = tf.keras.optimizers.Adam()
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=[tf.keras.metrics.MeanSquaredError()],
)

checkpoint_path = "checkpoints/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# model.evaluate(test_dataset)
model.load_weights(checkpoint_path.format(epoch=epoch_to_test))

"""
0-Right ankle
1-Right knee
2-Right hip
3-Left hip
4-Left knee
5-Left ankle
6-Right wrist
7-Right elbow
8-Right shoulder
9-Left shoulder
10-Left elbow
11-Left wrist
12-Neck
13-Head top
"""

if train_mode:
    y = np.zeros((2000, 14, 3)).astype(np.uint8)
    if 1:  # for low profile GPU
        batch_size = 20
        for i in range(0, 2000, batch_size):
            if i + batch_size >= 2000:
                # last batch
                y[i:2000] = model(data[i : i + batch_size]).numpy()  # .astype(np.uint8)
            else:
                # other batches
                y[i : i + batch_size] = model(
                    data[i : i + batch_size]
                ).numpy()  # .astype(np.uint8)
                print("=", end="")
        print(">")
    else:  # for RTX 3090
        print("Start inference.")
        y[0:1000] = model(data[0:1000]).numpy()  # .astype(np.uint8)
        print("Half.")
        y[1000:2000] = model(data[1000:2000]).numpy()  # .astype(np.uint8)
        print("Complete.")

    if eval_mode:
        # calculate pckh score
        # print(label.shape)  # (2000, 14, 3)
        y = y[:, :, 0:2].astype(float)
        label = label[:, :, 0:2].astype(float)
        score_j = np.zeros(14)
        pck_metric = 0.5
        for i in range(1000, 2000):
            # validation part
            pck_h = Eclidian2(label[i][12], label[i][13])
            for j in range(14):
                pck_j = Eclidian2(y[i][j], label[i][j])
                # pck_j <= pck_h * 0.5 --> True
                if pck_j <= pck_h * pck_metric:
                    # True estimation
                    score_j[j] += 1
        # convert to percentage
        score_j = score_j * 0.1
        score_avg = sum(score_j) / 14
        print(score_j)
        print("Average = %f%%" % score_avg)
    else:
        # show result images
        import cv2

        # generate result images
        for t in range(2000):
            skeleton = y[t]
            print(skeleton)
            img = data[t].astype(np.uint8)
            # draw the joints
            for i in range(14):
                cv2.circle(
                    img,
                    center=tuple(skeleton[i][0:2]),
                    radius=2,
                    color=(0, 255, 0),
                    thickness=2,
                )
            # draw the lines
            for j in (
                (13, 12),
                (12, 8),
                (12, 9),
                (8, 7),
                (7, 6),
                (9, 10),
                (10, 11),
                (2, 3),
                (2, 1),
                (1, 0),
                (3, 4),
                (4, 5),
            ):
                cv2.line(
                    img,
                    tuple(skeleton[j[0]][0:2]),
                    tuple(skeleton[j[1]][0:2]),
                    color=(0, 0, 255),
                    thickness=1,
                )
            # solve the mid point of the hips
            cv2.line(
                img,
                tuple(skeleton[12][0:2]),
                tuple(skeleton[2][0:2] // 2 + skeleton[3][0:2] // 2),
                color=(0, 0, 255),
                thickness=1,
            )

            cv2.imwrite("./result/lsp_%d.jpg" % t, img)
            cv2.imshow("test", img)
            cv2.waitKey(1)
else:
    # visualize the dataset
    model.evaluate(test_dataset)

    number_of_picture = 10

    # select an image to visualize
    from config import vis_img_id

    y = model.predict(data[vis_img_id : vis_img_id + number_of_picture])

    title_set = [
        "Right ankle",
        "Right knee",
        "Right hip",
        "Left hip",
        "Left knee",
        "Left ankle",
        "Right wrist",
        "Right elbow",
        "Right shoulder",
        "Left shoulder",
        "Left elbow",
        "Left wrist",
        "Neck",
        "Head top",
    ]

    from scipy.ndimage import label

    temp = np.array([])
    # loop through pictures
    for t in range(number_of_picture):
        points = {}  # dictionary body parts' coordinate
        plt.figure(figsize=(15, 7), dpi=150)
        for i in range(14):
            # get heatmap
            heatmap = y[t, :, :, i]

            # masked heatmap
            thresh = 0.007
            mask = np.array(heatmap)
            mask[mask < thresh] = 0

            # find clusters on heatmap
            labeled_array, num_features = label(
                mask
            )  # clustered masks, number of clusters
            cc = regionprops(labeled_array)  # object of clustered masks
            face_location = {
                c.num_pixels: c.centroid for c in cc
            }  # size of cluster: coordinates of center of cluster
            coords = [round(i) for i in face_location[max(face_location.keys())]]
            coords.append(heatmap[coords[0], coords[1]])
            points[title_set[i]] = coords
            if i == 0:
                np_coords = np.array([coords], ndmin=2)
            else:
                np_coords = np.append(np_coords, np.array([coords], ndmin=2), axis=0)

            # plot images
            plt.subplot(3, 6, i + 1)
            plt.imshow(heatmap)
            plt.title(title_set[i])

        plt.subplot(3, 6, 15)
        plt.imshow(data[vis_img_id + t].astype(np.uint8))
        plt.subplot(3, 6, 16, projection="3d")
        plt.scatter(
            np_coords[:, 2], np_coords[:, 0], np_coords[:, 1], marker="o", c="red"
        )
        # plt.savefig("demo.png")
        plt.show()
pass
