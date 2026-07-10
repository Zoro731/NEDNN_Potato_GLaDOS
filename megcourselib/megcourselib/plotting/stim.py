import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from PIL import Image

from megcourselib.constants import SCREEN_H, SCREEN_W


def plot_scene_fixations(
    scene_id: int,
    img: Image.Image,
    df: pd.DataFrame,
):
    """Generate and save a fixation scatter plot for a single scene."""
    im_w, im_h = img.size
    # assert sceneID is int
    scene_id = int(scene_id)
    # Convert screen-pixel coordinates to image-pixel coordinates

    x_offset = (SCREEN_W - im_w) / 2
    y_offset = (SCREEN_H - im_h) / 2
    x = df["mean_gx"] - x_offset
    y = df["mean_gy"] - y_offset
    # flip y to match image coordinates (y increases downward)
    y = im_h - y
    # Dot size proportional to duration, clipped to reasonable range
    sizes = (df["duration"] * 400).clip(20, 600)

    fig = plt.figure(figsize=(8, 6))
    plt.imshow(img)

    sc = plt.scatter(
        x,
        y,
        s=sizes,
        c=df["time_in_trial"],
        cmap="magma",
        alpha=0.75,
        edgecolors="white",
    )
    plt.colorbar(sc, label="time in trial [s]")

    # remove x/y ticks and set limits to image size
    plt.xticks([])
    plt.yticks([])
    plt.xlim(0, im_w)
    plt.ylim(im_h, 0)  # y increases downward, consistent with imshow

    # despine
    sns.despine(left=True, bottom=True)

    plt.tight_layout()
    return fig
