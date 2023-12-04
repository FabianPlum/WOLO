import os
import pandas as pd
from umap import UMAP
import plotly.express as px
import argparse

if __name__ == "__main__":
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    # Data input and output
    ap.add_argument("-i", "--input_folder", required=False, type=str, default="EXTRACTED_POSE_FEATURE_VECTORS")
    ap.add_argument("-n", "--normalisation", required=False, type=str, default="min_max")

    args = vars(ap.parse_args())

    input_folder = args["input_folder"]
    normalisation = args["normalisation"]

    out_folder = "UMAP_2D_" + normalisation + "-normalisation"
    os.mkdir(out_folder)

    df_ant_poses_list = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if
                         os.path.isfile(os.path.join(input_folder, f))]

    print(df_ant_poses_list)

    df_ant_poses_dfs = [pd.read_pickle(filename) for filename in df_ant_poses_list]

    # stack the DataFrames
    df_ant_poses = pd.concat(df_ant_poses_dfs, ignore_index=True, axis=0)

    print("INFO: Combined all extracted features...")

    features = df_ant_poses.loc[:, 'raw_pose_0':]  # use all entries and exclude labels
    labels = df_ant_poses.loc[:, :"size_class"]

    if normalisation == "min_max":
        norm_features = (features - features.min()) / (features.max() - features.min())
        print("INFO: Normalised all extracted features (using min / max)")
    else:
        norm_features = (features - features.mean()) / features.std()
        print("INFO: Normalised all extracted features (using mean / std)")

    print("INFO: Normalised all extracted features (using min/max)")

    print("INFO: Extracted feature vector contains", norm_features.shape[0], "instances and", norm_features.shape[1],
          "features")

    print("INFO: Performing UMAP fitting...")

    umap_2d = UMAP(n_components=2,
                   init='pca')  # , random_state=0)
    proj_2d = umap_2d.fit_transform(norm_features)

    print("INFO: Embedding completed!")

    projections_and_labels_UMAP = labels.copy()
    projections_and_labels_UMAP["x"] = proj_2d[:, 0]
    projections_and_labels_UMAP["y"] = proj_2d[:, 1]

    projections_and_labels_UMAP.to_pickle(os.path.join(out_folder, "UMAP_df"))

    fig_size_2d = px.scatter(
        projections_and_labels_UMAP, x="x", y="y",
        color="size_class", labels={'color': 'size_class'},
        opacity=0.01,
        hover_name="video", hover_data=["material", "startframe", "id", "speed", "size_class"],
        width=600, height=500)

    fig_size_2d.write_image(os.path.join(out_folder, "UMAP_fig_size.svg"))
    fig_size_2d.write_html(os.path.join(out_folder, "UMAP_fig_size.html"))

    fig_speed_2d = px.scatter(
        projections_and_labels_UMAP, x="x", y="y",
        color="speed", labels={'color': 'speed'},
        hover_name="video", hover_data=["material", "startframe", "id", "speed", "size_class"],
        opacity=0.01,
        range_color=(0, 50),
        width=600, height=500)

    fig_speed_2d.write_image(os.path.join(out_folder, "UMAP_fig_speed.svg"))
    fig_speed_2d.write_html(os.path.join(out_folder, "UMAP_fig_speed.html"))

    fig_material_2d = px.scatter(
        projections_and_labels_UMAP, x="x", y="y",
        color="material", labels={'color': 'material'},
        hover_name="video", hover_data=["material", "startframe", "id", "speed", "size_class"],
        opacity=0.01,
        width=600, height=500)

    fig_material_2d.write_image(os.path.join(out_folder, "UMAP_fig_material.svg"))
    fig_material_2d.write_html(os.path.join(out_folder, "UMAP_fig_material.html"))

    print("INFO: Process completed. All plots have been exported.")
