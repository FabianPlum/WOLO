import os
import pandas as pd
from sklearn.manifold import TSNE
import plotly.express as px
import argparse

if __name__ == "__main__":
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    # Data input and output
    ap.add_argument("-p", "--perplexity", required=False, type=int, default=5)
    ap.add_argument("-i", "--input_folder", required=False, type=str, default="EXTRACTED_POSE_FEATURE_VECTORS")

    args = vars(ap.parse_args())

    perplexity = int(args["perplexity"])
    input_folder = args["input_folder"]

    out_folder = "perplexity_" + str(perplexity)
    os.mkdir(out_folder)

    df_ant_poses_list = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if
                         os.path.isfile(os.path.join(input_folder, f))]

    df_ant_poses_dfs = [pd.read_pickle(filename) for filename in df_ant_poses_list]

    # stack the DataFrames
    df_ant_poses = pd.concat(df_ant_poses_dfs, ignore_index=True, axis=0)

    print("INFO: Combined all extracted features...")

    features = df_ant_poses.loc[:, 'raw_pose_0':]  # use all entries and exclude labels
    labels = df_ant_poses.loc[:, :"size_class"]

    norm_features = (features - features.min()) / (features.max() - features.min())

    print("INFO: Normalised all extracted features (using min/max)")

    print("INFO: extracted feature vector contains", norm_features.shape[0], "instances and", norm_features.shape[1],
          "features")

    print("INFO: Performing UMAP fitting...")

    # run a 2D tSNE
    tsne = TSNE(n_components=2,
                random_state=0,
                init="pca",
                learning_rate="auto",
                perplexity=perplexity)
    projections = tsne.fit_transform(norm_features)

    print("INFO: Embedding completed!")

    projections_and_labels = labels.copy()
    projections_and_labels["x"] = projections[:, 0]
    projections_and_labels["y"] = projections[:, 1]

    projections_and_labels.to_pickle(os.path.join(out_folder, "tSNE_projection.pkl"))

    # now take the produced embedding and plot with different colour overlays to group by class, speed, or material
    fig_size = px.scatter(
        projections_and_labels, x="x", y="y",
        color="size_class", labels={'color': 'size_class'},
        opacity=0.01,
        hover_name="video", hover_data=["material", "startframe", "id", "speed", "size_class"],
        width=600, height=500)

    fig_size.write_image(os.path.join(out_folder, "tSNE_fig_size.svg"))
    fig_size.write_html(os.path.join(out_folder, "tSNE_fig_size.html"))

    fig_speed = px.scatter(
        projections_and_labels, x="x", y="y",
        color="speed", labels={'color': 'speed'},
        hover_name="video", hover_data=["material", "startframe", "id", "speed", "size_class"],
        range_color=(0, 50),
        opacity=0.01,
        width=600, height=500)

    fig_speed.write_image(os.path.join(out_folder, "tSNE_fig_speed.svg"))
    fig_speed.write_html(os.path.join(out_folder, "tSNE_fig_speed.html"))

    fig_material = px.scatter(
        projections_and_labels, x="x", y="y",
        color="material", labels={'color': 'material'},
        hover_name="video", hover_data=["material", "startframe", "id", "speed", "size_class"],
        opacity=0.01,
        width=600, height=500)

    fig_material.write_image(os.path.join(out_folder, "tSNE_fig_material.svg"))
    fig_material.write_html(os.path.join(out_folder, "tSNE_fig_material.html"))

    print("INFO: Process completed. All plots have been exported.")
