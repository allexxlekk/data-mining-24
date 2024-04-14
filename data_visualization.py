import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.colors as mcolors
import numpy as np
from scipy.interpolate import make_interp_spline
import seaborn as sns
from matplotlib.ticker import PercentFormatter
from sklearn.metrics import confusion_matrix

ACTIVITIES_NAME_DICT = {
    1: "Walking",
    2: "Running",
    3: "Shuffling",
    4: "Stairs (ascending)",
    5: "Stairs (descending)",
    6: "Standing",
    7: "Sitting",
    8: "Lying",
    13: "Cycling (sit)",
    14: "Cycling (stand)",
    130: "Cycling (sit, inactive)",
    140: "Cycling (stand, inactive)",
}
INVERSE_ACTIVITIES_NAME_DICT = {v: k for k, v in ACTIVITIES_NAME_DICT.items()}


ACTIVITIES_VALUE_DICT = {
    1: 0,
    2: 1,
    3: 2,
    4: 3,
    5: 4,
    6: 5,
    7: 6,
    8: 7,
    13: 8,
    14: 9,
    130: 10,
    140: 11,
}

PARTICIPANT_DATA_DICT = {
    1: "harth/S006.csv",
    2: "harth/S008.csv",
    3: "harth/S009.csv",
    4: "harth/S010.csv",
    5: "harth/S012.csv",
    6: "harth/S013.csv",
    7: "harth/S014.csv",
    8: "harth/S015.csv",
    9: "harth/S016.csv",
    10: "harth/S017.csv",
    11: "harth/S018.csv",
    12: "harth/S019.csv",
    13: "harth/S020.csv",
    14: "harth/S021.csv",
    15: "harth/S022.csv",
    16: "harth/S023.csv",
    17: "harth/S024.csv",
    18: "harth/S025.csv",
    19: "harth/S026.csv",
    20: "harth/S027.csv",
    21: "harth/S028.csv",
    22: "harth/S029.csv",
}


def calculate_activity_durations(participant):
    data = pd.read_csv(PARTICIPANT_DATA_DICT.get(participant))
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data["activity_code"] = data["label"].map(ACTIVITIES_VALUE_DICT)

    # Calculate the time difference between consecutive records
    data["time_diff"] = data["timestamp"].diff().dt.total_seconds().fillna(0)

    # Identify changes in activity to segment the data
    data["activity_change"] = data["activity_code"].diff().ne(0)

    # Start of each activity block
    data["activity_block"] = data["activity_change"].cumsum()

    # Sum up the durations for each activity block
    activity_durations = (
        data.groupby(["activity_block", "activity_code"])["time_diff"]
        .sum()
        .reset_index()
    )

    # Map activity codes back to activity names
    # Inverse the ACTIVITIES_VALUE_DICT to revert the codes
    INVERSE_ACTIVITIES_VALUE_DICT = {v: k for k, v in ACTIVITIES_VALUE_DICT.items()}

    # Revert the activity codes to their original form
    activity_durations["original_activity_code"] = activity_durations[
        "activity_code"
    ].map(INVERSE_ACTIVITIES_VALUE_DICT)

    # Now apply the ACTIVITIES_NAME_DICT to the original activity codes
    activity_durations["activity_name"] = activity_durations[
        "original_activity_code"
    ].map(ACTIVITIES_NAME_DICT)

    # Aggregate durations by activity
    total_durations = activity_durations.groupby("activity_name")["time_diff"].sum()

    return total_durations


def plot_activity_durations(durations):
    # Convert seconds to minutes
    durations_minutes = durations / 60

    # Plot
    fig, ax = plt.subplots()
    durations_minutes.plot(kind="bar", ax=ax, color="skyblue")
    ax.set_title("Total Time Spent on Each Activity (in minutes)")
    ax.set_ylabel("Duration (minutes)")
    ax.set_xlabel("Activity")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def plot_activity(activity, participant, blockPlot=True):
    data = pd.read_csv(PARTICIPANT_DATA_DICT.get(participant))
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data["activity_code"] = data["label"].map(ACTIVITIES_VALUE_DICT)
    activity_data = data[data["label"] == activity]

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 10))
    fig.suptitle(
        f"Sensor Readings for Activity {ACTIVITIES_NAME_DICT.get(activity)}",
        fontsize=16,
    )

    sensors = ["back_x", "back_y", "back_z", "thigh_x", "thigh_y", "thigh_z"]
    titles = [
        "Back X-axis",
        "Back Y-axis",
        "Back Z-axis",
        "Thigh X-axis",
        "Thigh Y-axis",
        "Thigh Z-axis",
    ]
    for i, ax in enumerate(axes.flat):
        ax.plot(
            activity_data["timestamp"],
            activity_data[sensors[i]],
            label=f"{titles[i]} readings",
            marker="o",
            linestyle="-",
        )
        ax.set_title(titles[i])
        ax.set_xlabel("Time")
        ax.set_ylabel("Acceleration")
        ax.legend()
        plt.setp(ax.get_xticklabels(), rotation=45)
        ax.grid(True)  # Adding grid for better readability

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show(block=blockPlot)


def plot_activity_change_for_participant(participant_number, blockPlot=True):
    data = pd.read_csv(PARTICIPANT_DATA_DICT.get(participant_number))
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data["activity_code"] = data["label"].map(ACTIVITIES_VALUE_DICT)
    fig, ax = plt.subplots(figsize=(15, 8))
    cmap = plt.get_cmap("tab20", len(ACTIVITIES_VALUE_DICT))
    norm = mcolors.BoundaryNorm(range(len(ACTIVITIES_VALUE_DICT) + 1), cmap.N)

    sc = ax.scatter(
        data["timestamp"],
        data["activity_code"],
        c=data["activity_code"],
        s=10,
        cmap=cmap,
        norm=norm,
    )
    plt.yticks(
        list(ACTIVITIES_VALUE_DICT.values()), list(ACTIVITIES_NAME_DICT.values())
    )
    plt.colorbar(sc, ticks=range(len(ACTIVITIES_VALUE_DICT)), label="Activities")

    ax.set_title(f"Activity Changes Over Time for participant: {participant_number}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Activity")

    plt.grid(True)
    plt.show(block=blockPlot)


def plot_activity_changes(participants):
    for participant in participants[0:-1]:
        plot_activity_change_for_participant(participant, False)
    plot_activity_change_for_participant(participants[-1])


def gather_all_durations(participants):
    durations = {}
    for participant in participants:
        activity_durations = calculate_activity_durations(participant)
        durations[participant] = activity_durations
    return durations


def plot_activity_histogram(activity_name, durations_dict, blockPlot=True):
    # Extract durations for the specified activity across all participants
    activity_durations = []
    participant_ids = []

    for participant, activities in durations_dict.items():
        # Convert seconds to minutes and append the duration
        duration = (
            activities.get(activity_name, 0) / 60
        )  # Default to 0 if the activity is not found
        activity_durations.append(duration)
        participant_ids.append(participant)

    # Convert to numpy array for mathematical operations
    durations_array = np.array(activity_durations)
    mean_duration = np.mean(durations_array)
    std_deviation = np.std(durations_array)

    # Create a histogram/bar plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(
        participant_ids, activity_durations, color="skyblue", label="Duration"
    )

    # Error bar for standard deviation and mean line
    plt.errorbar(
        participant_ids,
        activity_durations,
        yerr=std_deviation,
        fmt="o",
        color="red",
        label=f"Standard Deviation: {std_deviation:.2f}",
    )
    plt.axhline(
        y=mean_duration,
        color="green",
        linestyle="--",
        label=f"Mean Duration: {mean_duration:.2f} min",
    )

    # Optional: Smooth curve over the bars
    # Generate more x values for smoothness
    x_smooth = np.linspace(min(participant_ids), max(participant_ids), 300)
    spl = make_interp_spline(participant_ids, activity_durations, k=3)  # BSpline object
    y_smooth = spl(x_smooth)

    plt.plot(x_smooth, y_smooth, color="magenta")

    plt.xlabel("Participant")
    plt.ylabel("Duration in Minutes")
    plt.title(f"Time Spent on {activity_name} by Each Participant")
    plt.xticks(
        participant_ids, [f"{pid}" for pid in participant_ids]
    )  # Customize tick labels
    plt.grid(True)
    plt.legend()
    plt.show(block=blockPlot)


def find_top_participant_per_activity(durations_dict):
    top_participant_per_activity = {}

    activity_participants = {}

    for participant, activities in durations_dict.items():
        for activity_name, duration in activities.items():
            if activity_name not in activity_participants:
                activity_participants[activity_name] = {}
            activity_participants[activity_name][participant] = duration

    for activity_name, participants_durations in activity_participants.items():
        # Find the participant with the maximum duration for the current activity
        top_participant = max(participants_durations, key=participants_durations.get)
        top_participant_per_activity[activity_name] = top_participant

    return top_participant_per_activity


def plotHistory(history) -> None:
    fg = plt.figure()
    # plot loss during training
    plt.subplot(211)
    plt.title("Loss (Categorical Focal Crossentropy)")
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="test")
    plt.legend()
    # plot accuracy during training
    plt.subplot(212)
    plt.title("Accuracy")
    plt.plot(history.history["accuracy"], label="train")
    plt.plot(history.history["val_accuracy"], label="test")
    plt.legend()
    plt.show()


def displayConfusionMatrix(
    y_true,
    y_pred,
    labels,
    classes,
    filename="Screenshots/Sequential models testing/Confusion Matrix.png",
    ymap=None,
    figsize=(17, 17),
):
    """
    Downloaded from Github Gist: mesquita/nice_cm.py

    Generate matrix plot of confusion matrix with pretty annotations.
    The plot image is saved to disk.
    args:
      y_true:    true label of the data, with shape (nsamples,)
      y_pred:    prediction of the data, with shape (nsamples,)
      filename:  filename of figure file to save
      labels:    string array, name the order of class labels in the confusion matrix.
                 use `clf.classes_` if using scikit-learn models.
                 with shape (nclass,).
      classes:   aliases for the labels. String array to be shown in the cm plot.
      ymap:      dict: any -> string, length == nclass.
                 if not None, map the labels & ys to more understandable strings.
                 Caution: original y_true, y_pred and labels must align.
      figsize:   the size of the figure plotted.
    """
    # sns.set_theme(font_scale=2.8)

    if ymap is not None:
        y_pred = [ymap[yi] for yi in y_pred]
        y_true = [ymap[yi] for yi in y_true]
        labels = [ymap[yi] for yi in labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = "%.2f%%\n%d/%d" % (p, c, s)
            # elif c == 0:
            #    annot[i, j] = ''
            else:
                annot[i, j] = "%.2f%%\n%d" % (p, c)
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm = cm * 100
    cm.index.name = "True Label"
    cm.columns.name = "Predicted Label"
    fig, ax = plt.subplots(figsize=figsize)
    plt.yticks(va="center")

    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        ax=ax,
        xticklabels=classes,
        cbar=True,
        cbar_kws={"format": PercentFormatter()},
        yticklabels=classes,
        cmap="Blues",
    )
    plt.savefig(filename, bbox_inches="tight")


def plotHistogram(series: pd.Series, ax):
    # Calculate total sum of counts
    total_count = series.sum()

    # Plotting the bar plot
    bars = series.plot(
        kind="bar", color="skyblue", ax=ax
    )  # Use ax parameter to specify the subplot
    ax.set_xlabel("Activity")  # Set x-axis label for the subplot
    ax.set_ylabel("Count")  # Set y-axis label for the subplot
    ax.set_xticklabels(
        series.index, rotation=45, ha="right"
    )  # Set x-axis labels and rotate for readability

    # Annotate each bar with its percentage value
    for bar in bars.patches:
        count = bar.get_height()
        percentage = (count / total_count) * 100
        ax.annotate(
            f"{percentage:.2f}%",
            xy=(bar.get_x() + bar.get_width() / 2, count),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )


def plotLabelDistributionHistogram(data, label_columns, label_list) -> None:
    # Print the class distribution (%) of the train and test datasets
    column_sums_train = data["train_df"][label_columns].sum()
    column_sums_train.index = label_list

    column_sums_test = data["test_df"][label_columns].sum()
    column_sums_test.index = label_list

    # Set up the figure and subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    plotHistogram(column_sums_train, ax1)
    ax1.set_title("Train set label distribution")

    plotHistogram(column_sums_test, ax2)
    ax2.set_title("Test set label distribution")

    plt.tight_layout()
    plt.show()


# if __name__ == "__main__":
#     participants = [i for i in range(1, 5)]

#     # All participants
#     participants = [participant for participant in PARTICIPANT_DATA_DICT.keys()]
#     plot_activity_changes(participants)
#     durations = gather_all_durations(participants)

#     activity_names = list(ACTIVITIES_NAME_DICT.values())
#     for activity_name in activity_names[:-1]:
#         plot_activity_histogram(activity_name, durations, False)
#     plot_activity_histogram(activity_names[-1], durations)

#     # Get the participant with the most time per activity
#     top_participant_per_activity = find_top_participant_per_activity(durations)

#     # Plot the activity sensor data for the top participant

#     activity_participant_list = list(top_participant_per_activity.items())
#     for key,value in activity_participant_list[:-1]:
#         plot_activity(INVERSE_ACTIVITIES_NAME_DICT[key], value, False)

#     plot_activity(activity_participant_list[-1][0], activity_participant_list[-1][1])
