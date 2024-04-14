import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.colors as mcolors
import numpy as np
from scipy.interpolate import make_interp_spline
import seaborn as sns

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

sensors = ["back_x", "back_y", "back_z", "thigh_x", "thigh_y", "thigh_z"]

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

    # Plot
    fig, ax = plt.subplots()
    durations.plot(kind="bar", ax=ax, color="skyblue")
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

    # Downsampling: Sample at 1 second intervals (50 data points)
    sampled_data = activity_data.iloc[::50]

    # Ensure we have 30 seconds of data (30 points after downsampling)
    sampled_data = sampled_data.head(30)

    # Creating a larger figure to accommodate the correlation heatmap
    fig, axes = plt.subplots(
        nrows=4, ncols=2, figsize=(12, 14)
    )  # Adjusted the number of rows to 4
    fig.suptitle(
        f"Sensor Readings and Correlation Matrix for Activity {ACTIVITIES_NAME_DICT.get(activity)}. Participant: {participant}",
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

    # Plot sensor data in the first six subplots
    for i, ax in enumerate(axes.flat[:6]):
        ax.plot(
            activity_data["timestamp"],
            activity_data[sensors[i]],
            label=f"{titles[i]} readings",
            marker="o",
            linestyle="-",
        )
        sensor_data = activity_data[sensors[i]]
        mean_value = sensor_data.mean()
        ax.set_title(titles[i])
        ax.set_xlabel("Time")
        ax.set_ylabel("Acceleration")
        # Annotate the mean value on the plot
        ax.axhline(
            y=mean_value, color="r", linestyle="--", label=f"Mean: {mean_value:.2f}"
        )
        ax.legend()

        plt.setp(ax.get_xticklabels(), rotation=45)
        ax.grid(True)

    # Calculate the correlation matrix and plot it in the last subplot
    correlation_matrix = activity_data[sensors].corr()
    ax_correlation = axes.flat[6]
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", ax=ax_correlation)
    ax_correlation.set_title("Sensor Data Correlation Matrix")

    ax_all_sensors = axes.flat[7]
    for sensor in sensors:
        ax_all_sensors.plot(
            sampled_data["timestamp"],
            sampled_data[sensor],
            label=f"{sensor}",
            marker="o",
            linestyle="-",
        )
    ax_all_sensors.set_title("All Sensors Over Time")
    ax_all_sensors.legend()
    ax_all_sensors.set_xlabel("Time")
    ax_all_sensors.set_ylabel("Acceleration")
    ax_all_sensors.grid(True)

    # Adjust layout to prevent overlap and ensure all subplot titles and labels are visible
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
        durations[participant] = (
            activity_durations / 60
        )  # Convert each duration from seconds to minutes

    return durations


def plot_activity_duration_heatmap(durations):
    duration_df = pd.DataFrame(durations).fillna(0)

    plt.figure(figsize=(10, 6))
    sns.heatmap(duration_df, annot=True, cmap="YlGnBu", fmt=".1f")
    plt.title("Activity Duration Heatmap Across Participants (minutes)")
    plt.ylabel("Activity")
    plt.xlabel("Participant ID")
    plt.show()


def plot_activity_histogram(activity_name, durations_dict, blockPlot=True):
    # Extract durations for the specified activity across all participants
    activity_durations = []
    participant_ids = []

    for participant, activities in durations_dict.items():
        # Convert seconds to minutes and append the duration
        duration = activities.get(
            activity_name, 0
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


def calculate_mean_for_activity(activity, participant):
    # Load the data
    data = pd.read_csv(PARTICIPANT_DATA_DICT.get(participant))
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    activity_data = data[data["label"] == activity]

    # Define the sensor titles (assuming these are the column names in your data)
    sensors = ["back_x", "back_y", "back_z", "thigh_x", "thigh_y", "thigh_z"]

    # Dictionary to store mean values of each sensor
    sensor_mean_values = {}

    # Calculate the mean for each sensor
    for sensor in sensors:
        sensor_mean_values[sensor] = activity_data[sensor].mean()

    return sensor_mean_values

def analyze_activities(activities, participants):
    # Dictionary to store DataFrame rows
    all_means = []

    # Loop through each participant and activity to calculate means
    for participant in participants:
        for activity in activities:
            mean_values = calculate_mean_for_activity(activity, participant)
            mean_values['Activity'] = activity  # Add activity label for identification
            mean_values['Participant'] = participant  # Add participant ID
            all_means.append(mean_values)
    
    # Create a DataFrame from the collected mean values
    df_means = pd.DataFrame(all_means)
    df_pivot = df_means.pivot_table(index=['Activity'], columns=['Participant'], values=sensors)
    
    # Calculate the correlation matrix
    correlation_matrix = df_pivot.corr()

    # Create the heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix of Activity Sensor Means')
    plt.show()

if __name__ == "__main__":
    participants = [i for i in range(1, 5)]
    # All participants
    participants = [participant for participant in PARTICIPANT_DATA_DICT.keys()]

    # plot_activity_changes(participants)

    durations = gather_all_durations(participants)
    # activity_names = list(ACTIVITIES_NAME_DICT.values())
    # for activity_name in activity_names[:-1]:
    #     plot_activity_histogram(activity_name, durations, False)
    # plot_activity_histogram(activity_names[-1], durations)
    # plot_activity_duration_heatmap(durations)

    # Get the participant with the most time per activity
    top_participant_per_activity = find_top_participant_per_activity(durations)

    # Plot the activity sensor data for the top participant of each activity.

    activity_participant_list = list(top_participant_per_activity.items())

    for key, value in activity_participant_list[:-2]:
        plot_activity(INVERSE_ACTIVITIES_NAME_DICT[key], value, False)
    plot_activity(
        INVERSE_ACTIVITIES_NAME_DICT[activity_participant_list[-1][0]],
        activity_participant_list[-1][1],
        True,
    )
