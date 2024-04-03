# %%
import os 
import pickle
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np
import random

# %%
#For DEXTER: provide the path to the directory containing y_scores.pkl and y_true/pkl, e.g: experiment_results/2023_12_02_12_00_00, which will be appended with the method and env
TARGET_DIR = " "

#For PEDM or alternative model
TARGET_DIR_PEDM = " "

methods = ["PEDM_Detector", "DEXTER_Detector"]

#Specify the environments to evaluate on, e.g:
#envs = ["IMANOCartpoleEnv-v0", "IMANSCartpoleEnv-v0",  "MJCartpole-v0"]
envs = [" ", " ",  " "]

# %%
def eval_metrics(scores, anom_occurrence, axleft, axright):

    fpr, tpr, _thresholds = metrics.roc_curve(anom_occurrence, scores)
    auroc = metrics.auc(fpr, tpr)

    axleft.plot(fpr, tpr, color='darkorange', lw=2)
    axleft.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axleft.set_xlim([0.0, 1.0])
    axleft.set_ylim([0.0, 1.05])
    axleft.set_xlabel('False Positive Rate')
    axleft.set_ylabel('True Positive Rate')
    axleft.set_title(f'(AUROC: {auroc:.2f}) (samples = {len(scores)})')

    anomaly_scores = [score for score, anomaly in zip(scores, anom_occurrence) if anomaly == 1]
    non_anomaly_scores = [score for score, anomaly in zip(scores, anom_occurrence) if anomaly == 0]
    
    min_score = min(scores)
    max_score = max(scores)
    
    bins = np.linspace(min_score, max_score, 50)

    axright.hist(anomaly_scores, bins=bins, alpha=0.5, label='Attacked', color='r')
    axright.hist(non_anomaly_scores, bins=bins, alpha=0.5, label='Unattacked', color='b')
    axright.set_xlabel('Score (log scale)')
    axright.set_ylabel('Frequency')
    axright.set_title(f'Anomaly Scores')
    axright.legend(loc='upper right')

# %%
restuls_by_exp_id = {}

for method in methods:
    for env in envs:
    
        base_dir = os.path.join(TARGET_DIR, method, env) if "MJ" not in env else os.path.join(TARGET_DIR_PEDM, method, env)
        scenarios = os.listdir(base_dir)
        scenarios = [s for s in scenarios if ".txt" not in s and ".DS_Store" not in s]

        print(scenarios)

        for scenario in scenarios:

            data_base_identifier = os.path.join(base_dir, scenario, "100_ep")

            fname_scores = os.path.join(data_base_identifier, "y_scores.pkl")
            fname_labels = os.path.join(data_base_identifier, "y_true.pkl")

            with open(fname_scores, "rb") as f:
                scores = pickle.load(f)
            with open(fname_labels, "rb") as f:
                labels = pickle.load(f)    
            print(f"loaded data for {data_base_identifier}")

            experiment_id = f"{env}_{scenario}"

            if experiment_id not in restuls_by_exp_id.keys():
                restuls_by_exp_id[experiment_id] = {method: {"scores": scores, "labels": labels}}
            else:
                restuls_by_exp_id[experiment_id][method] = {"scores": scores, "labels": labels}



# %%
def compute_cusum_scores(values, mean):
    scores = [0]
    for value in values:
        scores.append(max(scores[-1] + value - mean, 0))
    return scores

# %%
ep_length = 200
n_episodes = 1000
percentile_cutoff = 0.99

def get_detector_thresh(unattacked_scores):
        mean_unattacked_score = np.mean(unattacked_scores)
        unattacked_episodes = [random.choices(unattacked_scores, k=ep_length) for _ in range(n_episodes)]
        cusum_scores = [compute_cusum_scores(ep, mean_unattacked_score) for ep in unattacked_episodes]
        cusum_max_scores = [max(ep) for ep in cusum_scores]
        detector_thresh = np.sort(cusum_max_scores)[int(n_episodes*percentile_cutoff)]

        return mean_unattacked_score, detector_thresh

# %%
n_exps = len(restuls_by_exp_id.keys())
n_methods = len(methods)


# %%
def get_time_of_first_detection(unattacked_scores, attacked_scores):

    mean_unattacked_score, detector_thresh = get_detector_thresh(unattacked_scores)
        
    # make cusum calculations
    attacked_episodes = [random.choices(attacked_scores, k=ep_length) for _ in range(n_episodes)]
    cusum_scores = [compute_cusum_scores(ep, mean_unattacked_score) for ep in attacked_episodes]
    
    return np.mean([np.argwhere(np.array(ep) > detector_thresh)[0][0] 
                    if np.any(np.array(ep) > detector_thresh) else len(ep) 
                    for ep in cusum_scores])


# %%
fig, axs = plt.subplots(n_exps, n_methods*2, figsize=(20*n_methods, 10*n_exps))

for exp_n, experiment_id in enumerate(list(restuls_by_exp_id.keys())):

    time_of_first_detection = [None, None]
    
    for i, method in enumerate(methods):

        scores = restuls_by_exp_id[experiment_id][method]["scores"]
        labels = restuls_by_exp_id[experiment_id][method]["labels"]
        
        attacked_scores = [score for score, anomaly in zip(scores, labels) if anomaly == 1]
        unattacked_scores = [score for score, anomaly in zip(scores, labels) if anomaly == 0]

        neg_attacked_scores = [-1*score for score in attacked_scores]
        neg_unattacked_scores = [-1*score for score in unattacked_scores]

        time_of_first_detection_normal = get_time_of_first_detection(unattacked_scores, attacked_scores)
        time_of_first_detection_flipped = get_time_of_first_detection(neg_unattacked_scores, neg_attacked_scores)
        
        use_flipped = True if time_of_first_detection_flipped < time_of_first_detection_normal else False

        time_of_first_detection[i] = time_of_first_detection_flipped if use_flipped else time_of_first_detection_normal

        eval_metrics([-1*s for s in scores] if use_flipped else scores, 
                     labels, 
                     axs[exp_n, i*2], 
                     axs[exp_n, i*2+1])

    # Calculate the position for the title
    title_y = axs[exp_n, 0].get_position().y1 + 0.001  # Adjust as needed
    
    # Add a title for each row of subplots
    row_title = f'Avg Time: {time_of_first_detection[0]} -- Experiment {experiment_id} Metrics -- Avg Time: {time_of_first_detection[1]}'
    fig.text(0.5, title_y, row_title, fontsize=25, ha='center')

save_path = f"all_results.pdf"
plt.savefig(save_path)

# %%
