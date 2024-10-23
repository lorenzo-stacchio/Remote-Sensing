import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

# Function to extract test_accuracy from TensorBoard logs
def extract_variable(logdir, variable = "test_accuracy"):
    # Load TensorBoard event file
    ea = event_accumulator.EventAccumulator(logdir)
    ea.Reload()
    # Get test_accuracy data
    accuracy_events = ea.Scalars(variable)
    steps = [event.step for event in accuracy_events]
    values = [event.value for event in accuracy_events]
    return steps, values

# Directory containing all log files
m_dir = '/home/vrai/disk2/LorenzoStacchio/Remote Sensing/Remote Sensing/classification/'

# log_dir = f'{m_dir}logs/'
# exp = "RECSIS45"
# log_dir = f'{m_dir}logs/{exp}/'

exp = "Clip_ViT_30"
log_dir = f'{m_dir}logs_objd/{exp}/'

# Iterate through each subdirectory (one log file per run)
all_logs = [os.path.join(log_dir, d) for d in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, d))]

# Plotting each log's test accuracy
plt.figure(figsize=(10, 6))
for log in all_logs:
    steps, values = extract_variable(log,"test_accuracy")
    plt.plot(steps, values, label=os.path.basename(log))

# Customize plot
plt.title("Test Accuracy Over Epochs")
plt.xlabel("Steps")
plt.ylabel("Test Accuracy")
plt.legend(loc="best")
plt.grid(True)
# plt.show()
plt.savefig(m_dir + f"recap_{exp}.png")