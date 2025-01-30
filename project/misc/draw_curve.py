"""
analysis method for experiment results.
across 5 fold cross validation and then print the metrics.
meanwhile can print the CM.
"""

# %%
import torch 
import torch.nn.functional as F
import os 
import sys

sys.path.append('/workspace/Walk_Video_PyTorch/project')

from models.pytorchvideo_models import WalkVideoClassificationLightningModule
from dataloader.data_loader import WalkDataModule
from pytorchvideo.transforms.functional import uniform_temporal_subsample

from IPython.display import clear_output

clear_output()

from pytorch_lightning import seed_everything

seed_everything(42, workers=True)

import torchmetrics

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# %%
# define the metrics.
_accuracy = torchmetrics.classification.BinaryAccuracy()
_precision = torchmetrics.classification.BinaryPrecision()
_recall = torchmetrics.classification.BinaryRecall()
_f1_score = torchmetrics.classification.BinaryF1Score()
_aucroc = torchmetrics.classification.BinaryAUROC()

_confusion_matrix = torchmetrics.classification.BinaryConfusionMatrix(normalize='true')


# %%
class opt:
    num_workers = 8
    batch_size = 64

    fusion_method = 'slow_fusion'
    fix_layer = 'all'

    model = "resnet"

    img_size = 224
    lr=0.0001
    model_class_num = 1
    model_depth = 50

    transfor_learning = True
    pre_process_flag = True
    split_pad_data_path = '/workspace/data/three_part_split_pad_dataset/A/'

# DATA_PATH = "/workspace/data/segmentation_dataset_512"
# DATA_PATH = "/workspace/data/split_pad_dataset_512"
DATA_PATH = "/workspace/data/three_part_split_pad_dataset"

# %%
def get_best_ckpt(length: str, frame: str, fold: str):

    ckpt_path = '/workspace/Walk_Video_PyTorch/logs/resnet/'
    version = '516_1_8'
    ckpt_path_list = os.listdir(ckpt_path)
    ckpt_path_list.sort()

    final_ckpt_list = [] 

    for i in ckpt_path_list:
        if version in i:
            final_ckpt_list.append(i)

    final_ckpt_list.sort()
    
    for name in final_ckpt_list:
        if length in name.split('_') and frame in name.split('_'):
            ckpt = name

    ckpt = os.path.join(ckpt_path, ckpt, fold, 'checkpoints')
    
    Acc = 0.0

    ckpt_list = os.listdir(ckpt)
    ckpt_list.sort()
    ckpt_list.remove('last.ckpt')

    for num, l in enumerate(ckpt_list):
        acc = l[-11:-5] # accuracy

        if float(acc) > float(Acc):
            Acc = acc
            NUM = num

    return os.path.join(ckpt, ckpt_list[NUM])

# ckpt_path = get_best_ckpt('1', '31', 'fold0')

# %%
def get_inference(test_data, model):
        
    pred_list = []
    label_list = []

    for i, batch in enumerate(test_data):

        # input and label
        video = batch['video'].detach().cuda() # b, c, t, h, w

        label = batch['label'].detach().cuda() # b, class_num

        model.cuda().eval()

        # pred the video frames
        with torch.no_grad():
            preds = model(video)

        # when torch.size([1]), not squeeze.
        if preds.size()[0] != 1 or len(preds.size()) != 1 :
            preds = preds.squeeze(dim=-1)
            preds_sigmoid = torch.sigmoid(preds)
        else:
            preds_sigmoid = torch.sigmoid(preds)

        pred_list.append(preds_sigmoid.tolist())
        label_list.append(label.tolist())

        total_pred_list = []
        total_label_list = []

        for i in pred_list:
            for number in i:
                total_pred_list.append(number)

        for i in label_list:
            for number in i: 
                total_label_list.append(number)

    pred = torch.tensor(total_pred_list)
    label = torch.tensor(total_label_list)

    return total_pred_list, total_label_list

# %%
VIDEO_LENGTH = ['1']
VIDEO_FRAME = ['8']

# %%
fold_num = os.listdir(os.path.join(DATA_PATH, 'C'))
fold_num.sort()
if 'raw' in fold_num:
    fold_num.remove('raw')

symbol = '_'

one_condition_pred_list = []
one_condition_label_list = []

total_pred_list = []
total_label_list = []

for length in VIDEO_LENGTH:

    for frame in VIDEO_FRAME:

        for fold in fold_num:

            opt.train_path = os.path.join(DATA_PATH, 'C', fold)

            #################
            # start k Fold CV
            #################
            
            opt.clip_duration = int(length)
            opt.uniform_temporal_subsample_num = int(frame)
           
            ckpt_path = get_best_ckpt(length, frame, fold)

            print('#' * 50)
            print('Strat %s, %s length, %s frames' % (fold, length, frame))
            print('the data path: %s' % opt.train_path)
            print('ckpt: %s' % ckpt_path)
            model = WalkVideoClassificationLightningModule(opt).load_from_checkpoint(ckpt_path)

            data_module = WalkDataModule(opt)
            data_module.setup()
            test_data = data_module.test_dataloader()

            pred_list, label_list = get_inference(test_data, model)

            one_condition_pred_list.append(pred_list)

            one_condition_label_list.append(label_list)

        # total 5 fold pred and label
        for i in one_condition_pred_list:
            for number in i:
                total_pred_list.append(number)

        for i in one_condition_label_list:
            for number in i: 
                total_label_list.append(number)

        pred = torch.tensor(total_pred_list)    
        label = torch.tensor(total_label_list)

        print('*' * 100)
        print('the result of %ss %sframe:' % (length, frame))
        print('accuracy: %s' % _accuracy(pred, label))
        print('precision: %s' % _precision(pred, label))
        print('_binary_recall: %s' % _recall(pred, label))
        print('_binary_f1: %s' % _f1_score(pred, label))
        print('_aurroc: %s' % _aucroc(pred, label))
        print('_confusion_matrix: %s' % _confusion_matrix(pred, label))
        print('#' * 100)

# %%
import seaborn as sns
import matplotlib.pyplot as plt

cm = _confusion_matrix(pred, label).to(float)

total = sum(sum(cm))

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        cm[i][j] = float(cm[i][j] / total)

ax = sns.heatmap(cm, annot=True, fmt=".4f", xticklabels=['ASD', 'non_ASD'], yticklabels=['ASD', 'non_ASD'])

ax.set_title('Confusion Matrix')
ax.set(xlabel="pred class", ylabel="ground truth")
ax.xaxis.tick_top()
plt.show()

# %%
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import RocCurveDisplay


fpr, tpr, _ = roc_curve(label, pred, pos_label=1)
roc_auc = auc(fpr, tpr)
roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name="proposed method").plot()
plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
plt.axis("square")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("AUROC curves")
plt.legend()
plt.show()

# %% 
len(one_condition_pred_list[0])
# %%
one_acc = []
one_f1 = []
one_auroc = []

for i in range(len(one_condition_label_list)):
    one_batch_pred = torch.tensor(one_condition_pred_list[i]).cpu()
    one_batch_label = torch.tensor(one_condition_label_list[i]).cpu()

    print('*' * 100)
    print('the result of Fold %s, %ss %sframe:' % (fold, opt.clip_duration, opt.uniform_temporal_subsample_num))
    print('accuracy: %s' % _accuracy(one_batch_pred, one_batch_label))
    one_acc.append(_accuracy(one_batch_pred, one_batch_label).tolist())
    print('precision: %s' % _precision(one_batch_pred, one_batch_label))
    print('recall: %s' % _recall(one_batch_pred, one_batch_label))
    print('f1_score: %s' % _f1_score(one_batch_pred, one_batch_label))
    one_f1.append(_f1_score(one_batch_pred, one_batch_label).tolist())
    print('aurroc: %s' % _aucroc(one_batch_pred, one_batch_label))
    one_auroc.append(_aucroc(one_batch_pred, one_batch_label).tolist())
    print('confusion_matrix: %s' % _confusion_matrix(one_batch_pred, one_batch_label))
    print('#' * 100)

    cm = _confusion_matrix(one_batch_pred, one_batch_label)
    ax = sns.heatmap(cm, annot=True, fmt="3d")

    ax.set_title('Fold%d confusion matrix' % i)
    ax.set(xlabel="pred class", ylabel="ground truth")
    ax.xaxis.tick_top()
    plt.show()
# %%
import numpy as np 

print(np.mean(one_acc), np.var(one_acc), np.std(one_acc))
print(np.mean(one_f1), np.var(one_f1), np.std(one_f1))
print(np.mean(one_auroc), np.var(one_auroc), np.std(one_auroc))

# %%
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np 
cm =  _confusion_matrix(pred, label)
# cm = np.array([[0,0], [0,0]])
ax = sns.heatmap(cm, annot=True, annot_kws={"fontsize": 20}, vmin=0, vmax=1, fmt=".4f", xticklabels=['ASD', 'non_ASD'], yticklabels=['ASD', 'non_ASD'])

# color bar font size 
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=15)

ax.set_title('C. C part, walking end.', fontsize=20)
ax.set_xlabel("pred class", fontsize=20)
ax.set_ylabel("ground truth", fontsize=20)
ax.xaxis.tick_top()

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.savefig('C.png')
plt.show()

# %%
