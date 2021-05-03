import torch
import pdb
import numpy as np
import torch.nn.functional as F
from keras import backend as K


# SR : Segmentation Result
# GT : Ground Truth

def get_accuracy(SR,GT,threshold=0.5):
    SR = (SR > threshold).float()
    GT = (GT == torch.max(GT)).float()
    corr = torch.sum(SR==GT)
    tensor_size = SR.size(0)*SR.size(1)*SR.size(2)*SR.size(3)
    acc = float(corr)/float(tensor_size)

    return acc

def get_TP_FN_FP(SR, GT, threshold=0.5):
    # Sensitivity == Recall
    SR = (SR > threshold).float()
    GT = (GT == torch.max(GT)).float()

    # TP : True Positive
    # FN : False Negative
    TP = ((SR==1).float()+(GT==1).float())==2
    FN = ((SR==0).float()+(GT==1).float())==2
    FP = ((SR==1).float()+(GT==0).float())==2
    return float(torch.sum(TP)), float(torch.sum(FN)), float(torch.sum(FP))

def get_TP_TN_FP_FN(SR, GT, threshold=0.5):
    # Sensitivity == Recall
    SR = (SR > threshold).float()
    GT = (GT == torch.max(GT)).float()

    # TP : True Positive
    # FN : False Negative
    TP = ((SR==1).float()+(GT==1).float())==2
    FN = ((SR==0).float()+(GT==1).float())==2
    FP = ((SR==1).float()+(GT==0).float())==2
    TN = ((SR==0).float()+(GT==0).float())==2
    return float(torch.sum(TP)), float(torch.sum(TN)), \
            float(torch.sum(FP)), float(torch.sum(FN))

def get_sensitivity(SR,GT,threshold=0.5):
    # Sensitivity == Recall
    SR = (SR > threshold).float()
    GT = (GT == torch.max(GT)).float()

    # TP : True Positive
    # FN : False Negative
    TP = ((SR==1).float()+(GT==1).float())==2
    FN = ((SR==0).float()+(GT==1).float())==2

    SE = float(torch.sum(TP))/(float(torch.sum(TP+FN)) + 1e-6)     
    
    return SE

def get_specificity(SR,GT,threshold=0.5):
    SR = (SR > threshold).float()
    GT = (GT == torch.max(GT)).float()

    # TN : True Negative
    # FP : False Positive
    TN = ((SR==0).float()+(GT==0).float())==2
    FP = ((SR==1).float()+(GT==0).float())==2

    SP = float(torch.sum(TN))/(float(torch.sum(TN+FP)) + 1e-6)
    
    return SP

def get_precision(SR,GT,threshold=0.5):
    SR = (SR > threshold).float()
    GT = (GT == torch.max(GT)).float()

    # TP : True Positive
    # FP : False Positive
    TP = ((SR==1).float()+(GT==1).float())==2
    FP = ((SR==1).float()+(GT==0).float())==2

    PC = float(torch.sum(TP))/(float(torch.sum(TP+FP)) + 1e-6)

    return PC

def get_F1(SR,GT,threshold=0.5):
    # Sensitivity == Recall
    SE = get_sensitivity(SR,GT,threshold=threshold)
    PC = get_precision(SR,GT,threshold=threshold)

    F1 = 2*SE*PC/(SE+PC + 1e-6)

    return F1

def get_JS(SR,GT,threshold=0.5):
    # JS : Jaccard similarity
    # also known as intersection over union, IoU
    SR = (SR > threshold).float()
    GT = (GT == torch.max(GT)).float()
    
    Inter = torch.sum((SR+GT)==2)
    Union = torch.sum((SR+GT)>=1)
    
    JS = float(Inter)/(float(Union) + 1e-6)
    
    return JS

def get_DC(SR,GT,threshold=0.5):
    # DC : Dice Coefficient
    SR = (SR > threshold).float()
    GT = (GT == torch.max(GT)).float()

    Inter = torch.sum((SR+GT)==2)
    DC = float(2*Inter)/(float(torch.sum(SR)+torch.sum(GT)) + 1e-6)

    return DC

def recall_m(y_true, y_pred):
  true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
  possible_positives = np.sum(np.round(np.clip(y_true, 0, 1)))
  recall = true_positives / (possible_positives + K.epsilon())
  return recall

def precision_m(y_true, y_pred):
  true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
  predicted_positives = np.sum(np.round(np.clip(y_pred, 0, 1)))
  precision = true_positives / (predicted_positives + K.epsilon())
  return precision

def f1_m(y_true, y_pred):
  y_true = (y_true > 0.5).float()
  y_pred = (y_pred > 0.5).float()
  y_true = y_true.detach().cpu().numpy()
  y_pred = y_pred.detach().cpu().numpy()
  precision = precision_m(y_true, y_pred)
  recall = recall_m(y_true, y_pred)
  return 2*((precision*recall)/(precision+recall+K.epsilon()))

def jaccard_m(y_true, y_pred):
  y_true = (y_true > 0.5).float()
  y_pred = (y_pred > 0.5).float()
  y_true = y_true.detach().cpu().numpy()
  y_pred = y_pred.detach().cpu().numpy()
  intersection = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
  union = np.sum(y_true)+np.sum(y_pred)-intersection
  return intersection/(union+K.epsilon())
  

if __name__ == "__main__":
    pass
