from sklearn.metrics import roc_auc_score
import torch


# 输入两个AUC的值，进行计算
def compAUC(true_label, predictions):
    true_label = true_label.squeeze()
    predictions = predictions.squeeze()
    ture_label = true_label.type(torch.int)
    true_label_np = ture_label.cpu().numpy()
    predictions_np = predictions.cpu().numpy()
    print(true_label_np)
    print(predictions_np)
    auc = roc_auc_score(true_label_np, predictions_np)
    return auc

# 计算 sn sp Mcc
def SN_SP_MCC(true_label, pred_label):
    true_label = true_label.squeeze()
    pred_label = pred_label.squeeze()

    ture_label = true_label.type(torch.int)
    pred_label = pred_label.type(torch.int)

    # 重点观察真实样本为1 预测为1 fp 真实标签为 1 预测为 0
    TP = ((pred_label == 1) & (ture_label == 1)).sum().item()
    TN = ((pred_label == 0) & (ture_label == 0)).sum().item()
    FP = ((pred_label == 1) & (ture_label == 0)).sum().item()
    FN = ((pred_label == 0) & (ture_label == 1)).sum().item()

    # 计算 Sensitivity (SN), Specificity (SP), Matthews Correlation Coefficient (MCC)
    SN = TP / (TP + FN) if (TP + FN) != 0 else 0
    SP = TN / (TN + FP) if (TN + FP) != 0 else 0
    MCC_numerator = (TP * TN) - (FP * FN)
    MCC_denominator = ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) ** 0.5
    MCC = MCC_numerator / MCC_denominator if MCC_denominator != 0 else 0

    print(f"TP: {TP}, FP: {FP}, Sensitivity (SN): {SN}, Specificity (SP): {SP}, Matthews Correlation Coefficient (MCC): {MCC}")

