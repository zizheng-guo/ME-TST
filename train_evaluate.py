from training_utils import *

def final_evaluation(TP_spot, FP_spot, FN_spot, dataset_name, pred_list, emotion_type, gt_tp_list):
    #Spotting
    precision = TP_spot/(TP_spot+FP_spot)
    recall = TP_spot/(TP_spot+FN_spot)
    F1_score = (2 * precision * recall) / (precision + recall)
    print('----Spotting----')
    print('Final Result for', dataset_name)
    print('TP:', TP_spot, 'FP:', FP_spot, 'FN:', FN_spot)
    print('Precision = ', round(precision, 4))
    print('Recall = ', round(recall, 4))
    print('F1-Score = ', round(F1_score, 4))
    # print("COCO AP@[.5:.95]:", round(metric_final.value(iou_thresholds=np.round(np.arange(0.5, 1.0, 0.05), 2), mpolicy='soft')['mAP'], 4))

    print('\n----Recognition (Consider TP only)----')
    gt_tp_spot = []
    pred_tp_spot = []
    for index in range(len(gt_tp_list)):
        if(gt_tp_list[index]!=-1):
            gt_tp_spot.append(gt_tp_list[index])
            pred_tp_spot.append(pred_list[index])
    print('Predicted    :', pred_tp_spot)
    print('Ground Truth :', gt_tp_spot)
    f1_recog = recognition_evaluation(dataset_name, emotion_type, gt_tp_spot, pred_tp_spot, show=True)
    # print('Accuracy Score:', round(accuracy_score(gt_tp_spot, pred_tp_spot), 4))

    if dataset_name == "SAMMLV":
        print('\n----wo others----',)
        gt_tp_spot_3emo = []
        pred_tp_spot_3emo = []
        for index in range(len(gt_tp_spot)):
            if(gt_tp_spot[index]!=3):
                gt_tp_spot_3emo.append(gt_tp_spot[index])
                pred_tp_spot_3emo.append(pred_tp_spot[index])         
        print('Predicted    :', pred_tp_spot_3emo)
        print('Ground Truth :', gt_tp_spot_3emo)  
        f1_recog =  recognition_evaluation(dataset_name, emotion_type-1, gt_tp_spot_3emo, pred_tp_spot_3emo, show=True)

    print('SRTS:', round(F1_score * f1_recog, 4))