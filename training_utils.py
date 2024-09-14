import numpy as np
from collections import Counter
from scipy.signal import find_peaks
from Utils.mean_average_precision.mean_average_precision import MeanAveragePrecision2d
from numpy import argmax
from sklearn.metrics import confusion_matrix
import random

def smooth(y, box_pts):
    y = [each_y for each_y in y]
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def spotting(final_samples, subject_count, pred_interval, total_gt_spot, p, metric_final, k_p):
    pred_subject = []
    gt_subject = []
    metric_video = MeanAveragePrecision2d(num_classes=1)
    for videoIndex, video in enumerate(final_samples[subject_count]):
        preds = []
        gt = []
        score_plot = np.array(pred_interval[videoIndex])
        score_plot = smooth(score_plot, k_p*2)
        score_plot_agg = score_plot.copy()
        threshold = score_plot_agg.mean() + p * (max(score_plot_agg) - score_plot_agg.mean()) #Moilanen threshold technique

        peaks, _ = find_peaks(score_plot_agg, height=threshold, distance=k_p)
        if(len(peaks)==0): #Occurs when no peak is detected, simply give a value to pass the exception in mean_average_precision
            preds.append([0, 0, 0, 0, 0, 0, 0]) 
        for peak in peaks:
            preds.append([peak-k_p, 0, peak+k_p, 0, 0, 0, peak]) #Extend left and right side of peak by k frames

        for samples in video: 
            gt.append([samples[0], 0, samples[2], 0, 0, 0, 0, samples[1]])
            total_gt_spot += 1
        # print(preds)
        # print(gt)
        metric_video.add(np.array(preds),np.array(gt))
        metric_final.add(np.array(preds),np.array(gt)) #IoU = 0.5 according to MEGC2020 metrics
        pred_subject.append(preds)
        gt_subject.append(gt)
    return pred_subject, gt_subject, total_gt_spot, metric_video, metric_final

def confusionMatrix(gt, pred, show=False):
    TN_recog, FP_recog, FN_recog, TP_recog = confusion_matrix(gt, pred).ravel()
    f1_score = (2*TP_recog) / (2*TP_recog + FP_recog + FN_recog)
    num_samples = len([x for x in gt if x==1])
    average_recall = TP_recog / (TP_recog + FN_recog)
    average_precision = TP_recog / (TP_recog + FP_recog)
    return f1_score, average_recall, TP_recog, FP_recog, FN_recog, TN_recog, num_samples, average_precision, average_recall


def sequence_evaluation(total_gt_spot, metric_final): #Get TP, FP, FN for final evaluation
    TP_spot = int(sum(metric_final.value(iou_thresholds=0.5)[0.5][0]['tp'])) 
    FP_spot = int(sum(metric_final.value(iou_thresholds=0.5)[0.5][0]['fp']))
    FN_spot = total_gt_spot - TP_spot
    print('TP:', TP_spot, 'FP:', FP_spot, 'FN:', FN_spot)
    return TP_spot, FP_spot, FN_spot

def convertLabel(label):
    label_dict = { 'negative' : 0, 'positive' : 1, 'surprise' : 2, 'others' : 3 }
    return label_dict[label]
    
def splitVideo(y1_pred, subject_count, final_samples, final_dataset_spotting): #To split y1_act_test by video
    prev=0
    y1_pred_video = []
    for videoIndex, video in enumerate(final_samples[subject_count-1]):
        countVideo = len([video for subject in final_samples[:subject_count-1] for video in subject])
        y1_pred_each = y1_pred[prev:prev+len(final_dataset_spotting[countVideo+videoIndex])+1]
        y1_pred_video.append(y1_pred_each)
        prev += len(final_dataset_spotting[countVideo+videoIndex])
    return y1_pred_video

def recognition(result, preds, metric_video, final_emotions, subject_count, pred_list, gt_tp_list, final_samples, pred_window_list, pred_single_list,frame_skip):
    cur_pred = []
    cur_tp_gt = []
    pred_gt_recog = []
    cur_pred_window = []
    cur_pred_single = []
    pred_emotion = result #splitVideo(result, subject_count+1, final_samples, final_dataset_spotting) #Split predicted emotion by video
    pred_match_gt = sorted(metric_video.value(iou_thresholds=0.5)[0.5][0]['pred_match_gt'].items())
    for video_index, video_match in pred_match_gt: #key=video_index, value=match index for each video
        for pred_index, sample_index in enumerate(video_match): #pred_index=index of prediction array, sample_index=index of emotion array
            pred_onset = max(0, preds[video_index][pred_index][0])
            # pred_peak = max(0, preds[video_index][pred_index][-1]) 
            pred_offset = max(0, preds[video_index][pred_index][2])
            pred_emotion_list = pred_emotion[video_index][max(0, pred_onset +1):max(1, pred_offset -1)]
            most_common_emotion, _ = Counter(pred_emotion_list).most_common(1)[0]
            cur_pred.append(most_common_emotion)
            
            pred_gt_recog.append(argmax(pred_emotion[video_index][final_samples[subject_count][video_index][0][0]])) #Predicted emotion on gt onset label
            gt_label = final_emotions[subject_count][video_index][sample_index] #Get video emotion    
            if(sample_index!=-1):
                cur_tp_gt.append(convertLabel(gt_label))
            else:
                cur_tp_gt.append(-1)
    pred_list.extend(cur_pred)
    gt_tp_list.extend(cur_tp_gt)
    pred_window_list.extend(cur_pred_window)
    pred_single_list.extend(cur_pred_single)
    print('Predicted with k_p     :', cur_pred)
    return pred_list, cur_pred, gt_tp_list, pred_window_list, pred_single_list

def recognition_evaluation(dataset_name, emotion_class, final_gt, final_pred, show=False):
    if(emotion_class == 5):
        label_dict = { 'negative' : 0, 'positive' : 1, 'surprise' : 2, 'others' : 3 }
    else:
        label_dict = { 'negative' : 0, 'positive' : 1, 'surprise' : 2 }
    
    #Display recognition result
    precision_list = []
    recall_list = []
    f1_list = []
    ar_list = []
    TP_all = 0
    FP_all = 0
    FN_all = 0
    TN_all = 0
    try:
        for emotion, emotion_index in label_dict.items():
            gt_recog = [1 if x==emotion_index else 0 for x in final_gt]
            pred_recog = [1 if x==emotion_index else 0 for x in final_pred]
            try:
                f1_recog, ar_recog, TP_recog, FP_recog, FN_recog, TN_recog, num_samples, precision_recog, recall_recog = confusionMatrix(gt_recog, pred_recog, show)
                if(show):
                    print(emotion.title(), 'Emotion:')
                    print('TP:', TP_recog, '| FP:', FP_recog, '| FN:', FN_recog, '| TN:', TN_recog)
#                     print('Total Samples:', num_samples, '| F1-score:', round(f1_recog, 4), '| Average Recall:', round(recall_recog, 4), '| Average Precision:', round(precision_recog, 4))
                TP_all += TP_recog
                FP_all += FP_recog
                FN_all += FN_recog
                TN_all += TN_recog
                precision_list.append(precision_recog)
                recall_list.append(recall_recog)
                f1_list.append(f1_recog)
                ar_list.append(ar_recog)
            except Exception as e:
                pass
        precision_list = [0 if np.isnan(x) else x for x in precision_list]
        recall_list = [0 if np.isnan(x) else x for x in recall_list]
        precision_all = np.mean(precision_list)
        recall_all = np.mean(recall_list)
        f1_all = (2 * precision_all * recall_all) / (precision_all + recall_all)
        UF1 = np.mean(f1_list)
        UAR = np.mean(ar_list)
        print('------ After adding ------')
        print('TP:', TP_all, 'FP:', FP_all, 'FN:', FN_all, 'TN:', TN_all)
        print('Precision:', round(precision_all, 4), 'Recall:', round(recall_all, 4))
        print('UF1:', round(UF1, 4), '| UAR:', round(UAR, 4), '| F1-Score:', round(f1_all, 4))
        return f1_all
    except:
        return None


def downSampling(Y_spot, ratio):
    #Downsampling non expression samples to make ratio expression:non-expression 1:ratio

    rem_index = list(index for index, i in enumerate(Y_spot) if np.sum(i)>0)
    rem_count = int(len(rem_index) * ratio)

    #Randomly remove non expression samples (With label 0) from dataset
    if len([index for index, i in enumerate(Y_spot) if np.sum(i)==0]) <= rem_count:
        rem_count = len([index for index, i in enumerate(Y_spot) if np.sum(i)==0]) - 2
    rem_index += random.sample([index for index, i in enumerate(Y_spot) if np.sum(i)==0], rem_count) 
    rem_index.sort()
    
    # Simply return 50 index
    if len(rem_index) == 0:
        print('No index selected')
        rem_index = [i for i in range(50)]
    return rem_index