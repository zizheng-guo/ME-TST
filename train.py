import os
import time
from sklearn.model_selection import LeaveOneGroupOut
import torch
from torch.utils.data import DataLoader
from Utils.mean_average_precision_str.mean_average_precision import MeanAveragePrecision2d
from numpy import argmax
import torch.nn as nn
from sklearn.utils import class_weight
import torch.nn.functional as F
from training_utils import *
from dataloader import *
from network import *
from feature_extraction import *

def train_model(train, X_spot, Y_spot, Y1_spot, groupsLabel_spot, groupsLabel_recog, final_dataset_spotting, final_subjects, final_samples, final_videos, final_emotions, emotion_type, epochs, lr, batch_size, dataset_name, k_p, k, ratio, frame_skip, strategy, note):
    
    # Create model directory
    if train:
        os.makedirs("save_models/%s_%semo_%s/" % (dataset_name, emotion_type-1, note), exist_ok=True)
    start = time.time()
    loso = LeaveOneGroupOut()
    subject_count = 0
    transform = None
    device = torch.device('cuda')

    # Spot
    spot_train_index = []
    spot_test_index = []
    metric_final = MeanAveragePrecision2d(num_classes=1)

    # LOSO
    for train_index, test_index in loso.split(X_spot, X_spot, groupsLabel_spot):
        spot_train_index.append(train_index)
        spot_test_index.append(test_index)

    total_gt_spot = 0

    pred_list = []
    gt_tp_list = []
    pred_window_list = []
    pred_single_list = []

    # neutral
    pred_neutral_list = []
    gt_tp_neutral_list = []

    # Training and Testing
    subjects_unique = sorted(np.unique(final_subjects))
    for subject_count in range(len(subjects_unique)): 

        # Use copy to ensure the original value is not modified
        X_spot_train, X_spot_test   = [X_spot[i] for i in spot_train_index[subject_count]], [X_spot[i] for i in spot_test_index[subject_count]]
        Y_spot_train, Y_spot_test   = [Y_spot[i] for i in spot_train_index[subject_count]], [Y_spot[i] for i in spot_test_index[subject_count]]
        Y1_spot_train, Y1_spot_test = [Y1_spot[i] for i in spot_train_index[subject_count]], [Y1_spot[i] for i in spot_test_index[subject_count]]

        print('Subject : ' + str(subject_count+1), ', spNO.', subjects_unique[subject_count])

        # # Create final dataset for training
        rem_index = downSampling(Y_spot_train, ratio)
        X_train_final = [X_spot_train[i] for i in rem_index]
        Y_train_final = [Y_spot_train[i] for i in rem_index]
        Y1_train_final = [argmax(Y1_spot_train[i],-1) for i in rem_index]

        rem_index = downSampling(Y_spot_test, ratio)
        X_val_final = [X_spot_test[i] for i in rem_index]
        Y_val_final = [Y_spot_test[i] for i in rem_index]
        Y1_val_final = [argmax(Y1_spot_test[i],-1) for i in rem_index]
        # Create final dataset for testing
        X_test_final = X_spot_test
        Y_test_final = Y_spot_test
        Y1_test_final = argmax(Y1_spot_test,-1).tolist()

        # Initialize training dataloader
        X_train_final = torch.Tensor(np.array(X_train_final)) #.permute(0,4,1,2,3) #.permute(0,3,1,2)
        Y_train_final = torch.Tensor(np.array(Y_train_final))
        Y1_weight_final= torch.Tensor(np.array(Y1_train_final)).type(torch.long)
        Y1_train_final= torch.Tensor(F.one_hot(torch.tensor(Y1_train_final)).float())
        train_dl = DataLoader(
            OFFSTRDataset((X_train_final[:, :][:, None, :], Y_train_final, Y1_train_final), transform=transform, train=True),
            batch_size=batch_size,
            shuffle=True,
        )
        # Initialize validation dataloader
        X_val_final = torch.Tensor(np.array(X_val_final).astype(float)) #.permute(0,4,1,2,3) #.permute(0,3,1,2)
        Y_val_final = torch.Tensor(np.array(Y_val_final))
        Y1_val_final = torch.Tensor(F.one_hot(torch.tensor(Y1_val_final)).float())
        val_spot_dl = DataLoader(
            OFFSTRDataset((X_val_final[:, :][:, None, :],  Y_val_final, Y1_val_final), transform=transform, train=False),
            batch_size=batch_size,
            shuffle=False,
        )
        # Initialize testing dataloader
        X_test_final = torch.Tensor(np.array(X_test_final)) #.permute(0,4,1,2,3) #.permute(0,3,1,2)
        Y_test_final = torch.Tensor(np.array(Y_test_final))
        Y1_test_final = torch.Tensor(F.one_hot(torch.tensor(Y1_test_final)).float())
        test_spot_dl = DataLoader(
            OFFSTRDataset((X_test_final[:, :][:, None, :],  Y_test_final, Y1_test_final), transform=transform, train=False),
            batch_size=batch_size,
            shuffle=False,
        )

        # Loss function
        loss_fn_spot = nn.MSELoss()
        class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.array([i for i in range(emotion_type)]), y=np.array(Y1_weight_final[Y1_weight_final != emotion_type]))
        class_weights[-1] = class_weights[-1] * 6
        if strategy == 0:
            class_weights[-1] = 0

        class_weights = torch.tensor(class_weights,dtype=torch.float).cuda()
        loss_fn_recog = nn.CrossEntropyLoss(weight=class_weights,reduction='mean') 
        print('Class Weights:', class_weights)

        print('------Initializing Network-------') #To reset the model at every LOSO testing
        # model and optimizer
        model = Net_Mamba(out_channels=emotion_type).cuda()
        model = nn.DataParallel(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=len(train_dl))

        if train: # Train
            for epoch in range(epochs):
                model.train()
                for batch in train_dl:
                    x   = batch[0].to(device)
                    y    = batch[1].to(device)
                    y1   = batch[2].to(device)
                    optimizer.zero_grad()
                    yhat, yhat1 = model(x)
                    yhat = yhat.view(-1)
                    y = y.view(-1)
                    yhat1 = yhat1.view(-1, emotion_type)
                    y1 = y1.view(-1, emotion_type)
                    loss_spot = loss_fn_spot(yhat, y)
                    loss_recog = loss_fn_recog(yhat1, y1)
                    loss = loss_spot * 0.9 + loss_recog * 0.1
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                # Validation
                if epoch % 6 == 0:
                    model.eval()
                    val_loss = 0.0
                    val_spot_loss = 0.0    
                    for batch in val_spot_dl:
                        x   = batch[0].to(device)
                        y    = batch[1].to(device)
                        y1   = batch[2].to(device)
                        yhat, yhat1 = model(x)
                        yhat = yhat.view(-1)
                        y = y.view(-1)
                        yhat1 = yhat1.view(-1, emotion_type)
                        y1 = y1.view(-1, emotion_type)
                        loss_spot = loss_fn_spot(yhat, y)
                        loss_recog = loss_fn_recog(yhat1, y1)
                        loss = loss_spot * 0.9 + loss_recog * 0.1
                        val_loss += loss.data.item()
                        val_spot_loss += loss_spot.data.item()
                    val_loss = val_loss / len(val_spot_dl)
                    val_spot_loss = val_spot_loss / len(val_spot_dl)
                    print('Epoch ' + str(epoch) + '             Loss' + str(val_loss) + '               Loss_spot' + str(val_spot_loss))

            # Save models
            torch.save(model.state_dict(), os.path.join("save_models/%s_%semo_%s/subject_%s.pkl" % (dataset_name, emotion_type-1, note, str(final_subjects[subject_count]))))
            # Testing spot-then-recognize
            model.eval()
            video_num = []
            for video_index, video in enumerate(final_samples[subject_count]):
                countVideo = len([video for subject in final_samples[:subject_count] for video in subject])
                video_num.append(len(final_dataset_spotting[countVideo+video_index]))

            videocount = 0
            framecount = 0
            result_all = []
            result1_all = []
            result_video = np.zeros((video_num[videocount]+1)*k//2)
            result1_video = np.zeros((video_num[videocount]+1)*k//2)

            for batch in test_spot_dl:
                x  =  batch[0].to(device)
                yhat, yhat1 = model(x)
                yhat = yhat.cpu().data.numpy()

                if strategy == 2:
                    yhat1 = torch.max(yhat1[:,:,0:4], 2)[1].tolist()  
                else:
                    yhat1 = torch.max(yhat1, 2)[1].tolist()  

                for i in range(len(yhat)):
                    if framecount == video_num[videocount]:
                        framecount = 0
                        videocount += 1
                        result_all.append(result_video)
                        result1_all.append(result1_video)
                        result_video = np.zeros((video_num[videocount]+1)*k//2)
                        result1_video = np.zeros((video_num[videocount]+1)*k//2)
                    if i == 0:
                        result_video[framecount*k//2:(framecount+2)*k//2] = yhat[i]
                        result1_video[framecount*k//2:(framecount+2)*k//2] = yhat1[i]
                    else:
                        result_video[(framecount+1)*k//2:(framecount+2)*k//2] = yhat[i][k//2:]
                        result1_video[(framecount+1)*k//2:(framecount+2)*k//2] = yhat1[i][k//2:]    
                    framecount += 1
                if framecount == video_num[videocount] and videocount == len(video_num) -1:
                    result_all.append(result_video)
                    result1_all.append(result1_video)

            print('---- Spotting Results ----')
            preds, gt, total_gt_spot, metric_video, metric_final = spotting(final_samples, subject_count, result_all, total_gt_spot, 0.55, metric_final, k_p)
            TP_spot, FP_spot, FN_spot = sequence_evaluation(total_gt_spot, metric_final)
            print('---- Recognition Results ----')
            pred_list, preds_reg, gt_tp_list, pred_window_list, pred_single_list = recognition(result1_all, preds, metric_video, final_emotions, subject_count, pred_list, gt_tp_list, final_samples, pred_window_list, pred_single_list, frame_skip)

        else: # Test
            model.load_state_dict(torch.load("weights/%s_%semo/subject_%s.pkl" % (dataset_name, emotion_type-1, str(final_subjects[subject_count]))))
            model.eval()
            # Testing spot-then-recognize
            video_num = []
            for video_index, video in enumerate(final_samples[subject_count]):
                countVideo = len([video for subject in final_samples[:subject_count] for video in subject])
                video_num.append(len(final_dataset_spotting[countVideo+video_index]))

            videocount = 0
            framecount = 0
            result_all = []
            result1_all = []
            result_video = np.zeros((video_num[videocount]+1)*k//2)
            result1_video = np.zeros((video_num[videocount]+1)*k//2)

            with torch.no_grad():
                for batch in test_spot_dl:
                    x  =  batch[0].to(device)
                    yhat, yhat1 = model(x)
                    yhat = yhat.cpu().data.numpy()
                    if strategy == 2:
                        yhat1 = torch.max(yhat1[:,:,0:4], 2)[1].tolist()  
                    else:
                        yhat1 = torch.max(yhat1, 2)[1].tolist()  

                    for i in range(len(yhat)):
                        if framecount == video_num[videocount]:
                            framecount = 0
                            videocount += 1
                            result_all.append(result_video)
                            result1_all.append(result1_video)
                            result_video = np.zeros((video_num[videocount]+1)*k//2)
                            result1_video = np.zeros((video_num[videocount]+1)*k//2)
                        if i == 0:
                            result_video[framecount*k//2:(framecount+2)*k//2] = yhat[i]
                            result1_video[framecount*k//2:(framecount+2)*k//2] = yhat1[i]
                        else:
                            result_video[(framecount+1)*k//2:(framecount+2)*k//2] = yhat[i][k//2:]
                            result1_video[(framecount+1)*k//2:(framecount+2)*k//2] = yhat1[i][k//2:]    
                        framecount += 1
                    if framecount == video_num[videocount] and videocount == len(video_num) -1:
                        result_all.append(result_video)
                        result1_all.append(result1_video)

            print('---- Spotting Results ----')
            preds, gt, total_gt_spot, metric_video, metric_final = spotting(final_samples, subject_count, result_all, total_gt_spot, 0.55, metric_final, k_p)
            TP_spot, FP_spot, FN_spot = sequence_evaluation(total_gt_spot, metric_final)
            print('---- Recognition Results ----')
            pred_list, preds_reg, gt_tp_list, pred_window_list, pred_single_list = recognition(result1_all, preds, metric_video, final_emotions, subject_count, pred_list, gt_tp_list, final_samples, pred_window_list, pred_single_list, frame_skip)

    end = time.time()
    print('Total time taken for training & testing: ' + str(end-start) + 's')

    TP_neutral = 0
    FP_neutral = 0
    for i in range(len(pred_list)):
        if pred_list[i] == emotion_type-1:
            if gt_tp_list[i] == -1:
                FP_neutral += 1
            else:
                TP_neutral += 1
        else:
            pred_neutral_list.append(pred_list[i])
            gt_tp_neutral_list.append(gt_tp_list[i])

    return TP_spot, FP_spot, FN_spot, metric_final, pred_list, gt_tp_list,TP_spot-TP_neutral, FP_spot-FP_neutral, FN_spot+TP_neutral, pred_neutral_list, gt_tp_neutral_list