import numpy as np
import torch.nn.functional as F
import torch

def determine_emotion(emotion_type):
    # 4 emotions
    if emotion_type == 4:
        label_dict = { 'negative' : 0, 'positive' : 1, 'surprise' : 2, 'others' : 3}
        emotion_dict = { 'happy': 'positive', 'disgust': 'negative', 'fear': 'negative', 'anger': 'negative', 'sad': 'negative', 'surprise': 'surprise', 'others': 'others' }

    if emotion_type == 5:
        label_dict = { 'negative' : 0, 'positive' : 1, 'surprise' : 2, 'others' : 3, 'neutral' : 4}
        emotion_dict = { 'happy': 'positive', 'disgust': 'negative', 'fear': 'negative', 'anger': 'negative', 'sad': 'negative', 'surprise': 'surprise', 'others': 'others' }

    return label_dict, emotion_dict


def pseudo_labeling(frame_skip, dataset, final_samples, final_emotions, label_dict, emotion_type, k):
    #### Pseudo-labeling
    pseudo_y = []
    pseudo_y1 = []
    video_count = 0 

    for subject_index, subject in enumerate(final_samples):
        for video_index, video in enumerate(subject):
            # samples_arr = []
            pseudo_y_each = [0]*((len(dataset[video_count])+1)*k//2)
            pseudo_y1_each = [emotion_type-1]*((len(dataset[video_count])+1)*k//2)
            for sample_index, sample in enumerate(video):
                onset = sample[0] + 1
                apex = sample[1]
                offset = sample[2] - 1
                start = onset #int((onset+apex)/2)
                end = offset #int((offset+apex)/2)
                for frame_index, frame in enumerate(range(start, end+1)):
                    if frame < len(pseudo_y_each):
                        pseudo_y_each[frame] = 1 # Hard label
                        pseudo_y1_each[frame] = label_dict[final_emotions[subject_index][video_index][sample_index]]
            pseudo_y_group = []
            pseudo_y1_group = []
            for i in range(len(dataset[video_count])):
                pseudo_y_group.append(pseudo_y_each[i*k//2:(i+2)*k//2])
                pseudo_y1_group.append(pseudo_y1_each[i*k//2:(i+2)*k//2])
            pseudo_y1.append(pseudo_y1_group)
            pseudo_y.append(pseudo_y_group)
            video_count+=1
            
    print('Total video:', len(pseudo_y))
    # Integrate all videos into one dataset
    pseudo_y = [y for x in pseudo_y for y in x]
    pseudo_y1 = [y1 for x in pseudo_y1 for y1 in x]
    print('Total frames:', len(pseudo_y))
    # print('Distribution hard label:', Counter(pseudo_y))
    # print('Emotion label:', Counter(pseudo_y1))
    # print('Distribution:', Counter(pseudo_y1))

    return pseudo_y, pseudo_y1

def prepare_spot_data(dataset_name, dataset, final_subjects, final_samples, pseudo_y, pseudo_y1):
    #To split the dataset by subjects
    Y_spot = np.array(pseudo_y)
    Y1_spot = F.one_hot(torch.tensor(pseudo_y1)).numpy()
    videos_len = []
    groupsLabel_spot = Y_spot.copy()
    prevIndex = 0
    countVideos = 0

    #Get total frames of each video
    for video_index in range(len(dataset)):
        videos_len.append(len(dataset[video_index]))

    # print('Frame Index for each subject:-')
    for subject_index in range(len(final_samples)):
        countVideos += len(final_samples[subject_index])
        index = sum(videos_len[:countVideos])
        if dataset_name == "CASME_3":
            groupsLabel_spot[prevIndex:index] = final_subjects[subject_index][-1]
        else:
            groupsLabel_spot[prevIndex:index] = final_subjects[subject_index][1:3]
        # print('Subject', final_subjects[subject_index], ':', prevIndex, '->', index)
        prevIndex = index

    X_spot = []
    for video_index, video in enumerate(dataset):
        X_spot.append(video)
    X_spot = [frame for video in X_spot for frame in video]
    print('\nTotal X:', len(X_spot), ', Total Y:', len(Y_spot), ', Total Y1:', len(Y1_spot))

    return X_spot, Y_spot, Y1_spot, groupsLabel_spot