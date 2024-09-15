import numpy as np

def loso_split_label(y, final_subjects, final_videos, final_samples, final_dataset_spotting, final_emotions):
    emotion_list = ['negative', 'positive', 'surprise', 'others'] #
    #To split the dataset by subjects
    groupsLabel = np.zeros(len(y)) # For spotting
    groupsLabel1 = [] # For recognition
    prevIndex = 0
    countVideos = 0
    # countSamples = 0
    videos_len = []

    #Get total frames of each video
    for video_index in range(len(final_dataset_spotting)):
        videos_len.append(len(final_dataset_spotting[video_index]))

    print('Frame Index for each subject (Spotting):-')
    for subject_index in range(len(final_samples)):
        countVideos += len(final_samples[subject_index])
        index = sum(videos_len[:countVideos])
        groupsLabel[prevIndex:index] = subject_index
        print('Subject', final_subjects[subject_index], ':', prevIndex, '->', index)

        # for video_index in range(len(final_videos[subject_index])):
        #     countSamples += 1
        #     samples_index = sum(videos_len[:countSamples])
        #     print('Video', final_videos[subject_index][video_index], ':', prevIndex, '->', samples_index)
        #     prevIndex = samples_index

        prevIndex = index

    #Get total frames of each video
    print('\nFrame Index for each subject (Recognition):-')
    for subject_index in range(len(final_samples)):
        for video_index in range(len(final_samples[subject_index])):
            for sample_index in range(len(final_samples[subject_index][video_index])):
                if(final_emotions[subject_index][video_index][sample_index] in emotion_list):
                    groupsLabel1.append(subject_index)
        if(subject_index in np.unique(groupsLabel1)):
            print('Subject', final_subjects[subject_index], ':', len(groupsLabel1)-len(final_samples[subject_index]), '->', len(groupsLabel1)-1)
        else:
            print('Subject', final_subjects[subject_index], ':', 'Not available')

    return groupsLabel, groupsLabel1