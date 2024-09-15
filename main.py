import argparse
from distutils.util import strtobool
from load_images import *
from load_label import *
from load_excel import *
from feature_extraction import *
from prepare_training import *
from train_evaluate import *
from prepare_data import *
from train import *
import pickle
import warnings
warnings.filterwarnings('ignore')

# RANDOM
RANDOM_SEED = 100
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
general_generator = torch.Generator()
general_generator.manual_seed(RANDOM_SEED)
train_generator = torch.Generator()
train_generator.manual_seed(RANDOM_SEED)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def main(config):
    flow_process = config.flow_process
    train = config.train
    dataset_name = config.dataset_name
    note = config.note
    # Setting paramaters
    emotion_type = 4 + 1 # class + netural
    epochs = 30
    lr = 0.0001

    if dataset_name == "CASME_3":
        frame_skip = 1 # Depends on frame rate
        k = 50 # Depends on the average ME interval length
        strategy = 2 # Synergy strategy   0:w/o; 1:recognition mainly; 2:spotting mainly
        ratio = 4 # Depends on dataset size
        batch_size = 256 # Depends on dataset size
    elif dataset_name == "SAMMLV":
        frame_skip = 7
        k = 30
        strategy = 2
        ratio = 0.5
        batch_size = 32

    # Determine Emotion
    label_dict, emotion_dict = determine_emotion(emotion_type)
    codeFinal = load_excel(dataset_name)

    if flow_process:
        print("\n ------ Loading Images ------")
        images, subjects, subjectsVideos = load_images(dataset_name,frame_skip)
        print('\n ------ Loading Ground Truth From Excel ------')
        final_images, final_subjects, final_videos, final_samples, final_emotions = load_label(dataset_name, images, subjects, subjectsVideos, codeFinal, frame_skip)
        print('\n ------ Feature Extraction & Pre-processing ------')
        final_dataset_spotting = feature_extraction_spotting(dataset_name, final_images, k)
        pickle.dump(final_dataset_spotting, open("cache/"+dataset_name + "_dataset.pkl", "wb")) 
    else:
        subjects, subjectsVideos = load_information(dataset_name,frame_skip)
        images = None
        final_images, final_subjects, final_videos, final_samples, final_emotions = load_label(dataset_name, images, subjects, subjectsVideos, codeFinal, frame_skip)
        if dataset_name == "CASME_3":
            final_dataset_spotting = []
            for i in range(5):
                final_dataset_spotting.extend(pickle.load( open("cache/"+dataset_name + "_dataset_" + str(i+1) +".pkl", "rb" ) )) # CASME_3 is too large and is divided into five parts
        elif dataset_name == "SAMMLV":
            final_dataset_spotting = pickle.load( open("cache/"+dataset_name + "_dataset.pkl", "rb" ) )

    k_p = cal_k_p(dataset_name, final_samples)
    print('\n ------ Preparing Label ------')
    pseudo_y, pseudo_y1 = pseudo_labeling(frame_skip, final_dataset_spotting, final_samples, final_emotions, label_dict, emotion_type, k)
    X_spot, Y_spot, Y1_spot, groupsLabel_spot = prepare_spot_data(dataset_name, final_dataset_spotting, final_subjects, final_samples, pseudo_y, pseudo_y1)
    groupsLabel_spot, groupsLabel_recog = loso_split_label(np.array(pseudo_y), final_subjects, final_videos, final_samples, final_dataset_spotting, final_emotions)

    # Training & Evaluation
    print('\n ------ Training ------')
    TP_spot, FP_spot, FN_spot, metric_final, pred_list, gt_tp_list, TP_spot_neutral, FP_spot_neutral, FN_spot_neutral, pred_neutral_list, gt_tp_neutral_list = train_model(train, X_spot, Y_spot, Y1_spot, groupsLabel_spot, groupsLabel_recog, final_dataset_spotting, final_subjects, final_samples, final_videos, final_emotions, emotion_type, epochs, lr, batch_size, dataset_name, k_p, k, ratio, frame_skip, strategy, note)
    
    print('\n\n ------------------------ Evaluation ------------------------')
    final_evaluation(TP_spot, FP_spot, FN_spot, dataset_name, pred_list, emotion_type ,gt_tp_list)
    if strategy == 1:
        print('\n\n ------------------------ Evaluation with strategy_1 ------------------------')
        final_evaluation(TP_spot_neutral, FP_spot_neutral, FN_spot_neutral, dataset_name, pred_neutral_list, emotion_type ,gt_tp_neutral_list)
    print('\n ------ Completed ------')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # input parameters
    parser.add_argument('--dataset_name', type=str, default='SAMMLV')
    parser.add_argument('--train', type=strtobool, default=True)
    parser.add_argument('--flow_process', type=strtobool, default=False) 
    parser.add_argument('--note', type=str, default='note') 
    
    config = parser.parse_args()
    main(config)
