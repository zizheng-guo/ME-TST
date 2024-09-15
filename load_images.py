import glob
import natsort
import numpy as np
import cv2

def load_images(dataset_name, frame_skip):
  images = []
  subjects = []
  subjectsVideos = []
  print('Loading images from dataset', dataset_name)

  # For dataset CASME_3
  if(dataset_name == 'CASME_3'):
      for i, dir_sub in enumerate(natsort.natsorted(glob.glob("../dataset/"+ dataset_name + "/data/*"))): 
        print('Subject: ' + dir_sub.split('/')[-1])
        subjects.append(dir_sub.split('/')[-1].split('.')[-1])
        subjectsVideos.append([])
        for dir_sub_vid in natsort.natsorted(glob.glob(dir_sub + "/*")):
          subjectsVideos[-1].append(dir_sub_vid.split('/')[-1]) # Ex:'CASME_sq/data/spNO.1/a'
          image = []
          count = 0
          for dir_sub_vid_img in natsort.natsorted(glob.glob(dir_sub_vid  +"/color/*.jpg")): 
            while int(dir_sub_vid_img.split('/')[-1].split('.')[0]) != count:
              image.append(cv2.imread(dir_sub_vid_img, 0)) # 0 / 1
              count += 1
          print('Done -> ' + dir_sub_vid.split('/')[-1])
          images.append(np.array(image))

  # For dataset SAMMLV
  elif(dataset_name == 'SAMMLV'):
      for i, dir_vid in enumerate(natsort.natsorted(glob.glob("../dataset/"+ dataset_name + "/SAMM_longvideos/*"))): 
        subject = dir_vid.split('/')[-1].split('_')[0]
        if (subject not in subjects): #Only append unique subject name
          subjects.append(subject)
          subjectsVideos.append([])
        subjectsVideos[-1].append(dir_vid.split('/')[-1])
        image = []
        i = 0
        for dir_vid_img in natsort.natsorted(glob.glob(dir_vid + "/*.jpg")):
          if i % frame_skip ==0:
            image.append(cv2.imread(dir_vid_img, 0))
          i += 1
        image = np.array(image)
        print('Done -> ' + dir_vid.split('/')[-1])
        images.append(image)
  

  print('Loading images from dataset', dataset_name, 'All Done')
  return images, subjects, subjectsVideos


def load_information(dataset_name, frame_skip):
  subjects = []
  subjectsVideos = []

  print('Loading images from dataset', dataset_name)

  # For dataset CASME_3
  if(dataset_name == 'CASME_3'):
      for i, dir_sub in enumerate(natsort.natsorted(glob.glob("../dataset/"+ dataset_name + "/data/*"))):
        print('Subject: ' + dir_sub.split('/')[-1])
        subjects.append(dir_sub.split('/')[-1].split('.')[-1])
        subjectsVideos.append([])
        for dir_sub_vid in natsort.natsorted(glob.glob(dir_sub + "/*")):
          subjectsVideos[-1].append(dir_sub_vid.split('/')[-1]) # Ex:'CASME_sq/data/spNO.1/a'
          print('Done -> ' + dir_sub_vid.split('/')[-1])

  # For dataset SAMMLV
  elif(dataset_name == 'SAMMLV'):
      for i, dir_vid in enumerate(natsort.natsorted(glob.glob("../dataset/"+ dataset_name + "/SAMM_longvideos/*"))): 
        subject = dir_vid.split('/')[-1].split('_')[0]
        if (subject not in subjects): #Only append unique subject name
          subjects.append(subject)
          subjectsVideos.append([])
        subjectsVideos[-1].append(dir_vid.split('/')[-1])
        print('Done -> ' + dir_vid.split('/')[-1])

  print('Loading images from dataset', dataset_name, 'All Done')
  return subjects, subjectsVideos