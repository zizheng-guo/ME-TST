# ME-TST: Synergistic Spotting and Recognition of Micro-Expression via Temporal State Transition



## 🔧 Setup

STEP1: `bash setup.sh`

STEP2: `conda activate ME-TST`

STEP3: `pip install -r ./requirements.txt` 



## 💻 Example of Using Pre-trained Models

If you want to run the pre-trained model on SAMMLV, use `python main.py --dataset_name SAMMLV --train False --flow_process False`



## 💻 Examples of Neural Network Training

STEP 1: Download the $CAS(ME)^3$ raw data by asking the paper authors

STEP 2: Modify `main.py; load_excel.py; load_images.py`

STEP 3: Run `python main.py --dataset_name CASME_3 --train True --flow_process True`

