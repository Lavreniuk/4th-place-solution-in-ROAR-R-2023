# ROAD-R Challenge 
It is [4nd place solution](https://eval.ai/web/challenges/challenge-page/2081/leaderboard/4900) of the DeepDrive team (task2) in [ROAD-R Challenge for Neurips2023](https://sites.google.com/view/road-r/).
The code is built on top of [3D-RetinaNet for ROAD](https://github.com/gurkirt/road-dataset).

ROAD-R contains comprising 22 (~8-minutes long) videos annotated with road events together with a set of 243 requirements expressing hard facts about the world that are commonly known by humans (e.g., “a traffic light cannot be red and green at the same time”).

The first task requires developing models for scenarios where only little annotated data is available at training time. 
More precisely, only 3 out of 15 videos (from the training partition train_1 of the ROAD-R dataset) are used for training the models in this task.
The videos' ids are: 2014-07-14-14-49-50_stereo_centre_01, 2015-02-03-19-43-11_stereo_centre_04, and 2015-02-24-12-32-19_stereo_centre_04.

The second task requires that the models' predictions are compliant with the 243 requirements provided in `constraints/requirements.txt`.

## Table of Contents
- <a href='#dep'>Dependencies and data preparation</a>
- <a href='#training'>Training</a>
- <a href='#testing'>Testing</a>
- <a href='#prostprocessing'>Post-processing</a>



## Dependencies and data preparation
For the dataset preparation and packages required to train the models, please see the [Requirements](https://github.com/gurkirt/3D-RetinaNet#requirements) section from 3D-RetinaNet for ROAD as well as use local requirements.txt file.

To download the pretrained weights, please see the end of the [Performance](https://github.com/gurkirt/3D-RetinaNet#performance) section from 3D-RetinaNet for ROAD.

## Training

Task1:
```
DATA_ROOT="${HOME}/roadr/road-dataset/"
EXPDIR="${HOME}/roadr/experiments/"
MODEL_PATH="${HOME}/roadr/ROAD-R-2023-Challenge/kinetics-pt/"
TASK=1
EXP_ID="task1"

python main.py ${TASK} ${DATA_ROOT} ${EXPDIR}/${EXP_ID} ${MODEL_PATH} --MODE="train" --VAL_STEP=10 --LR=0.0041 --MAX_EPOCHS=130 --BATCH_SIZE 2 --MILESTONES 100,125
```

Task2:
```
DATA_ROOT="${HOME}/roadr/road-dataset/"
EXPDIR="${HOME}/roadr/experiments/"
MODEL_PATH="${HOME}/roadr/ROAD-R-2023-Challenge/kinetics-pt/"
TASK=2
EXP_ID="task2"

python main.py ${TASK} ${DATA_ROOT} ${EXPDIR}/${EXP_ID} ${MODEL_PATH} --MODE="train" --VAL_STEP=2 --LR=0.0041 --MAX_EPOCHS=30 --MILESTONES=20,25 --MODEL_TYPE I3D
```

## Testing 

Task1:
```
EXP_ID="task1"
EXP_NAME="../experiments/task1road/logic-ssl_cache_None_0.0/resnet50RCGRU512-Pkinetics..."
python main.py ${TASK} ${DATA_ROOT} ${EXPDIR}/${EXP_ID} ${MODEL_PATH} --MODE="gen_dets" --TEST_SUBSETS=test --EVAL_EPOCHS=130 --EXP_NAME=${EXP_NAME}
```

Task2:
```
EXP_ID="task2"
EXP_NAME="../experiments/task2road/logic-ssl_cache_None_0.0/resnet50I3D512-Pkinetics..."
python main.py ${TASK} ${DATA_ROOT} ${EXPDIR}/${EXP_ID} ${MODEL_PATH} --MODE="gen_dets" --TEST_SUBSETS=test --EVAL_EPOCHS=30 --EXP_NAME=${EXP_NAME} --MODEL_TYPE I3D
```

## Post-processing

From roadr/ROAD-R-2023-Challenge/postprocessing for Task2:
```
python post_processing_raw.py --file_path ../../pred_detections-30-08-50_test.pkl --requirements_path ../constraints/WDIMACS_requirements.txt --threshold 0.4
```
