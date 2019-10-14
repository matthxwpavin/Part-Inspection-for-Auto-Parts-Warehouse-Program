# Part-Inspection-for-Auto-Parts-Warehouse-Program 

## Overview
### This block diagram show the overview working system.The system detect auto parts and show result on screen for identity of parts.And use result in seaching function on CarSpareParts Program for  trading,store checking,editing,finding and improving product. It split 2 Parts.
#### 1.Object_detection_webcam
#### 2.CarSpareParts Program
### This repository we talk about Object_detection_webcam only.We use TensorFlow’s Object Detection API to train a classifier for multiple object
[![Fed-Tech-Computer-Vision.jpg](https://i.postimg.cc/Y9Qhm09K/Fed-Tech-Computer-Vision.jpg)](https://postimg.cc/BLQqrqHM)


### 1 - Gathering images
in my case I use camera phone to take photos , I gather 10 classes, Each class has 100 sample.
#### 1.Disc Brake Caliper Piston
[![IMG-20190825-103909.jpg](https://i.postimg.cc/KYM98mC1/IMG-20190825-103909.jpg)](https://postimg.cc/5Hf5PW4J)

#### 2.Piston Ring
[![IMG-20190825-112847.jpg](https://i.postimg.cc/85bm7c4z/IMG-20190825-112847.jpg)](https://postimg.cc/jwL79sBp)

#### 3.Turbo Oil Hose
[![IMG-20190825-112001.jpg](https://i.postimg.cc/rpVm8pDJ/IMG-20190825-112001.jpg)](https://postimg.cc/TKsG0fqL)

#### 4.Air Compressor
[![IMG-20190928-140011-1.jpg](https://i.postimg.cc/Y0dBkQsN/IMG-20190928-140011-1.jpg)](https://postimg.cc/bSSB9S8d)

#### 5.Cylinder Liner
[![IMG-20190928-132133-1.jpg](https://i.postimg.cc/wBpff7W9/IMG-20190928-132133-1.jpg)](https://postimg.cc/XrskX7SP)

#### 6.Air Filter
[![IMG-20190928-133623.jpg](https://i.postimg.cc/mDh5MKX9/IMG-20190928-133623.jpg)](https://postimg.cc/WFL8cYPp)

#### 7.Alternator Vac Pump Inlet Oil Hose
[![IMG-20190928-131137.jpg](https://i.postimg.cc/nhZSTXSp/IMG-20190928-131137.jpg)](https://postimg.cc/zVtkGG1c)

#### 8.Timing Belt
[![IMG-20190928-133156-1.jpg](https://i.postimg.cc/wBnhfPxL/IMG-20190928-133156-1.jpg)](https://postimg.cc/8FBJsZ6z)

#### 9.Engine Oil
[![IMG-20190928-133824-1.jpg](https://i.postimg.cc/J7NbxPfH/IMG-20190928-133824-1.jpg)](https://postimg.cc/vDmgHLHG)

#### 10.Hexagon Head Bolt
[![IMG-20190928-135403.jpg](https://i.postimg.cc/d0ZCyBR1/IMG-20190928-135403.jpg)](https://postimg.cc/nsZM8qPb)


### 2 - Convert extensions 
convert all images extensions to `xx.jpg` , using Format Factory http://www.pcfreetime.com/

### 3 - Label images 
[![Untitled.jpg](https://i.postimg.cc/Dw9r5Hws/Untitled.jpg)](https://postimg.cc/hX0QtCxP)
using `labelImg` https://github.com/tzutalin/labelImg
unzip labelImg\
run cmd and go to labelImg dir
```
conda install pyqt=5 
pyrcc5 -o resources.py resources.qrc
python labelImg.py

  ```
   
### 4 - Split images manually randomly
all my samples = 1000 ,700 for train ,300 for test

### 5 - Download  `faster_rcnn_inception_v2_coco`
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md

### 6 - Create new directory 
`C:\tensorflow-gpu\models\research\object_detection`
open cmd
```
cd C:\tensorflow-gpu\models\research\object_detection
mkdir images
mkdir inference_graph
mkdir training
```

### 7 - Configure environment variable

Configure PYTHONPATH environment variable

PYTHONPATH variable must be created that points to the directories
\models \
\models\research \
\models\research\slim  
##### NOTE : every time you run your project must add this lines
```
set PYTHONPATH=C:\tensorflow-gpu\models;C:\tensorflow-gpu\models\research;C:\tensorflow-gpu\models\research\slim
echo %PYTHONPATH%
set PATH=%PATH%;PYTHONPATH
echo %PATH%
```
### 8 - Compile Protobufs
Protobuf (Protocol Buffers) libraries must be compiled , it used by TensorFlow to configure model and training parameters
Open Anaconda Prompt and go to `C:\tensorflow-gpu\models\research`
```
protoc --python_out=. .\object_detection\protos\anchor_generator.proto .\object_detection\protos\argmax_matcher.proto .\object_detection\protos\bipartite_matcher.proto .\object_detection\protos\box_coder.proto .\object_detection\protos\box_predictor.proto .\object_detection\protos\eval.proto .\object_detection\protos\faster_rcnn.proto .\object_detection\protos\faster_rcnn_box_coder.proto .\object_detection\protos\grid_anchor_generator.proto .\object_detection\protos\hyperparams.proto .\object_detection\protos\image_resizer.proto .\object_detection\protos\input_reader.proto .\object_detection\protos\losses.proto .\object_detection\protos\matcher.proto .\object_detection\protos\mean_stddev_box_coder.proto .\object_detection\protos\model.proto .\object_detection\protos\optimizer.proto .\object_detection\protos\pipeline.proto .\object_detection\protos\post_processing.proto .\object_detection\protos\preprocessor.proto .\object_detection\protos\region_similarity_calculator.proto .\object_detection\protos\square_box_coder.proto .\object_detection\protos\ssd.proto .\object_detection\protos\ssd_anchor_generator.proto .\object_detection\protos\string_int_label_map.proto .\object_detection\protos\train.proto .\object_detection\protos\keypoint_box_coder.proto .\object_detection\protos\multiscale_anchor_generator.proto .\object_detection\protos\graph_rewriter.proto
```
```
(tensorflow-gpu) C:\tensorflow-gpu\models\research> python setup.py build
(tensorflow-gpu) C:\tensorflow-gpu\models\research> python setup.py install
```

### 9 - Generate Training Data
TFRecords is an input data to the TensorFlow training model
creat `.csv` files from `.xml` files 
```
cd C:\tensorflow-gpu\models\research\object_detection
python xml_to_csv.py
```
This creates a `train_labels.csv` and `test_labels.csv` file in the `\object_detection\images` folder.
```
python generate_tfrecord.py --csv_input=images\train_labels.csv --image_dir=images\train --output_path=train.record
python generate_tfrecord.py --csv_input=images\test_labels.csv --image_dir=images\test --output_path=test.record
```
### 10 - Edit `generate_tfrecord.py`
edit `generate_tfrecord.py` and put classes names
[![Untitled1.jpg](https://i.postimg.cc/CKzJ5k6M/Untitled1.jpg)](https://postimg.cc/XZ69DGbt)

### 11 - Create a label map and edit the training configuration file.
create `labelmap.pbtxt` to `\training` dir

edit it to classese
```
#!/usr/bin/env python
# -*- coding: utf-8 -*- 
item {
  id: 1
  name: 'Disc Brake Caliper Piston'
}

item {
  id: 2
  name: 'Piston Ring'
}

item {
  id: 3
  name: 'Turbo Oil Hose'
}

item {
  id: 4
  name: 'Air Compressor'
}

item {
  id: 5
  name: 'Cylinder Liner'
}

item {
  id: 6
  name: 'Air Filter'
}

item {
  id: 7
  name: 'Alternator Vac Pump Inlet Oil Hose'
}

item {
  id: 8
  name: 'Timing Belt'
}

item {
  id: 9
  name: 'Engine Oil'
}

item {
  id: 10
  name: 'Hexagon Head Bolt'
}
```

### 12 - Configure object detection tranning pipeline
copy `faster_rcnn_inception_v2_pets.config`
paste it in  `\training` dir and edit it 


#### a - 
 In the `model` section change `num_classes` to number of different classes,in my case = `10`

#### b - 
 fine_tune_checkpoint : `C:/tensorflow-gpu/models/research/object_detection/faster_rcnn_inception_v2_coco_2018_01_28/model.ckpt`

#### c - 
In the `train_input_reader` section change `input_path` and `label_map_path` as : <br/>
Input_path : `C:/tensorflow-gpu/models/research/object_detection/train.record` <br/>
Label_map_path: `C:/tensorflow-gpu/models/research/object_detection/training/labelmap.pbtxt`

#### d - 
In the `eval_config` section change `num_examples` as : <br/>
Num_examples = number of  files in   `\images\test` directory.

#### e -
In the `eval_input_reader` section change `input_path` and `label_map_path` as :<br/>
Input_path : `C:/tensorflow-gpu/models/research/object_detection/test.record` <br/>
Label_map_path: `C:/tensorflow-gpu/models/research/object_detection/training/labelmap.pbtxt`

### 13 - Run the Training
```
python legacy/train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/faster_rcnn_inception_v2_pets.config
```

### 14 - Tensorboard 
in cmd type `(tensorflow-gpu) C:\tensorflow-gpu\models\research\object_detection>tensorboard --logdir=training` to monitor the training
[![Untitled2.jpg](https://i.postimg.cc/6pjN4str/Untitled2.jpg)](https://postimg.cc/yDZG2bJx)


### 15 - Export Inference Graph
 training is complete(loss below `0.05`) ,the last step is to generate the frozen inference graph (.pb file)
change “XXXX” in “model.ckpt-XXXX” should be replaced with the highest-numbered .ckpt file in the training folder:

```
python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/faster_rcnn_inception_v2_pets.config --trained_checkpoint_prefix training/model.ckpt-XXXX --output_directory inference_graph
```
### 16. Use Trained Object Detection Classifier
The object detection classifier is all ready to go! I’ve written Python scripts to test it out on an image, video, or webcam feed.

Before running the Python scripts, we need to modify the NUM_CLASSES variable in the script to equal the number of classes We want to detect. (For my auto parts, there are ten classes we want to detect, so NUM_CLASSES = 10.)

[![Untitled.jpg](https://i.postimg.cc/ncgn3jXs/Untitled.jpg)](https://postimg.cc/WhmR346j)

[![Untitled1.jpg](https://i.postimg.cc/T2Wxp27f/Untitled1.jpg)](https://postimg.cc/LhRwWSFb)

[![Untitled2.jpg](https://i.postimg.cc/MGhs4wym/Untitled2.jpg)](https://postimg.cc/CZsHBWrR)

[![Untitled3.jpg](https://i.postimg.cc/gJVKTZcG/Untitled3.jpg)](https://postimg.cc/BLvD8t1z)

[![Untitled4.jpg](https://i.postimg.cc/YCvzwV2n/Untitled4.jpg)](https://postimg.cc/bZjtxmtk)

[![Untitled5.jpg](https://i.postimg.cc/q7fyHv5q/Untitled5.jpg)](https://postimg.cc/fVKJxDZQ)

[![Untitled6.jpg](https://i.postimg.cc/7PQCyLC4/Untitled6.jpg)](https://postimg.cc/V5nsqmBZ)

[![Untitled7.jpg](https://i.postimg.cc/fLRbG4Dk/Untitled7.jpg)](https://postimg.cc/5HD1zRHM)

[![Untitled8.jpg](https://i.postimg.cc/rp8y46dg/Untitled8.jpg)](https://postimg.cc/K1HyXHDg)

[![Untitled9.jpg](https://i.postimg.cc/nzNCBRfz/Untitled9.jpg)](https://postimg.cc/7GMHyN68)
