# tf-utils
Utilities for TensorFlow

## tfod_traindata_generator.py

Train data generator for TensorFlow Object Detection API.

```
python tfod_traindata_generator.py target/fruits-TFRecords-export 0.8
```

|args||
|-|-|
|target/fruits-TFRecords-export|Exported directory by *VoTT TensorFlow Record*. |
|0.8|TFRecord files of 80% are merged into **train.tfrecord**, and 20% are merged into **val.tfrecord**.|

tfod_traindata_generator.py outputs **train.tfrecord** and **val.tfrecord**.
