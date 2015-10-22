# movement-analysis

Classification of human movement based on accelerometer / gyro data from mobile phones

* `data-gathering` contains all the code required for collecting live data from smartphones and dump it to disk
* `datasets` contains training and verification data that we use to train the model
* `logs` track model performance over time
* `models` pre-trained models
* `src` application that does the data analysis

The `data-gathering` and `src` application can work together. If you have them running both, you can let src monitor the raw-data folder of data-gathering. Then whenever live data flows in through the node app it'll do live classification of the data. It then writes it back to the node app so that it can be shown to a user in a monitoring web app.
