# movement-analysis

Classification of human movement based on accelerometer / gyro data from mobile phones

* `data-gathering` contains all the code required for collecting live data from smartphones and dump it to disk
* `datasets` contains training and verification data that we use to train the model
* `logs` track model performance over time
* `models` pre-trained models
* `src` application that does the data analysis

## Test the live classifier yourself

First set up your environment:

1. Install node.js and python 2.7
2. Go into `data-gathering` folder and run `npm install`
3. Go into `src` folder and run `easy install sklearn docopt`

The application contains of three parts: a website that runs on a phone and gets data; a node.js app that runs on computer and receives the data; a python app that classifies the data. Get two terminals and...

In terminal 1:

```bash
$ cd src
$ python dataset.py --model ../models/1s_6sps.pkl --data=../data-gathering/raw-data/
```

In terminal 2:

```bash
$ cd data-gathering
$ node server.js
```

Now open the monitoring application on your computer, so you can see the live classification, at http://localhost:9321/server.

Next you want to start gathering data. Make sure your phone and computer are on the same wifi network, and look up the IP of your computer. 

1. Navigate to http://YOURIP:9321 on your mobile phone
2. Press the 'Start measurement' button
3. Put the phone (with the screen down) in your left front pocket
4. See data flowing in! (In terminal 2 it should say 'Start measurement')

After a few seconds the classifier starts showing data in your web browser!

**Note** Default measurement time is only 30s, which probably not enough to demo. Change it in data-gathering/client/accelerometer-position/index.html (copy the beepWithTimeout lines).
