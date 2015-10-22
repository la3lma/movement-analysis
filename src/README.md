# classifier

This app creates the model and classifies data. To make it run together with the node.js app, run:

```bash
$ python dataset.py --model ../models/1s_6sps.pkl --data=../data-gathering/raw-data/
```

Now it will monitor the raw-data folder of the node app, so when data flows in it auto-classifies it. It emits this data to stdout and to `../data-gathering/classification`. The node app has a monitor web app that can show this in a browser so you can show it to other people.

## Model training

There's a bunch of models in ../model directory, use the `--model` flag to pass it in. If you run it without this flag it'll start building a new model (will take long)
