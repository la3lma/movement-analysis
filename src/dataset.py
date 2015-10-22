'''Usage: dataset.py [--model=x --data=feed] [--plot] [--pca]

Options:
    --model=m       Path to trained model, omit to rebuild the model
    --data=feed     Path to data file to monitor for live data.
                    If you pass in a folder, it'll pick the last touched file in the folder.
    --plot          Show confusion matrix in a separate window
    --pca           Apply PCA to the datasets before doing classifications
'''
import time
from numpy import fft
from numpy import array
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from pickle import BINSTRING
import math
import os, time, sys
from sklearn import svm
from sklearn import grid_search
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib
import docopt

class sample_file:
    def __init__(self, filename):
        self.filename = filename
        self.timestamps = []
        raw_data = []
        with open(filename) as file:
            lines = [line for line in file]
            for line in lines[2:]:
                parts = [float(measurement.strip()) for measurement in line.split(';')]
                self.timestamps.append(parts[0])
                raw_data.append(parts[1:])
        self.data = []
        self.gyros = []
        self.accelerations = []
        for i in range(len(raw_data) - 1):
            current = raw_data[i]
            next = raw_data[i + 1]
            gyro = []
            for j in range(3):
                delta = next[j] - current[j]
                if abs(delta) > 180:
                    delta -= 360 * delta / abs(delta)
                gyro.append(delta)
            gyro = [next[j] - current[j] for j in range(3)]
            acceleration = current[3:6]
            self.data.append([gyro[0], gyro[1], gyro[2], acceleration[0], acceleration[1], acceleration[2]])
            self.gyros.append(gyro)
            self.accelerations.append(acceleration)

    def get_frequencies(self):
        num_seconds = float(self.timestamps[-2] - self.timestamps[0]) / float(1000)
        samples_per_second = len(self.data) / num_seconds
        num_samples = len(self.data)
        oscilations_per_sample = [float(oscilations) / num_samples for oscilations in range(0, num_samples)]
        return [ops * samples_per_second for ops in oscilations_per_sample]

    def get_buckets(self, first, last, num_buckets, hertz_cutoff=float(5)):

        if arguments['--pca']:
            # Transform all of the original data to be a single component
            # along the first principal component
            pca = PCA(n_components=1, copy=True, whiten=True)
            numpy_data = array(self.data)
            transformed_dataset = PCA.fit_transform(pca, numpy_data)
            print(pca.explained_variance_ratio_)
            slice=transformed_dataset[first:last]
        else:
            # Otherwise just pick the beta component from the gyro data
            slice = self.data[first:last]
            slice = [column[1] for column in slice]

        transformed = fft.fft(slice)
        absolute = [abs(complex) for complex in transformed]

        frequencies = self.get_frequencies()

        buckets = [0 for i in range(num_buckets)]
        width = hertz_cutoff / num_buckets
        for i in range(1, len(absolute)):
            index = int(frequencies[i] / width)
            if index >= num_buckets:
                break;
            buckets[index] += absolute[i]
        return buckets

    def get_samples(self):
        result = []
        segmentsize=30
        # Reduce this to very little to get very large trainingsets
        stride=5
        noOfBuckets=40
        for  start in range(0, len(self.data) - segmentsize, stride):
            if start + segmentsize <= len(self.data):
                segments_buckets = self.get_buckets(start, start + segmentsize, noOfBuckets)
                result.append(segments_buckets)
        return result

    def keep_last_lines(self, num_lines):
        self.data = self.data[-num_lines:]

class dataset:
    def __init__(self, foldername, filters = {'dancing': 0, 'walking': 1, 'sitting':2}):
        self.data = []
        self.target = []
        self.activities = []
        noOfSamples = 0
        for activity, number in filters.iteritems():
            samples = get_samples(foldername, filter=activity)
            for sample in samples:
                noOfSamples +=1
                self.data.append(sample)
                self.target.append(number)
                self.activities.append(activity)
        print "foldername= ", foldername, "noOfSamples= ", noOfSamples


def get_samples(foldername, filter=None):
    samples = []
    for file in os.listdir(foldername):
        if filter and file.find(filter) == -1:
            continue
        for sample in sample_file(foldername + '/' + file).get_samples():
            samples.append(sample)

    return samples

if __name__ == '__main__':
    arguments = docopt.docopt(__doc__)
    filters = {'dancing': 0, 'walking': 1, 'sitting':2}
    if arguments['--model']:
        clf = joblib.load(arguments['--model'])
    else:
        training = dataset('../datasets/training', filters)

        svr = svm.SVC()
        exponential_range = [pow(10, i) for i in range(-4, 1)]
        parameters = {'kernel':['linear', 'rbf'], 'C':exponential_range, 'gamma':exponential_range}
        clf = grid_search.GridSearchCV(svr, parameters, n_jobs=2, verbose=True)
        clf.fit(training.data, training.target)
        joblib.dump(clf, '../models/1s_6sps.pkl')
        print clf

    print 'best_score:', clf.best_score_, 'best C:', clf.best_estimator_.C, 'best gamma:', clf.best_estimator_.gamma
    validation = dataset('../datasets/validation')

    predicted = clf.predict(validation.data)
    truedata =  map(lambda x: filters[x], validation.activities)
    # http://scikit-learn.org/stable/auto_examples/calibration/plot_calibration_curve.html
    precision=precision_score(truedata, predicted, average='macro')
    recall=recall_score(truedata, predicted, average='macro')

    print "predicted = ", predicted
    print "truedata  = ", truedata
    print "macro precision = ", precision
    print "macro recall = ", recall

    # Write precision/recall to a file so that we can se how
    # the precision of the project's output improves over time.
    ts = time.time()
    record = str(ts) + ", " +  str(precision) + ", " +  str(recall) + "\n";
    with open("../logs/precision-recall-time-evolution.csv", "a") as myfile:
        myfile.write(record)

    # Compute confusion matrix
    cm = confusion_matrix(truedata, predicted)
    print "confusion:"
    print(cm)

    if arguments['--plot']:
        # Show confusion matrix in a separate window
        plt.matshow(cm)
        plt.title('Confusion matrix')
        plt.colorbar()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

    data_feed = arguments['--data']

    print "Monitoring file " + data_feed

    last_touched = 0
    if data_feed:
        while True:
            try:
                if (os.path.isdir(data_feed)):
                    #max(os.listdir('.'), )
                    all_files_in_df = map(lambda f: os.path.join(data_feed, f), os.listdir(data_feed))
                    data_file = max(all_files_in_df, key = os.path.getmtime)
                else:
                    data_file = data_feed

                # get last modified time
                stat_result = os.stat(data_file)
                # file changed?
                if stat_result.st_mtime != last_touched:
                        sample = sample_file(data_file)
                        sample.keep_last_lines(180)
                        samples = sample.get_samples()
                        sys.stdout.write("Classification: ")

                        pr = clf.predict(samples)
                        with open('../data-gathering/classification', 'w') as f:
                            f.truncate()
                            f.write(str(pr))

                        print pr
                else:
                    print "File didn't change"

                last_touched = stat_result.st_mtime
            except:
                print "Unexpected error", sys.exc_info()[0]

            time.sleep(1)
