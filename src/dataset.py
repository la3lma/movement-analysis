from numpy import fft
from matplotlib import pyplot
from pickle import BINSTRING
import math
import os
from sklearn import svm

class sample_file:
    def __init__(self, filename):
        self.filename = filename
        with open(filename) as file:
            self.data = []
            lines = [line for line in file]
            for line in lines[2:]:
                parts = [float(measurement.strip()) for measurement in line.split(';')]
                self.data.append(parts)
                
    def get_frequencies(self):
        num_seconds = float(self.data[-1][0] - self.data[0][0]) / float(1000)
        samples_per_second = len(self.data) / num_seconds
        num_samples = len(self.data)
        oscilations_per_sample = [float(oscilations) / num_samples for oscilations in range(0, num_samples)]
        return [ops * samples_per_second for ops in oscilations_per_sample]
    
    def get_buckets(self, num_buckets, hertz_cutoff=float(5)):
        one_dimentional = [column[2] for column in self.data]
        transformed = fft.fft(one_dimentional)
        absolute = [abs(complex) for complex in transformed]
        
        frequencies = self.get_frequencies()
        
        buckets = [0 for i in range(num_buckets)]
        width = hertz_cutoff / num_buckets
        for i in range(1, len(absolute)):
            index = int(frequencies[i] / width)
            if index >= num_buckets:
                break;
            buckets[index] += absolute[i]
#         pyplot.plot([i * width for i in range(num_buckets)], buckets, label = self.filename)
        return buckets
    
    def get_samples(self):
        buckets = self.get_buckets(40)
        return [buckets]
        
class dataset:
    def __init__(self, foldername, filters = {'dancing': 0, 'walking': 1, 'sitting':2}):
        self.data = []
        self.target = []
        self.activities = []
        for activity, number in filters.iteritems():
            samples = get_samples(foldername, filter=activity)
            for sample in samples:
                self.data.append(sample)
                self.target.append(number)
                self.activities.append(activity)
            
def get_samples(foldername, filter=None):
    samples = []
    for file in os.listdir(foldername):
        if filter and file.find(filter) == -1:
            continue
        for sample in sample_file(foldername + '/' + file).get_samples():
            samples.append(sample)
        
    return samples
          
if __name__ == '__main__':
    filters = {'dancing': 0, 'walking': 1, 'sitting':2}
    training = dataset('../datasets/training', filters)
    
    clf = svm.SVC(gamma=0.001, C=100.)
    clf.fit(training.data, training.target)

    validation = dataset('../datasets/validation')
    for i in range(len(validation.data)):
        print 'expected', filters[validation.activities[i]], 'got', clf.predict(validation.data[i])
#     pyplot.legend()
#     pyplot.show()
    
    
    