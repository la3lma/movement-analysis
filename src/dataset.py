from numpy import fft
from matplotlib import pyplot
from pickle import BINSTRING
import math

class dataset:
    def __init__(self, filename):
        with open(filename) as file:
            self.data = []
            lines = [line for line in file]
            for line in lines[1:]:
                parts = [float(measurement.strip()) for measurement in line.split(';')]
                self.data.append(parts)
                
    def get_frequencies(self):
        num_seconds = float(self.data[-1][0] - self.data[0][0]) / float(1000)
        samples_per_second = len(self.data) / num_seconds
        num_samples = len(self.data)
        oscilations_per_sample = [float(oscilations) / num_samples for oscilations in range(0, num_samples)]
        return [ops * samples_per_second for ops in oscilations_per_sample]
    
    def get_buckets(self, num_buckets, hertz_cutoff=float(5)):
        one_dimentional = [column[2] for column in walking.data]
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
        pyplot.plot([i * width for i in range(num_buckets)], buckets)
        pyplot.show()
        return buckets
        
                
if __name__ == '__main__':
    walking = dataset('../datasets/jan-walking-30s-v1.csv')
    jan = dataset('../datasets/jan-walking-30s-v1.csv')
    
    walking.get_buckets(40)
    jan.get_buckets(40)
    
    
    