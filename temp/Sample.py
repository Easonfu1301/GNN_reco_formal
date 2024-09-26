import numpy as np
from temp.Sample_gen.gen_sample import *


class Sample:
    def __init__(self, N_track):
        self.N_track = N_track
        self.samples = []

        pass

    def warning(self, message):
        print(f"\033[93m{message}\033[0m")

    def Log(self, text):
        print(f"\033[94m{text}\033[0m")

    def generate_sample(self, number=1):
        samples = []
        for i in range(number):
            samples.append(gen_sample(self.N_track))
        self.samples = samples

    def visualize_sample(self, index=-1):
        if index == -1:
            self.warning("Have not implemented yet")
            return False

        sample = self.get_sample(index)
        self.Log(f"Visualizing sample {index}")
        visualzie_sample(sample)
        pass

    def get_sample(self, index=-1):
        if index == -1:
            self.warning("Have not implemented yet")
            return False
        return self.samples[index]

    def load_sample(self):
        pass

    def export_sample(self):
        pass


if __name__ == '__main__':
    sample = Sample(1000)
    sample.generate_sample(1)
    sample.visualize_sample(0)
