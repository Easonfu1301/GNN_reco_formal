from trkr.Sample import HitSample
from trkr.Train import Train
import numpy as np






if __name__ == "__main__":
    # Create a sample object
    sample = HitSample()
    sample.generate_sample(1, 100)
    print(sample)
