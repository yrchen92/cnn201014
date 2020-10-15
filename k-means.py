import argparse
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd

class Point:
    def __init__(self, lat, lon):
        self.lat = lat
        self.lon = lon
    def e_dist(self, p):
        return math.sqrt(math.pow(self.lat - p.lat, 2.0)
            + math.pow(self.lon - p.lon, 2.0))

class KMeans:
    def __init__(self, geo_locs, k):
        self.geo_locs = geo_locs
        self.k = k
        self.cluster = None
        self.means = []

    def init_means(self, p):
        ps_ = [point for point in self.geo_locs]
        p_ = random.choice(ps_)
        clusters = {}
        clusters.setdefault(0, []).append(p_)
        ps_.remove(p_)

        for i in range(self.k):
            self.means.append(random.choice(self.geo))


    def fit(self, plot_flag):
        if len(self.geo_locs) < self.k:
            return -1
        self.init_means()


def main(args):
    geo_locs = []
    df = pd.read_csv(args.input)
    for i, r in df.iterrows():
        loc = Point(float(r['LAT'], float(r['LON'])))
        geo_locs.append(loc)
    model = KMeans(geo_locs, args.k)




if __name__ == "__main__":
    parser = argparse.Argumentation(description="k-means")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--input", type=str, default="data/input")
    parser.add_argument("--output", type=str, default="data/output")
    args = parser.parser_args()
    main(args)
#https://github.com/kjahan/k-means