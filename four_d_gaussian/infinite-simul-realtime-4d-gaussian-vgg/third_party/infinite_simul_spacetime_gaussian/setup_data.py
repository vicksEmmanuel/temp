from dataset.preprocess import DatasetPreprocessor
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--videopath", default="/home/featurize/work/infinite-simul-4d-gaussian/data/cut_roasted_beef", type=str)
parser.add_argument("--startframe", default=0, type=int)
parser.add_argument("--endframe", default=300, type=int)
parser.add_argument("--downscale", default=1, type=int)



args = parser.parse_args()
videopath = args.videopath
startframe = args.startframe
endframe = args.endframe
downscale = args.downscale

dataset = DatasetPreprocessor(
    startframe=args.startframe,
    endframe=args.endframe,
    scale=args.downscale
)

dataset(videopath)