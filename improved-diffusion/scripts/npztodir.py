import os
import numpy as np
from PIL import Image
import argparse


parser = argparse.ArgumentParser(
prog = "NPZ to Image directory",
description = "Converts an npz file containing images into an image directory")
parser.add_argument('inputfile')
parser.add_argument('outputdir')

args = parser.parse_args()

outdir = args.outputdir

images = np.load(args.inputfile)["arr_0"]

os.makedirs(outdir)

for i in range(len(images)):
    image = Image.fromarray(images[i])
    image.save(os.path.join(outdir, f"{i}.png"))
