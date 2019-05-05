import cv2
import numpy as np
import sys


def readImagesAndTimes():
  
  filenames = [
               "image/memorial0061.jpg",
               "image/memorial0062.jpg",
               "image/memorial0063.jpg",
               "image/memorial0064.jpg",
               "image/memorial0065.jpg",
               "image/memorial0066.jpg",
               "image/memorial0067.jpg",
               "image/memorial0068.jpg",
               "image/memorial0069.jpg",
               "image/memorial0070.jpg",
               "image/memorial0071.jpg",
               "image/memorial0072.jpg",
               "image/memorial0073.jpg",
               "image/memorial0074.jpg",
               "image/memorial0075.jpg",
               "image/memorial0076.jpg"
               ]

  images = []
  for filename in filenames:
    im = cv2.imread(filename)
    images.append(im)
  
  return images

if __name__ == '__main__':
  
  # Read images
  print("Reading images ... ")
  
  if len(sys.argv) > 1:
    # Read images from the command line
    images = []
    for filename in sys.argv[1:]:
      im = cv2.imread(filename)
      images.append(im)
    needsAlignment = False
  else :
    # Read example images
    images = readImagesAndTimes()
    needsAlignment = False
  
  # Align input images
  if needsAlignment:
    print("Aligning images ... ")
    alignMTB = cv2.createAlignMTB()
    alignMTB.process(images, images)
  else :
    print("Skipping alignment ... ")
  
  # Merge using Exposure Fusion
  print("Merging using Exposure Fusion ... ");
  mergeMertens = cv2.createMergeMertens()
  exposureFusion = mergeMertens.process(images)

  # Save output image
  print("Saving output ... exposure-fusion.jpg")
  cv2.imwrite("exposure-fusion.jpg", exposureFusion * 255)


