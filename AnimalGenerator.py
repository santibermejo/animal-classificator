from keras.utils import Sequence
import numpy as np
import cv2

class AnimalGenerator(Sequence):
  
  def __init__(self, paths, labels, batch_size, shape, shuffle = False, use_cache = False):
    self.paths, self.labels = paths, labels
    self.batch_size = batch_size
    self.shape = shape
    self.shuffle = shuffle
    self.use_cache = use_cache
    if use_cache == True:
        self.cache = np.zeros((paths.shape[0], shape[0], shape[1], shape[2]), dtype=np.float16)
        self.is_cached = np.zeros((paths.shape[0]))
    self.on_epoch_end()
    
  def __len__(self):
    return int(np.ceil(len(self.paths)/float(self.batch_size)))
  
  def __getitem__(self, idx):
    indexes = self.indexes[idx * self.batch_size : (idx+1) * self.batch_size]

    paths = self.paths[indexes]
    X = np.zeros((paths.shape[0], self.shape[0], self.shape[1], self.shape[2]))
    # Generate data
    if self.use_cache == True:
        X = self.cache[indexes]
        for i, path in enumerate(paths[np.where(self.is_cached[indexes] == 0)]):
            image = self.__load_image(path)
            self.is_cached[indexes[i]] = 1
            self.cache[indexes[i]] = image
            X[i] = image
    else:
        for i, path in enumerate(paths):
            X[i] = self.__load_image(path)

    y = self.labels[indexes]

    return X, y
  
  def __load_image(self, path):
    im = cv2.imread(path)
    im = cv2.resize(im, (self.shape[0], self.shape[1]), interpolation=cv2.INTER_CUBIC)
    im = np.divide(im, 255.)
    return im
    
  def on_epoch_end(self):
      # Updates indexes after each epoch
      self.indexes = np.arange(len(self.paths))
      if self.shuffle == True:
          np.random.shuffle(self.indexes)

  def __iter__(self):
      """Create a generator that iterate over the Sequence."""
      for item in (self[i] for i in range(len(self))):
          yield item