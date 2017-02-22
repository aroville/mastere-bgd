
import numpy as np

# import pdb; pdb.set_trace()
# from IPython.core.debugger import Pdb; ipdb = Pdb(); ipdb.set_trace()


class DataSet(object):
    def __init__(self, filename_data, filename_gender, nbdata, l2_normalize=False, batch_size=128):
        self.nbdata = nbdata
        # taille des images 48*48 pixels en niveau de gris
        self.dim = 2304
        self.data = None
        self.label = None
        self.batchSize = batch_size
        self.curPos = 0    
                
        f = open(filename_data, 'rb')
        self.data = np.empty([nbdata, self.dim], dtype=np.float32)
        for i in xrange(nbdata):
            self.data[i, :] = np.fromfile(f, dtype=np.uint8, count=self.dim)
        f.close()

        f = open(filename_gender, 'rb')
        self.label = np.empty([nbdata, 2], dtype=np.float32)
        for i in xrange(nbdata):
            self.label[i, :] = np.fromfile(f, dtype=np.float32, count=2)
        f.close()
        
        print 'nb data : ', self.nbdata
        
        tmpdata = np.empty([1, self.dim], dtype=np.float32)
        tmplabel = np.empty([1, 2], dtype=np.float32)
        arr = np.arange(nbdata)
        np.random.shuffle(arr)
        tmpdata = self.data[arr[0], :]
        tmplabel = self.label[arr[0], :]
        for i in xrange(nbdata-1):
            self.data[arr[i], :] = self.data[arr[i+1], :] 
            self.label[arr[i], :] = self.label[arr[i+1], :] 
        self.data[arr[nbdata-1], :] = tmpdata
        self.label[arr[nbdata-1], :] = tmplabel 
        
        if l2_normalize:
            self.data /= np.sqrt(np.expand_dims(np.square(self.data).sum(axis=1), 1))

    def next_training_batch(self):
        if self.curPos + self.batchSize > self.nbdata:
            self.curPos = 0
        xs = self.data[self.curPos:self.curPos+self.batchSize, :]
        ys = self.label[self.curPos:self.curPos+self.batchSize, :]
        self.curPos += self.batchSize
        return xs,ys
    
    def mean_accuracy(self, tf_session, loc_acc, loc_x, loc_y):
        acc = 0
        for i in xrange(0, self.nbdata, self.batchSize):
            cur_batch_size = min(self.batchSize, self.nbdata - i)
            dict = {
                loc_x: self.data[i:i+cur_batch_size, :],
                loc_y: self.label[i:i+cur_batch_size, :]}
            
            acc += tf_session.run(loc_acc, dict) * cur_batch_size
        acc /= self.nbdata
        return acc
