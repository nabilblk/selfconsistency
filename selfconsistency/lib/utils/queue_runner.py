import tensorflow as tf
import numpy as np
import time
import multiprocessing as mp
import threading
import queue
            
class CustomRunner(object):
    """
    This class manages the the background threads needed to fill
        a queue full of data.
        
    # Need to call the following code block after initializing everything
    self.sess.run(tf.global_variables_initializer())

    if self.use_tf_threading:
        self.coord = tf.train.Coordinator()
        self.net.train_runner.start_p_threads(self.sess)
        tf.train.start_queue_runners(sess=self.sess, coord=self.coord)
        
    """
    def __init__(self, arg_less_fn, override_dtypes=None,
                 n_threads=1, n_processes=3, max_size=30,batch_size = 5,patch_size=128):
        # arg_less_fn should be function that returns already ready data
        # in the form of numpy arrays. The shape of the output is
        # used to shape the output tensors. Should be ready to call at init_time
        # override_dtypes is the typing, default to numpy's encoding.
        self.data_fn = arg_less_fn
        self.n_threads = n_threads
        self.n_processes = n_processes
        self.max_size = max_size
        self.use_pool = False
        data = self.data_fn()
        self.nb_elements_per_example = len(data)
        self.inps = []
        shapes, dtypes = [], []
        # d.shape[1:] is the patch size
        # TODO: Useless loop
        for i, d in enumerate(data):
            inp = tf.placeholder(dtype=d.dtype, shape=[None] + list(d.shape[1:]))
            self.inps.append(inp)
            # remove batching index for individual element
            shapes.append(d.shape[1:])
            dtypes.append(d.dtype)
        # The actual queue of data.
        self.tf_queue = tf.FIFOQueue(shapes=shapes,
                                           # override_dtypes or default
                                           dtypes=override_dtypes or dtypes,
                                           capacity=2000)

        # The symbolic operation to add data to the queue
        self.enqueue_op = self.tf_queue.enqueue_many(self.inps)

    def get_inputs(self, batch_size):
        """
        Return's tensors containing a batch of images and labels
        
        if tf_queue has been closed this will raise a QueueBase exception
        killing the main process if a StopIteration is thrown in one of the
        data processes.
        """
        return self.tf_queue.dequeue_up_to(tf.reduce_min([batch_size, self.tf_queue.size()]))

    def thread_main(self, sess, stop_event):
        """
        Function run on alternate thread. Basically, keep adding data to the queue.
        """
        #TODO : RECHECK
        # np.random.seed(1)
        while (not stop_event.isSet()):
            try:
                data = self.data_fn()
            except StopIteration as e:
                stop = True
                break
            fd = {}
            for i, d in enumerate(data):
                fd[self.inps[i]] = d
            sess.run(self.enqueue_op, feed_dict=fd)


        
    def set_data_fn(self, fn):
        self.data_fn = fn
        
    def start_p_threads(self, sess):
        """ Start background threads to feed queue """
        # self.processes = []
        # self.queue = mp.Queue(self.max_size)
        
        # for n in range(self.n_processes):
        #     p = mp.Process(target=self.process_main, args=(self.queue,))
        #     p.daemon = True # thread will close when parent quits
        #     p.start()
        #     self.processes.append(p)
            
        self.threads = []
        self.thread_event_killer = []
        for n in range(self.n_threads):
            kill_thread = threading.Event()
            self.thread_event_killer.append(kill_thread)
            
            t = threading.Thread(target=self.thread_main, args=(sess, kill_thread))
            t.daemon = True # thread will close when parent quits
            t.start()
            self.threads.append(t)
        return self.threads
    
    def kill_programs(self):
        # Release objects here if need to
        # threads should die in at least 5 seconds because
        # nothing blocks for more than 5 seconds
        
        # Sig term, kill first so no more data
        # [p.terminate() for p in self.processes]
        # [p.join() for p in self.processes]
        
        # kill second after purging
        [e.set() for e in self.thread_event_killer]
    