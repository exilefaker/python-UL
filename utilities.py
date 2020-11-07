import numpy as np
import array

def shuffle_arrays(a,b):
	r = np.random.RandomState()
	state = r.get_state()
	r.set_state(state)
	r.shuffle(a)
	r.set_state(state)
	r.shuffle(b)
	return a,b