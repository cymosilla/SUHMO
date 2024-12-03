import numpy as np
import h5py

bisiclefile = "plot000200.2d.hdf5"

#class box:
#    def __init__(self,
#                 lo_x : int,
#                 lo_y : int,
#                 hi_x : int,
#                 hi_y : int) -> None:
#        self.lo_x = lo_x
#        self.lo_y = lo_y
#        self.hi_x = hi_x
#        self.hi_y = hi_y
#
#    def size(self) -> int :
#        return (hi_x - lo_x) * (hi_y - lo_y)


with h5py.File(bisiclefile, "r") as f:
    varlist = []
    # Datasets names are stored in base group attributes
    for k in f.attrs.keys():
        varlist.append(str(f.attrs[k]))
    print(varlist)

    # Read the Chombo data
    ChomboGroup = f["Chombo_global"]
    #print(ChomboGroup.attrs.keys())

    # Read level data
    levelGroup = f["level_0"]
    datasets = list(levelGroup)

    # Read boxes as numpy arrays
    boxes_h5 = levelGroup["boxes"][()]

    # Read up the whole dataset
    dataoffsets = levelGroup["data:offsets=0"][()]
    datafull = levelGroup["data:datatype=0"][()]

    # Data per box
    #sliced_nparrays = []
    #for i in range(len(boxes_h5)):
    #    sliced_nparrays.append(datafull[dataoffsets[i]:dataoffsets[i+1]-1])
    #print(sliced_nparrays)

    # size for each variable
    i_min = 999999
    i_max = 0
    j_min = 999999
    j_max = 0
    for b in (boxes_h5):
        i_min = np.amin([i_min,b[0]])
        j_min = np.amin([j_min,b[1]])
        i_max = np.amax([i_max,b[2]])
        j_max = np.amax([j_max,b[3]])
    #print(i_min,j_min,i_max,j_max)

    # Variables per box
    n_GC = 1
    var_sliced_nparrays = np.zeros(shape=(j_max+1,i_max+1))
    for i in range(len(boxes_h5)):    
        box = boxes_h5[i]
        offset_b = dataoffsets[i] 
        l_x = box[2]-box[0]+1 + 2*n_GC 
        l_y = box[3]-box[1]+1 + 2*n_GC 
        print(offset_b)
        print(l_x,l_y)
        print(box)
        for m in range(box[1],box[3]+1):
            for n in range(box[0],box[2]+1):
                n_loc = n-box[0]
                m_loc = m-box[1]
                var_sliced_nparrays[m,n] = datafull[offset_b + (m_loc+1)*l_x + n_loc + 1]
                #var_sliced_nparrays[m,n] = datafull[offset_b + (m)*l_x + n]
                #print(var_sliced_nparrays[0,n,m])

    # plot
    import matplotlib.pyplot as plt
    x = np.linspace(0, 319, 320)
    y = np.linspace(0, 63, 64)
    X, Y = np.meshgrid(x, y)
    plt.contourf(X, Y, var_sliced_nparrays)
    plt.colorbar()
    plt.show()
