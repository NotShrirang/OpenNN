def Positive_Negative_data(points):
    X = np.zeros((points, 2))
    y = np.zeros(points, dtype='uint8')
    for ix in range(points):
        n1 = random.randint(-10, 10)
        n2 = random.randint(-10, 10)
        if n1*n2 < 0:
            X[ix] = np.c_[n1, n2]
            y[ix] = 0
        else:
            X[ix] = np.c_[n1, n2]
            y[ix] = 1
    return X, y
        

def spiral_data(points, classes):
    X = np.zeros((points*classes, 2))
    y = np.zeros(points*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0, 1, points)  # radius
        t = np.linspace(class_number*4, (class_number+1)*4, points) + np.random.randn(points)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    return X, y

def vertical_data(samples, classes):
    X = np.zeros((samples*classes, 2))
    y = np.zeros(samples*classes, dtype='uint8')
    data = []
    for class_number in range(classes):
        ix = range(samples*class_number, samples*(class_number+1))
        X[ix] = np.c_[np.random.randn(samples)*.1 + (class_number)/3, np.random.randn(samples)*.1 + 0.5]
        y[ix] = class_number
    return X, y

def linear_data(hm, variance, step=2, correlation=True):
    val = 1
    ys = []
    for _ in range(hm):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation or correlation == 'pos':
            val += step
        elif correlation or correlation == 'neg':
            val -= step
    xs = [i for i in range(len(ys))]

    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)
