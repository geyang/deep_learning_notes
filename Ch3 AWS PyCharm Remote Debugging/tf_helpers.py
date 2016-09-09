
def get_shape_array(tensor):
    return to_array(tensor.get_shape())

def to_array(shape):
    return list(map(lambda d: d.value, list(shape)))

