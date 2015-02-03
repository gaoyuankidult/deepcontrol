from lasagne import layers

class ConnectLayer(Layer, list_of_layers, list_of_shapes):
    def __init__(self):
        super(ConnectLayer, self).__init__()
        