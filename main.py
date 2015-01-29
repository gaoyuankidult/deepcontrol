import lasagne as l
import einstein as E


setting = E.model.Setting()
setting.n_batches = 1
setting.learning_rate = 1e-4
setting.n_time_steps = 5
setting.n_input_features = 4
setting.n_output_features = 1

input_layer_setting = E.model.InputLayerSetting()
input_layer_setting.n_input_features = setting.n_input_features
setting.append_layer(layer=l.layers.InputLayer, layer_setting=input_layer_setting)

reshape_layer_setting = E.model.ReshapeLayerSetting()
reshape_layer_setting.reshape_shape = (setting.n_batches,setting.n_time_steps,setting.n_input_features)
setting.append_layer(l.layers.ReshapeLayer, reshape_layer_setting)

lstm_layer_setting = E.model.LSTMLayerSetting()
lstm_layer_setting.n_lstm_hidden_units = 20
setting.append_layer(l.layers.LSTMLayer, lstm_layer_setting)

reshape_layer_setting = E.model.ReshapeLayerSetting()
reshape_layer_setting.reshape_shape= (setting.n_batches * setting.n_time_steps, lstm_layer_setting.n_lstm_hidden_units)
setting.append_layer(l.layers.ReshapeLayer,reshape_layer_setting)

dense_layer_setting = E.model.DenseLayerSetting()
dense_layer_setting.dense_n_hidden_units = 20
setting.append_layer(l.layers.DenseLayer, dense_layer_setting)

reshape_layer_setting = E.model.ReshapeLayerSetting()
reshape_layer_setting.reshape_shape = (setting.n_batches, setting.n_time_steps, setting.n_output_features)
setting.append_layer(l.layers.ReshapeLayer, reshape_layer_setting)


model = E.model.Model(setting)
serial = E.serial.socket.SocketServer()

# first send n_time_steps information to the client
serial.send_int(setting.n_time_steps)


model.train()