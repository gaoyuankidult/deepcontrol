import einstein as E
import theano.tensor as TT




# Network Properties
ans = E.model.Setting(n_batches = 1,
                      learning_rate=2e-1,
                      n_time_steps=1,
                      n_input_features=4,
                      n_output_features=1,
                      n_iterations=100000) # Actor Network Setting

# First Layer is Input Layer
ans.append_layer(layer=E.layers.InputLayer,
                 layer_setting=E.model.InputLayerSetting(shape=(ans.n_batches,
                                                                ans.n_time_steps,
                                                                ans.n_input_features)))
ans.append_layer(layer=E.layers.DenseLayer,
                 layer_setting=E.model.DenseLayerSetting(dense_n_hidden_units=1,
                                                         nonlineariry=E.layers.nonlinearities.identity))

ans.append_layer(layer=E.layers.ReshapeLayer,
                 layer_setting=E.model.ReshapeLayerSetting(reshape_to=(ans.n_batches,
                                                                       ans.n_time_steps,
                                                                       1)))


class ModelNuActor(E.model.Model):
    def __init__(self, setting):
        super(ModelNuActor, self).__init__(setting)
        self.build_functions()

    def build_functions(self):
        super(ModelNuActor, self).build_functions()




cns = E.model.Setting(n_batches=1,
                      learning_rate=2e-4,
                      n_time_steps=200,  # Unfolding it for whole sequence
                      n_input_features=5,  # Action and current states
                      n_output_features=5,  # Reward and predicted states
                      n_iterations=100000)

# First Layer is Input Layer
cns.append_layer(layer=E.layers.InputLayer,
                 layer_setting=E.model.InputLayerSetting(shape=(cns.n_batches,
                                                                cns.n_time_steps,
                                                                cns.n_input_features)))
cns.append_layer(layer=E.layers.LSTMLayer,
                 layer_setting=E.model.LSTMLayerSetting(n_lstm_hidden_units=10))

cns.append_layer(layer=E.layers.ReshapeLayer,
                 layer_setting=E.model.ReshapeLayerSetting(reshape_to=(cns.n_batches,
                                                                       cns.n_time_steps,
                                                                       1)))
cns.append_layer(layer=E.layers.DenseLayer,
                 layer_setting=E.model.DenseLayerSetting(dense_n_hidden_units=cns._n_output_features,
                                                         nonlineariry=E.layers.nonlinearities.identity))

cns.append_layer(layer=E.layers.ReshapeLayer,
                 layer_setting=E.model.ReshapeLayerSetting(reshape_to=(cns.n_batches,
                                                                       cns.n_time_steps,
                                                                       cns._n_output_features)))


class ModelNuCritic(E.model.Model):
    def __init__(self, setting, mask):
        super(ModelNuCritic, self).__init__(setting, mask=mask)

    def build_functions(self):
        super(ModelNuCritic, self).build_functions()

    def get_input_shape(self):
        return self.layers[0].shape

    def get_output_shape(self):
        return self.layers[-1].shape


class ModelNu(object):
    def __init__(self, actor_model, critic_model):
        self.cost_confidence = 1
        self.baseline = None
        self.num_parameters = 4  # five parameters
        self.init_sigma = 3 # initial number sigma
        self.model_variances = E.tools.ones(self.num_parameters) * self.init_sigma
        self.best_reward = -1000
        self.actor_model = actor_model
        self.critic_model = critic_model

        self.current_weights = self.actor_model.get_all_params()[0].get_value()[:, 0]
        self.mean_reward = None
        self.current_sample_variances = None
        self.reward1 = None
        self.reward2 = None
        self.current_epoch = 0

        self.input_experiences = []
        self.output_experiences = []
        self.masks = []

    def check_best_reward(self, proposal):
        if proposal > self.best_reward:
            self.best_reward = proposal

    def update_mean_reward(self, reward1, reward2):
        self.reward1 = reward1
        self.reward2 = reward2
        self.mean_reward = (reward1 + reward2)/2.

    def update_baseline(self):
        if self.mean_reward == None:
            raise ValueError
        if self.baseline is None:
            # first learning step
            self.baseline = self.mean_reward
        self.baseline = 0.99 * (0.9 * self.baseline + 0.1 * self.mean_reward)

    def update_weights(self):
        #calc the gradients
        if self.reward1 != self.reward2:
            #gradient estimate alla SPSA but with likelihood gradient and normalization
            fakt = (self.reward1 - self.reward2) / (2. * self.best_reward - self.reward1 - self.reward2)
        else:
            fakt=0.
        self.current_weights += self.actor_model.setting.learning_rate * fakt * self.current_sample_variances

    def update_variances(self):
        norm = (self.best_reward - self.baseline)
        if norm != 0.0:
            fakt2=(self.mean_reward-self.baseline)/(self.best_reward-self.baseline)
        else:
            fakt2 = 0.0

        if fakt2 > 0.: #for sigma adaption alg. follows only positive gradients
            self.model_variances += \
                self.actor_model.setting.learning_rate * fakt2 * (self.current_sample_variances**2 -
                                              self.model_variances ** 2) / self.model_variances

    def train_critic_network(self, training_method=None, traing_step=None, mask=None):
        costs = []
        all_data = zip(self.input_experiences, self.output_experiences)
        if training_method == "direct training":
            for input, output, mask in all_data:
                critic_train_input = E.tools.theano_form(input, shape=self.critic_model.get_input_shape())
                critic_train_output = E.tools.theano_form(output, shape=self.critic_model.get_output_shape())
                costs.append(self.critic_model.train(critic_train_input, critic_train_output, mask=mask))
        elif training_method == "stocastic training":
            for _ in xrange(traing_step):
                input, output = E.tools.random.choose(all_data)
                self.critic_model.train(input, output, mask)

    def store_past_experiences(self, input, output, mask):
        self.input_experiences.extend(input)
        self.output_experiences.extend(output)
        self.masks.extend(mask)






