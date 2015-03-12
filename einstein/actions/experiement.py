import einstein as E

class Experiment(object):
    def __init__(self, task, model):
        self.task = task
        self.model = model

    def OneEpicode(self, all_params):
        """
        Give current value of weights, output all rewards
        :return:
        """
        rewards = []
        observations = []
        actions = []
        _all_params = self.model.get_all_params()
        _all_params[0].set_value(E.tools.theano_form(all_params), shape=(4, 1))
        self.task.reset()

        while not self.task.isFinished():
            obs = self.task.getObservation()
            observations.append(obs)
            states = E.tools.theano_form(obs, shape=[self.model.setting.n_batches,
                                                     1,
                                                     self.model.setting.n_input_features - 1]) # this is for each time step
            model_action_result = self.model.predict(states)
            actions.append(model_action_result.reshape(1))
            self.task.performAction(model_action_result)
            rewards.append(self.task.getReward())
        last_obs = self.task.getObservation()
        return rewards, actions, observations, last_obs, sum(rewards)
