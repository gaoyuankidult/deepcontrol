from model_nu_config import *
import einstein as E



if __name__ == "__main__":

    class ModelRecorder(object):
        def __init__(self):
            # all records
            self.records = []
            # current epoch number
            self.t = -1

        def record(self, value):
            assert E.tools.check_list_depth(value) == 2, \
                "The records should be two dimensional array, maybe forget using ModelRecorder.new_epoch() ?"
            self.records[self.t].append(value)

        def record_real_tests(self):
            pass

        def new_epoch(self):
            self.records.append([])
            self.t += 1

        def print_running_avg(self, current_epoch, interval, steps):
            assert isinstance(steps, int)
            assert isinstance(interval, int)
            if current_epoch%interval == 0:
                print "Current epoch %i, Interval %d, steps %d, average reward %f" \
                      % (current_epoch,
                         interval,
                         steps,
                         E.tools.mean(self.records[-steps::1]))





    actor_model = ModelNuActor(setting=ans)
    critic_model = ModelNuCritic(setting=cns)
    real_balance_task = E.tasks.BalanceTask(env=E.environments.RealCartPoleEnvironment(), maxsteps=200)
    sim_balance_task = E.tasks.BalanceTask(env=E.environments.SimCartPoleEnvironment(model=actor_model),
                                           maxsteps=200)
    real_experiment = E.actions.RealExperiment(task=real_balance_task,
                                               actor_model=actor_model)
    thought_experiment = E.actions.ThoughtExperiment(task=sim_balance_task,
                                                     actor_model=actor_model,
                                                     critic_model=critic_model)

    model_recorder = ModelRecorder()
    sys_sampler = E.samplers.SysSampler()

    model_nu = ModelNu(actor_model=actor_model, critic_model=critic_model)

    for n in xrange(2):
        model_recorder.new_epoch()
        for i in xrange(1):
            model_nu.current_epoch = i
            model_nu.current_sample_variances = sys_sampler.sample(model_variances=model_nu.model_variances)
            reward1 = real_experiment.one_epicode(model_nu.current_weights +
                                                  model_nu.current_sample_variances)

            model_nu.store_past_experiences(real_experiment.get_training_data(unfolding=200))
            model_nu.train_critic_network(training_method="direct training")


            model_nu.check_best_reward(reward1)
            reward2 = real_experiment.one_epicode(model_nu.current_weights -
                                                  model_nu.current_sample_variances)

            model_nu.store_past_experiences(real_experiment.get_training_data(unfolding=200))
            model_nu.train_critic_network(training_method="direct training")

            model_nu.check_best_reward(reward2)
            model_nu.update_mean_reward(reward1, reward2)
            model_nu.update_baseline()
            model_nu.update_weights()
            model_nu.update_variances()
            model_recorder.record(model_nu.mean_reward)
            model_recorder.print_running_avg(current_epoch=i, interval=50, steps=50)












