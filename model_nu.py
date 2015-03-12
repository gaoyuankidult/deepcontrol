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
            self.records[self.t].append(value)
            assert E.tools.check_list_depth(self.records) == 2, \
                "The records should be two dimensional array, maybe forget using ModelRecorder.new_epoch() ?"

        def record_real_tests(self):
            pass

        def new_epoch(self):
            self.records.append([])
            self.t += 1

        def print_running_avg(self, current_epoch, n_real_example, interval, steps):
            assert isinstance(steps, int)
            assert isinstance(interval, int)
            if current_epoch%interval == 0:
                print "Current epoch %i, real example %d, Interval %d, steps %d, average reward %f" \
                      % (current_epoch,
                         n_real_example,
                         interval,
                         steps,
                         E.tools.mean(self.records[-steps::1]))
        def print_real_and_sim_diff(self, real_experiment, sim_experiment, params):

            real_reward = real_experiment.one_epicode(params)

            sim_reward = sim_experiment.one_epicode(params)

            print "For the same parameters, the real reward of the system is %f, the sim reward of system is %f" % \
                  (real_reward, sim_reward)




    actor_model = ModelNuActor(setting=ans, name="Actor Model", default="default")
    critic_model = ModelNuCritic(setting=cns, name="Critic Model", default="default")
    real_balance_task = E.tasks.BalanceTask(env=E.environments.RealCartPoleEnvironment(), maxsteps=200)
    real_balance_task.randomInitialization = False
    real_balance_task.reset()
    sim_balance_task = E.tasks.BalanceTask(env=E.environments.SimCartPoleEnvironment(critic_model=critic_model),
                                           maxsteps=200)
    sim_balance_task.randomInitialization = False
    sim_balance_task.reset()
    real_experiment = E.actions.RealExperiment(task=real_balance_task,
                                               actor_model=actor_model)
    thought_experiment = E.actions.ThoughtExperiment(task=sim_balance_task,
                                                     actor_model=actor_model,
                                                     critic_model=critic_model)

    model_recorder = ModelRecorder()
    sys_sampler = E.samplers.SysSampler()

    model_nu = ModelNu(actor_model=actor_model, critic_model=critic_model)


    for n in xrange(4):
        model_recorder.new_epoch()
        cost1 = 1000
        cost2 = 1000
        for i in xrange(10000000):
            model_nu.current_epoch = i
            model_nu.current_sample_variances = sys_sampler.sample(model_variances=model_nu.model_variances)

            #if cost1 + cost2 < 0.003 * 20 and i >= 40:
            #    reward1 = thought_experiment.one_epicode(model_nu.current_weights +
            #                                          model_nu.current_sample_variances)
            #    model_nu.check_best_reward(reward1)
            #    reward2 = thought_experiment.one_epicode(model_nu.current_weights -
            #                                          model_nu.current_sample_variances)
            #    model_nu.check_best_reward(reward2)

            #    cost1 = 1000
            #    cost2 = 1000

            #else:
            reward1 = real_experiment.one_epicode(model_nu.current_weights +
                                                  model_nu.current_sample_variances)

            model_nu.store_past_experiences(*real_experiment.get_training_data(unfolding=cns.n_time_steps))
            cost1 = sum(model_nu.train_critic_network(training_method="stochastic training", training_step=400))


            model_nu.check_best_reward(reward1)
            reward2 = real_experiment.one_epicode(model_nu.current_weights -
                                                  model_nu.current_sample_variances)

            model_nu.store_past_experiences(*real_experiment.get_training_data(unfolding=cns.n_time_steps))
            cost2 = sum(model_nu.train_critic_network(training_method="stochastic training", training_percent=0.3))
            model_nu.check_best_reward(reward2)


            model_recorder.print_real_and_sim_diff(real_experiment, thought_experiment,
                                                   model_nu.current_weights + model_nu.current_sample_variances)
            print "cost", cost1+cost2
            model_nu.update_mean_reward(reward1, reward2)
            #if cost1 + cost2 < 0.003 * 200 and i >= 50:
            #    pass
            #else:
            model_nu.update_baseline()
            model_nu.update_weights()
            model_nu.update_variances()
            model_recorder.record(model_nu.mean_reward)
            model_recorder.print_running_avg(current_epoch=i, n_real_example=model_nu.n_real_examples,
                                             interval=50, steps=50)












