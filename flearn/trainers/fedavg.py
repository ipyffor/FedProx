import numpy as np
from tqdm import trange, tqdm
import tensorflow as tf
import heapq
from .fedbase import BaseFedarated
import random

from flearn.utils.tf_utils import process_grad


class Server(BaseFedarated):
    def __init__(self, params, learner, dataset):
        print('Using Federated avg to Train')
        self.inner_opt = tf.train.GradientDescentOptimizer(params['learning_rate'])
        super(Server, self).__init__(params, learner, dataset)

    def train(self):
        '''Train using Federated Proximal'''
        print('Training with {} workers ---'.format(self.clients_per_round))
        choices = ['loss', 'grad', 'grad_random']
        choice = 'select_based_' + choices[2]

        for i in range(self.num_rounds):
            # test model
            if i % self.eval_every == 0:
                stats = self.test()  # have set the latest model for all clients
                stats_train = self.train_error_and_loss()
                self.metrics.accuracies.append(stats)
                self.metrics.train_accuracies.append(stats_train)
                tqdm.write('At round {} accuracy: {}'.format(i, np.sum(stats[3]) * 1.0 / np.sum(stats[2])))  # testing accuracy
                tqdm.write('At round {} training accuracy: {}'.format(i, np.sum(stats_train[3]) * 1.0 / np.sum(stats_train[2])))
                tqdm.write('At round {} training loss: {}'.format(i, np.dot(stats_train[4], stats_train[2]) * 1.0 / np.sum(stats_train[2])))
                self.metrics.write(choice)

            # indices, selected_clients = self.select_clients(i, num_clients=self.clients_per_round)  # uniform sampling
            # np.random.seed(i)
            # active_clients = np.random.choice(selected_clients, round(self.clients_per_round * (1-self.drop_percent)), replace=False)
            clients = self.get_clients()
            csolns = []  # buffer for receiving client solutions
            all_delta_grads = []
            all_stats = []
            clients_list = clients.tolist()
            loss_all = []

            for idx, c in enumerate(clients_list):  # simply drop the slow devices
                # communicate the latest model
                c.set_params(self.latest_model)

                # solve minimization locally
                soln, stats, delta_grads, loss = c.solve_inner(num_epochs=self.num_epochs, batch_size=self.batch_size)
                loss_all.append(loss)
                # gather solutions from client
                csolns.append(soln)
                all_delta_grads.append(delta_grads)
                all_stats.append(stats)

                # track communication cost


            #
            if choice == 'select_based_loss':
                index = map(loss_all.index, heapq.nsmallest(self.clients_per_round, loss_all))
            elif choice == 'delect_based_grad':
                index = map(all_delta_grads.index, heapq.nlargest(self.clients_per_round, all_delta_grads))
            elif choice == 'select_based_grad_random':
                vis = [0]*len(all_delta_grads)
                sum_prob = sum(all_delta_grads)
                index = []
                for _ in range(self.clients_per_round):

                    x = random.uniform(0, sum_prob)
                    cumu_prob = 0.
                    for i, (item) in enumerate(all_delta_grads):
                        if vis[i] == 0:
                            cumu_prob += item
                            if x < cumu_prob:
                                index.append(i)
                                sum_prob -= all_delta_grads[i]
                                vis[i] = 1
                                break
                pass
            # update models
            if not isinstance(index, list):
                index = list(index)
            for j in index:
                self.metrics.update(rnd=i, cid=clients_list[j].id, stats=all_stats[j])
            csolns = [csolns[i] for i in index]
            self.latest_model = self.aggregate(csolns)

        # final test model
        stats = self.test()
        stats_train = self.train_error_and_loss()
        self.metrics.accuracies.append(stats)
        self.metrics.train_accuracies.append(stats_train)
        tqdm.write('At round {} accuracy: {}'.format(self.num_rounds, np.sum(stats[3]) * 1.0 / np.sum(stats[2])))
        tqdm.write('At round {} training accuracy: {}'.format(self.num_rounds, np.sum(stats_train[3]) * 1.0 / np.sum(stats_train[2])))
        self.metrics.write(choice)
