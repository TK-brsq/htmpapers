import random
from tqdm import tqdm
import csv
import numpy as np
from dataloader import DataLoader, MMDataLoader, PseudoDataLoader, PseudoDataLoader0, PseudoDataLoader1
#from GcCnn import ControlGCN
from B_GcCnn import ControlGCN
import matplotlib.pyplot as plt
from datetime import datetime
import pytz
from collections import namedtuple, defaultdict


class Experiment:
    def __init__(self, dataloader=DataLoader, use_train_in_testing=False, n_modals=1, n_modules=40, n_classes=10, dpc=1, T=25, recommendation=False, num_trainmovement=1, **kwargs):
        self.model = ControlGCN
        self.loader = dataloader
        self.train = use_train_in_testing
        self.n_modals = n_modals
        self.dpc = dpc
        self.T = T
        self.recommendation = recommendation
        self.num_trainmovement = num_trainmovement
        self.datasize = dpc * n_classes
        self.config_model = {
            'n_modals': n_modals,
            'n_modules': n_modules,
            'n_classes': n_classes,
            'T': T,
            'recommendation': recommendation,
        }
        self.config_loader = {
            'n_modals': n_modals,
            'data_per_class': dpc,
            'n_classes': n_classes,
        }


    def run(self, seed_=0):
        random.seed(seed_)
        np.random.seed(seed_)
        model = self.model(**self.config_model)
        train_loader = self.loader(train=True, **self.config_loader)
        test_loader = self.loader(train=self.train, **self.config_loader)
        storage = defaultdict(list)

        # --- train ---
        for iter, (vecs, target) in enumerate(tqdm(train_loader, desc='Train')):
            model.learn(vecs, target)
            #break
        
        # --- train movement ---
        if self.recommendation:
            map = {1: 0, 5: 1, 10: 2, 20: 3}
            for i in range(self.num_trainmovement):
                random.seed(seed_ + i)
                np.random.seed(seed_ + i)
                for iter, (vecs, target) in enumerate(tqdm(train_loader, desc='Train Movement {}'.format(i+1))):
                    results = model.learn_movement(vecs, target, map[self.dpc])
                    for k, v in results.items():
                        storage[k].append(v)
                    #break
        

        # --- eval ---
        random.seed(seed_)
        np.random.seed(seed_)
        pred_hist = np.zeros(self.T, dtype=np.float32)
        for iter, (vecs, target) in enumerate(tqdm(test_loader, desc='Eval')):
            results = model.infer(vecs, target, data_idx=iter)
            pred_t = results['pred_step_Eval']
            pred_hist[pred_t] += 1 if pred_t is not None else 0
            for k, v in results.items():
                storage[k].append(v)

        print('brief : {}'.format(np.mean(storage['brief_Eval'], axis=0)))
        print('mi_rate : {}'.format(np.mean(storage['mi_T_Eval'])))
        #print('accuracy {}'.format(np.cumsum(pred_hist) / float(len(test_loader))))

        return storage


    def run_seeds(self, n_seeds=1,):
        results = defaultdict(list)

        for i in range(n_seeds):
            storage = self.run(i)
            for k, v in storage.items():
                results[k].append(v)

        results = {k: np.array(v) for k, v in results.items()}
        return results


def plot_brief(brief, name=''):
    avg = np.mean(brief, axis=1)
    med = np.median(avg, axis=0)
    plt.plot(range(1,25+1), med, marker='.', label='label')
    plt.savefig('results2/week0/brief_avg' + name + '.png')
    plt.clf()


def save_result(results={}, config={}):
    japan_time = datetime.now(pytz.timezone("Asia/Tokyo"))
    now_ = japan_time.strftime("%m-%d-%H-%M")
    
    config['dataloader'] = config['dataloader'].__name__
    results.update(config)
    np.save("results/week0/" + now_ + ".npy", results)


if __name__ == '__main__':
    # setting
    config = {
        'dataloader': PseudoDataLoader0,
        'use_train_in_testing': True,  
        'n_modals': 2,
        'n_modules': 10,
        'n_classes': 10,
        'dpc': 1,
        'T': 25,
        'recommendation': False,
        'num_trainmovement': 1,
        'save': True,
        'n_seeds': 1,
    }
    with open("log_learn.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.write('learn')
    with open("log_infer.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.write('learn')

    # run
    experiment = Experiment(**config)
    results = experiment.run_seeds(n_seeds=config['n_seeds'])

    # save
    if config['save']:
        save_result(results, config)
    
    # ================================================

    # ================================================
    
