#  Numenta Platform for Intelligent Computing (NuPIC)
#  Copyright (C) 2020, Numenta, Inc.  Unless you have an agreement
#  with Numenta, Inc., for a separate license for this software code, the
#  following terms and conditions apply:
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero Public License version 3 as
#  published by the Free Software Foundation.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#  See the GNU Affero Public License for more details.
#
#  You should have received a copy of the GNU Affero Public License
#  along with this program.  If not, see http://www.gnu.org/licenses.
#
#  http://numenta.org/licenses/
#

import math
import numpy as np
import random
import copy
import csv
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import entropy
from htmresearch.support import numpy_helpers as np2
from nupic.bindings.math import Random, SparseMatrixConnections
from A_ApicalTemporalMemory import ApicalTiebreakPairMemory
from A_LocationModule import Superficial2DLocationModule
CELLWIDTH = 256


def myent(l):
    if max(l) == 0:
        return entropy([1./len(l)] * len(l))
    else:
        return entropy(l)


class WeightMatrix(object):
    def __init__(self, n_cells=40*CELLWIDTH**2, activate_thresh=40, potential_thresh=0, weight_thresh=1./2., delta=0.1, initial_weight=0.4):
        self.weight = SparseMatrixConnections(n_cells, n_cells)
        self.pre_active_cells = []
        self.active_cells = []
        self.active_threshold = activate_thresh
        self.potential_threshold = potential_thresh
        self.weight_threshold = weight_thresh
        self.delta = delta
        self.initial_weight = initial_weight
        self.rng = Random(42)


    def learn(self, pre, post, ss=1.):

        # Step 1 : activate segments and cells
        overlap = self.weight.computeActivity(pre, self.weight_threshold)
        active_segments = np.where(overlap >= self.active_threshold)[0]
        active_cells = self.weight.mapSegmentsToCells(active_segments)
        unpredicted_active_cells = np.setdiff1d(post, active_cells)

        potential_overlap = self.weight.computeActivity(pre)
        potential_segments = np.where(potential_overlap >= self.potential_threshold)[0]
        potential_segments = self.weight.filterSegmentsByCell(potential_segments, unpredicted_active_cells)
        potential_cells = self.weight.mapSegmentsToCells(potential_segments)
        #print("active_segments {}".format(active_segments))
        #print("potential_segments {}".format(potential_segments))


        # Step 2 : adjust wieght (this doesnt create new synapses)
        learnable_active_positive_segments = active_segments[np.isin(active_cells, post)]
        #learnable_active_negative_segments = active_segments[~np.isin(active_cells, post)]
        potential_segments = potential_segments[np.isin(potential_cells, unpredicted_active_cells)]
        best_segment_idx = np2.argmaxMulti(potential_overlap[potential_segments], potential_cells)
        learnable_potential_segments = potential_segments[best_segment_idx]
        #print("learnable_positive_segments {}".format(learnable_active_positive_segments))
        #print("learnable_negative_segments {}".format(learnable_active_negative_segments))
        #print("learnable_potential_segments {}".format(learnable_potential_segments))

        self._adjust_weight(learnable_active_positive_segments, pre, self.initial_weight, self.delta*ss, -self.delta*ss)
        #self._adjust_weight(learnable_active_negative_segments, [], self.initial_weight, self.delta*ss, -self.delta*ss)
        self._adjust_weight(learnable_potential_segments, pre, self.initial_weight, self.delta*ss, -self.delta*ss)

        # Step 3 : create new segments and synapses
        new_segment_cells = np.setdiff1d(unpredicted_active_cells, potential_cells)
        newSegments = self.weight.createSegments(new_segment_cells)
        n_newsynapses = len(pre)
        self.weight.growSynapsesToSample(newSegments, pre, n_newsynapses, self.initial_weight, self.rng)
        self.pre_active_cells = active_cells
        pass


    def learn_negative(self, pre, post, ss=-1.):
        
        # Step 1 : activate segments and cells
        potential_overlap = self.weight.computeActivity(pre)
        potential_segments = np.where(potential_overlap >= self.potential_threshold)[0]
        potential_segments = self.weight.filterSegmentsByCell(potential_segments, post)
        #print("potential_segments {}".format(potential_segments))


        # Step 2 : adjust wieght (this doesnt create new synapses)
        learnable_potential_segments = potential_segments
        #print("learnable_potential_segments {}".format(learnable_potential_segments))

        self._adjust_weight(learnable_potential_segments, pre, self.initial_weight, self.delta*ss, 0.)


    def _adjust_weight(self, learnable_segments, input, initial, inc, dec):
        self.weight.adjustSynapses(learnable_segments, input, inc, dec)
    
    
    def _grow_synapses(self, learnable_segments, input, initial):
        n_new_synapses = len(input)
        self.weight.growSynapsesToSample(learnable_segments, input, n_new_synapses, initial, self.rng)
    
    
    def infer(self, input):
        if len(input) == 0:
            return []
        
        overlap = self.weight.computeActivity(input, self.weight_threshold)
        active_segments = np.where(overlap >= self.active_threshold)[0]
        active_cells = np.unique(self.weight.mapSegmentsToCells(active_segments))

        return active_cells


class GridCellNetwork(object):
    def __init__(self, n_modules=40, n_modals=1,):
        # variables
        self.n_modals = n_modals
        self.L6_cpmodal = n_modules * CELLWIDTH**2
        self.L6_cpmodule = CELLWIDTH**2
        
        # L6
        self.L6 = {mi: [] for mi in range(n_modals)}
        cellCoordinateOffsets = tuple([i * 0.998 + 0.001 for i in range(2)])
        perModRange = 90.0 / float(n_modules)
        for i in range(n_modules):
            orientation = (float(i) * perModRange) + (perModRange / 2.0)
            config = {
                "cellsPerAxis": CELLWIDTH,
                "anchorInputSize": 32*128,
                "scale": float(CELLWIDTH),
                "orientation": np.radians(orientation),
                "activationThreshold": 16,
                "initialPermanence": 1.0,
                "connectedPermanence": 0.5,
                "learningThreshold": 16,
                "sampleSize": -1,
                "permanenceIncrement": 0.1,
                "permanenceDecrement": 0.0,
                "cellCoordinateOffsets": cellCoordinateOffsets,
                "anchoringMethod": "corners"
            }
            for mi in range(n_modals):
                self.L6[mi].append(Superficial2DLocationModule(**config))
        
        # L4
        config = {
            "initialPermanence": 1.0,
            "activationThreshold": int(math.ceil(0.9 * n_modules)),
            "reducedBasalThreshold": int(math.ceil(0.9 * n_modules)),
            "minThreshold": n_modules,
            "sampleSize": n_modules,
            "cellsPerColumn": 32,
            "columnCount": 128,
            "basalInputSize": n_modals*n_modules*CELLWIDTH**2
        }
        self.L4 = ApicalTiebreakPairMemory(**config)
        pass
    
    
    def movementCompute(self, displacement, mi=0):
        location = {
            "displacement": [-displacement["top"],
                             displacement["left"]],
        }
        for module in self.L6[mi]:
            module.movementCompute(**location)
        
        with open('log.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            act_cells = self.l6[mi][0].getActiveCells()
            writer.writerow(['act l6 after movement', act_cells])
        
        return location
    
    
    def sensoryCompute(self, vec=None, learn=True, mi=0):
        # L4
        basalInput = self.getLR(mi=mi) if learn \
            else [x for i in range(self.n_modals) for x in self.getLR(mi=i)]
        basal_GC = self.getLLR(mi=mi) if learn else None
        l4_input = {
            "activeColumns": vec,
            "basalInput": basalInput,
            "basalGrowthCandidates": basal_GC,
            "learn": learn,
        }
        self.L4.compute(**l4_input)
        with open('log.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['l4 active colum', vec])
            writer.writerow(['l4 active cells', self.L4.getActiveCells()])
            writer.writerow(['l4 growth candidates', self.L4.getWinnerCells()])

        # L6
        l6_input = {
            "anchorInput": self.L4.getActiveCells(),
            "anchorGrowthCandidates": self.L4.getWinnerCells(),
            "learn": learn,
        }
        if learn:
            for module in self.L6[mi]:
                module.sensoryCompute(**l6_input)
        else:
            for layer in self.L6.values():
                for module in layer:
                    module.sensoryCompute(**l6_input)
        
        with open('log.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            act_cells = self.L6[mi][0].getActiveCells()
            writer.writerow(['act l6 after sensory', act_cells])
        pass
    
    
    def reset(self):
        self.L4.reset()
        for layer in self.L6.values():
            for module in layer:
                module.reset()
        pass


    def activateRandomLocation(self, mi=0, phases=[]):
        if phases:
            for iter, module in enumerate(self.L6[mi]):
                module.activateFixedLocation(phases[iter])
            return phases
        else:
            active_phases = [module.activateRandomLocation() for module in self.L6[mi]]        
            return active_phases

    
    def getLR(self, mi=0):
        activeCells = np.array([], dtype="uint32")
        for i, module in enumerate(self.L6[mi]):
            activeCells = np.append(activeCells, module.getActiveCells() + self.L6_cpmodule * i + self.L6_cpmodal * mi)

        return activeCells

    
    def getLLR(self, mi=0):
        learnableCells = np.array([], dtype="uint32")

        for i, module in enumerate(self.L6[mi]):
            learnableCells = np.append(learnableCells, module.getLearnableCells() + self.L6_cpmodule * i + self.L6_cpmodal * mi)

        return learnableCells

    
    def getSALR(self, mi=0):
        cells = np.array([], dtype="uint32")
        for i, module in enumerate(self.L6[mi]):
            cells = np.append(cells, module.sensoryAssociatedCells + self.L6_cpmodule * i + self.L6_cpmodal * mi)
        return cells
    

    def getSALR_1module(self, mi=0):
        cells = np.array([], dtype="uint32")
        for i, module in enumerate(self.L6[mi]):
            cells = np.append(cells, module.sensoryAssociatedCells + self.L6_cpmodule * i + self.L6_cpmodal * mi)
            break
        return cells


class ControlGCN(object):
    def __init__(self, n_modals=1, n_modules=40, n_classes=10, T=25, recommendation=False):
        self.network = GridCellNetwork(n_modules=n_modules, n_modals=n_modals)
        self.l6_cells = n_modals * n_modules * CELLWIDTH**2
        self.l6_cpmodal = n_modules * CELLWIDTH**2
        self.network_move = WeightMatrix(n_cells=self.l6_cells, activate_thresh=n_modules)
        self.n_modals = n_modals
        self.n_modules = n_modules
        self.n_classes = n_classes
        self.reccommendation = recommendation
        self.linear = np.zeros((self.l6_cells, n_classes))
        self.current = {mi: None for mi in range(n_modals)}
        self.T = T
        self.ent_mean = [1.05, 0.90, 0.995, 1.26]
        self.ent_var = [0.856, 0.835, 0.763, 0.697]
        pass
    
    
    def learn(self, vecs=None, target=None):
        fixed_phases = []
        scatter = MyScatter()
        for mi in range(self.n_modals):
            self.reset()
            fixed_phases = self.network.activateRandomLocation(mi=mi, phases=fixed_phases)
            assigned_cells = []
            

            for t in range(25):
                with open('log_learn.csv', 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['time', t, 'mi', mi, 'pi', t, 'target', target])
                pi = t
                coordinate = self.pi2coordinate(pi)
                vec = (np.where(vecs[mi][pi] == 1))[0]
                self.move(coordinate, mi=mi)
                self.network.sensoryCompute(vec, learn=True, mi=mi)
                assigned_cell = self.network.getLR(mi=mi)#self.network.getSALR(mi=mi)
                scatter.add_cell(self.network.getLR(mi=mi)[0])

                #if t == 7:
                #    break
                #if t == 7:
                #    scatter.plot(name='L_target={}_step={}_place={}-{}.png'.format(target, t, mi, pi), title='place={}-{}'.format(mi, pi))

                assigned_cells.extend(assigned_cell)
            
            self.linear[list(set(assigned_cells)), target] += 1
        scatter.plot(name='target={}.png'.format(target))
        
        pass
    
    
    def learn_movement(self, vecs=None, target=None, dpc_idx=0):
        
        # Step0-0. Set mi_T & pi_T
        mi_T = [0]*25 + [1]*25
        pi_T = [i for i in range(25)] + [i for i in range(0, 25)]
        pair = list(zip(pi_T, mi_T))
        random.shuffle(pair)
        pi_T, mi_T = zip(*pair)

        # Step0-1. Initialize Variable
        self.reset()
        z_idx_prev = []
        ent_prev = entropy([1./self.n_classes]*self.n_classes)
        ent_mean = self.ent_mean[dpc_idx]
        ent_var = self.ent_var[dpc_idx]
        dist_seq = np.zeros((len(mi_T), self.n_modals, self.n_classes), dtype=np.float32)
        ent_seq = np.zeros((len(mi_T), self.n_modals, self.n_classes), dtype=np.float32)
        ent_diff = np.zeros((len(mi_T)), dtype=np.float32)
        ss_seq = np.zeros((len(mi_T)), dtype=np.float32)
        
        
        # Step1. Start Learning
        for t in range(len(mi_T)): #self.T
            self.share_l6() if self.n_modals > 1 else None
            
            # Update
            pi = pi_T[t]
            mi = mi_T[t]
            coordinate = self.pi2coordinate(pi)
            vec = (np.where(vecs[mi][pi] == 1))[0]
            self.move(coordinate, mi='all')
            self.network.sensoryCompute(vec, learn=False, mi='all')

            # Evaluation
            z_idx = [x for mi in range(self.n_modals) for x in self.network.getSALR(mi=mi)]
            z = np.zeros(self.l6_cells)
            z[z_idx] = 1
            y = np.matmul(z, self.linear)

            dist_list = self.classify()
            dist_seq[t] = dist_list
            ent_list = [myent(dist) for dist in dist_list]
            ent_diff[t] = ent_list[1] - ent_list[0]

            # learn movement
            distribution = y / sum(y) if max(y) != 0 else y
            ent = myent(distribution)
            ent_seq[t] = ent
            ss = (ent_mean - ent) / math.sqrt(ent_var)
            ss_seq[t] = ss
            # new
            if ss > 0:
                self.network_move.learn(z_idx_prev, z_idx, ss=ss)
            elif ss < 0:
                self.network_move.learn_negative(z_idx_prev, z_idx, ss=ss)
            
            # old
            """if ent < ent_prev:
                #tqdm.write('good, len(pre) and len(post) {} {}'.format(len(z_idx_prev), len(z_idx)))
                self.network_move.learn(z_idx_prev, z_idx)"""
            
            z_idx_prev = z_idx

        dict = {
            'dist_LM': dist_seq,
            'ent_LM': ent_seq,
            'ss_LM': ss_seq,
            'entdiff_LM': ent_diff,
            'mi_T_LM': mi_T,
            'pi_T_LM': pi_T,
            'target_LM': target,
        }
        return dict
    
    
    def infer(self, vecs=None, target=None, data_idx=0):
        self.reset()

        pi_T = range(25)
        #mi_T = [0]*20 + [1]*5
        mi_T = [random.randint(0, self.n_modals-1) for _ in range(25)]
        #mi_T = [0]*25
        random.shuffle(pi_T)
        #random.shuffle(mi_T)

        pred_flag = False
        pred_step = None
        dist_seq = np.zeros((self.T, self.n_classes), dtype=np.float32)
        dist2_seq = np.zeros((self.T, self.n_modals, self.n_classes), dtype=np.float32)
        brief_seq = np.zeros(self.T, dtype=np.float32)
        mf_seq = np.zeros((self.T, self.n_modals), dtype=np.float32)
        ent_seq = np.zeros(self.T, dtype=np.float32)
        ent_diff = np.zeros(self.T, dtype=np.float32)

        for t in range(self.T):
            # 0. prepare
            pi = pi_T[t]
            mi = mi_T[t]
            mi_prev = mi_T[t-1] if t > 0 else None
            coordinate = self.pi2coordinate(pi)
            vec = (np.where(vecs[mi][pi] == 1))[0]

            # 1. Share
            #tqdm.write('start share_l6') if self.n_modals > 1 else None
            self.share_l6(mi_prev=mi_prev, mi=mi, data_idx=data_idx) if self.n_modals > 1 else None

            # 2. Update
            #tqdm.write('start move and sensoryCompute')
            self.move(coordinate, mi=mi)
            self.network.sensoryCompute(vec, learn=False, mi=mi)

            # 3. Evaluation
            z_idx = [x for i in range(self.n_modals) for x in self.network.getSALR(mi=i)]
            z_for_scatter = [x for i in range(self.n_modals) for x in self.network.getSALR_1module(mi=i)]
            scatter = MyScatter()
            scatter.add_cells(z_for_scatter)
            scatter.plot(name='{}/step={}_place={}-{}.png'.format(data_idx, t, mi, pi), title='place={}-{}'.format(mi, pi))

            #print(z_idx)
            z = np.zeros(self.l6_cells)
            z[z_idx] = 1
            y = np.matmul(z, self.linear)

            dist_list = self.classify()
            dist2_seq[t] = dist_list
            ent_list = [myent(dist) for dist in dist_list]
            ent_diff[t] = ent_list[1] - ent_list[0]
            
            # 4.1 new classify
            distribution = y / sum(y) if max(y) != 0 else y
            ent = myent(distribution)
            dist_seq[t] = distribution
            ent_seq[t] = ent
            brief_seq[t] = distribution[target]

            # 4.2 old classify
            if not pred_flag and t >= 4 and max(distribution) >= 0.3:
                pred = np.argmax(y)
                pred_flag = True
                if pred == target:
                    pred_step = t
            
            # 5. reccommend
            if self.reccommendation:
                assert self.n_modals <= 2, 'recommendation for more than 2 modals is not implemented'

                #tqdm.write('start network_move.infer')
                z_idx_next = self.network_move.infer(z_idx)
                mf1 = np.sum(z_idx_next < self.l6_cpmodal) if len(z_idx_next) != 0 else 0
                mf2 = np.sum(z_idx_next >= self.l6_cpmodal) if len(z_idx_next) != 0 else 0
                mf = np.array([mf1, mf2])
                if np.all(mf == mf[0]):
                    continue
                else:
                    if t < self.T - 1:
                        mi_T[t+1] = np.argmax(mf)
                mf_seq[t] = mf
            
        # 6. return
        dict = {
            'dist_Eval': dist_seq,
            'dist2_Eval': dist2_seq,
            'brief_Eval': brief_seq,
            'pred_step_Eval': pred_step,
            'mf_Eval': mf_seq,
            'mi_T_Eval': mi_T,
            'pi_T_Eval': pi_T,
            'ent_Eval': ent_seq,
            'entdiff_Eval': ent_diff,
            'target_Eval': target,
            }

        return dict
    
    
    def share_l6(self, mi_prev=0, mi=0, data_idx=0):
        if mi_prev is None or mi_prev == mi:
            return
        
        for module_idx in range(self.n_modules):
            act_phases = self.network.L6[mi_prev][module_idx].getActivePhases()
            self.network.L6[mi][module_idx].activePhases = act_phases.copy()
            
        pass
    
    
    def move(self, coordinate=None, mi=0):
        current = {
            "top": coordinate["top"] + coordinate["height"]/2.,
            "left": coordinate["left"] + coordinate["width"]/2.
        }

        if mi == 'all':
            if self.current[0] is not None:
                for i in range(self.n_modals):
                    displacement = {"top": current['top'] - self.current[i]['top'],
                                    "left": current['left'] - self.current[i]['left'],}
                    self.network.movementCompute(displacement, mi=i)
        elif self.current[mi] is not None:
            displacement = {
                "top": current['top'] - self.current[mi]['top'],
                "left": current['left'] - self.current[mi]['left'],
            }
            self.network.movementCompute(displacement, mi=mi)
        
        if mi == 'all':
            self.current = {i: current for i in range(self.n_modals)}
        else:
            self.current[mi] = current
    
    
    def classify(self,):
        distributions = []
        for mi in range(self.n_modals):
            z = np.zeros(self.l6_cells)
            z_idx = [x for x in self.network.getSALR(mi=mi)]
            z[z_idx] = 1

            y = np.matmul(z, self.linear)
            dist = y / sum(y) if max(y) != 0 else y
            distributions.append(dist)
        
        return distributions


    def pi2coordinate(self, pi=None):
        scale = 20
        coordinate = {}
        coordinate['height'] = scale
        coordinate['width'] = scale
        coordinate['top'] = scale * ( pi // 5 )
        coordinate['left'] = scale * ( pi % 5 )
        return coordinate
    
    
    def reset(self):
        self.network.reset()
        self.current = {mi: None for mi in range(self.n_modals)}
        pass


class MyScatter:
    def __init__(self):
        self.x = []
        self.y = []
        self.x2 = []
        self.y2 = []


    def add_cell(self, c):
        if c <= CELLWIDTH**2:
            self.x.append(c % CELLWIDTH)
            self.y.append(c // CELLWIDTH)
        else:
            c = c % CELLWIDTH**2
            self.x2.append(c % CELLWIDTH)
            self.y2.append(c // CELLWIDTH)
        pass


    def add_cells(self, cells):
        for c in cells:
            self.add_cell(c)
        pass


    def plot(self, name='noname.png', title=''):
        plt.scatter(self.x, self.y, c='blue', alpha=0.5)
        plt.scatter(self.x2, self.y2, c='orange', alpha=0.5)
        plt.title(title)
        plt.xlim((0,255))
        plt.ylim((0,255))
        plt.savefig('picture/' + name)
        plt.clf()

