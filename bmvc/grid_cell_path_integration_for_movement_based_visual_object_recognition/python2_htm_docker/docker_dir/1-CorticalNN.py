import numpy as np
from typing import List, Set, Tuple, Dict
from collections import defaultdict

class Synapse:
    def __init__(self, presynaptic_cell: int, weight: float):
        self.presynaptic_cell = presynaptic_cell
        self.weight = weight
    
    def is_connected(self, weight_threshold: float) -> bool:
        return self.weight >= weight_threshold

class Segment:
    def __init__(self, cell_idx: int):
        self.cell_idx = cell_idx
        self.synapses: List[Synapse] = []
    
    def add_synapse(self, presynaptic_cell: int, weight=0.21):
        self.synapses.append(Synapse(presynaptic_cell, weight))
    
    def compute_activity(self, active_cells: Set[int], weight_threshold: float=0.5) -> int:
        return sum(1 for syn in self.synapses \
                   if syn.presynaptic_cell in active_cells and syn.is_connected(weight_threshold))

class Cell:
    def __init__(self, column_idx: int, cell_idx_in_column: int):
        self.column_idx = column_idx
        self.cell_idx_in_column = cell_idx_in_column
        self.segments: List[Segment] = []
    
    def add_segment(self):
        global_cell_idx = self.column_idx * CELLS_PER_COLUMN + self.cell_idx_in_column
        segment = Segment(global_cell_idx)
        self.segments.append(segment)
        return segment
    
    def get_max_segment_activity(self, act_cells: Set[int]) -> int:
        return np.max(segment.compute_activity(act_cells) for segment in self.segments)

    def get_max_segment(self, act_cells: Set[int]) -> Segment:
        idx = np.argmax(segment.compute_activity(act_cells) for segment in self.segments)
        return self.segments[idx]

class TemporalMemory:
    def __init__(self,
                 column_count = 128,
                 cells_per_column = 32,
                 activation_threshold = 13,
                 min_threshold = 10,
                 max_synapses = 20,
                 max_segments = 32,
                 max_new_synapse = 10,
                 weight_threshold = 0.5,
                 weight_inc = 0.1,
                 weight_dec = 0.1,):
        
        # architecture parameters
        self.column_count = column_count
        self.cells_per_column = cells_per_column
        self.total_cells = column_count * cells_per_column

        # parameters
        self.activation_threshold = activation_threshold
        self.min_threshold = min_threshold
        self.max_synapses = max_synapses
        self.max_segments = max_segments
        self.max_new_synapses = max_new_synapse
        self.weight_threshold = weight_threshold
        self.weight_inc = weight_inc
        self.weight_dec = weight_dec

        # initialize cells in coordinates style
        self.cells = []
        for col_idx in range(column_count):
            column_cells = []
            for cell_idx in range(cells_per_column):
                column_cells.append(Cell(col_idx, cell_idx))
            self.cells.append(column_cells)
        
        # state variable
        self.active_cells: Set[int] = set()
        self.winner_cells: Set[int] = set()
        # winner_cells can add new synapses in a best matching_segment
        self.predictive_cells: Set[int] = set()
        self.active_segments: Set[Segment] = set()
        self.matching_segments: Set[Segment] = set()
        # matching_segments can be a candidate for new synapses
        self.prev_active_cells: Set[int] = set()
        self.prev_winner_cells: Set[int] = set()
    
    def reset(self):
        self.active_cells.clear()
        self.winner_cells.clear()
        self.predictive_cells.clear()
        self.active_segments.clear()
        self.matching_segments.clear()
        self.prev_active_cells.clear()
        self.prev_winner_cells.clear()

    def compute(self, active_columns: np.ndarray, learn: bool = True):
        ''' Main process in one timestep'''

        self._compute_segment_activity()

        self._compute_cell_activity(active_columns)

        self._learn() if learn else None

        self._compute_predictions()

        self.prev_active_cells = self.active_cells.copy()
        self.prev_winner_cells = self.winner_cells.copy()

        result = {
            'active_cells': self.active_cells.copy(),
            'winner_cells': self.winner_cells.copy(),
            'predictive_cells': self.predictive_cells.copy(),
            'active_columns': set(active_columns)
        }

        return result
    
    def _compute_segment_activity(self):
        ''' update act_seg & match_seg'''

        self.active_segments.clear()
        self.matching_segments.clear()

        for column in self.cells:
            for cell in column:
                for segment in cell.segments:
                    activity = segment.compute_activity(
                        self.prev_active_cells, self.weight_threshold)
                    
                    if activity >= self.activation_threshold:
                        self.active_segments.add(segment)
                    elif activity >= self.min_threshold:
                        self.matching_segments.add(segment)
    
    def _compute_cell_activity(self, active_columns: np.ndarray):
        ''' update act_cells & win_cells '''
        self.active_cells.clear()
        self.winner_cells.clear()

        for col_idx in active_columns:
            # predictive_cells in t-1 ⋂ active_cells in t
            predicted_cells = []
            for cell_idx in range(self.cells_per_column):
                global_cell_idx = col_idx + self.cells_per_column * cell_idx
                if global_cell_idx in self.predictive_cells:
                    predicted_cells.append(global_cell_idx)
            # if some cells could predict, activate only predictive_cells
            if predicted_cells:
                for cell_idx in predicted_cells:
                    self.active_cells.add(cell_idx)
                    self.winner_cells.add(cell_idx)
            # if no cells could predict, activate all cells in the column
            else:
                for cell_idx in range(self.cells_per_column):
                    global_cell_idx = col_idx + self.cells_per_column * cell_idx
                    self.active_cells.add(global_cell_idx)

                    winner_cell = self._choose_winner_cell(col_idx)
                    self.winner_cells.add(winner_cell)
    
    def _choose_winner_cell(self, col_idx: int) -> int:
        ''' when burst occurs, best cell in the column must be chosen '''
        winner_cell = 0
        max_activity = -1
        for cell_idx in range(self.cells_per_column):
            cell = self.cells[col_idx][cell_idx]

            for segment in cell.segments:
                if segment in self.matching_segments:
                    activity = segment.compute_activity(self.prev_active_cells)
                    if activity > max_activity:
                        max_activity = activity
                        winner_cell = cell_idx

        return col_idx * self.cells_per_column + winner_cell

    def _learn(self):
        ''' adapt weights of act_seg & connect new synapses to best seg in winner cells'''
        for segment in self.active_segments:
            # adapt weights
            for synapse in segment.synapses:
                if synapse.presynaptic_cell in self.prev_active_cells:
                    synapse.weight = min(1.0, synapse.weight + self.weight_inc)
                else:
                    synapse.weight = max(0.0, synapse.weight - self.weight_dec)

        # connect new synapse to the best segment in winner_cells
        for winner_cell in self.winner_cells:
            self._learn_on_cell(winner_cell)
    
    def _learn_on_cell(self, cell_idx: int):
        ''' enforce weights & connect new synapses '''

        # access cell instance
        col_idx = cell_idx // self.cells_per_column
        cell_idx_in_col = cell_idx % self.cells_per_column
        cell = self.cells[col_idx][cell_idx_in_col]

        # find the best segment in the match_seg
        best_segment = None
        max_act = -1
        for segment in cell.segments:
            if segment in self.matching_segments:
                act = segment.compute_activity(self.prev_active_cells)
                if act > max_act:
                    max_act = act
                    best_segment = segment

        # if no best segment found, create a new one
        if best_segment is None and len(cell.segments) < self.max_segments:
            best_segment = cell.add_segment()

        # most contributed segment for act_cells can enforce weights
        if best_segment is not None:
            # adapt weights
            for synapse in best_segment.synapses:
                if synapse.presynaptic_cell in self.prev_active_cells:
                    synapse.weight = min(1.0, synapse.weight + self.weight_inc)
                else:
                    synapse.weight = max(0.0, synapse.weight - self.weight_dec)

            # best segment can grow synapses
            self._grow_synapses(best_segment, self.prev_active_cells)
    
    def _grow_synapses(self, segment: Segment, presynaptic_cells: Set[int]):
        ''' best segment can grow new synapses to the prev_active_cells '''
        
        if len(segment.synapses) >= self.max_synapses:
            return
        
        # list synapse destination
        existing_presynaptic = {s.presynaptic_cell for s in segment.synapses}
        candidates = presynaptic_cells - existing_presynaptic

        # decide how many synapses to grow
        max_new = min(self.max_new_synapses,
                      len(candidates),
                      self.max_synapses - len(segment.synapses))
        
        # grow new synapses
        if max_new > 0 and candidates:
            new_presynaptic = np.random.choice(list(candidates), max_new, replace=False)
            for presynaptic_cell in new_presynaptic:
                segment.add_synapse(presynaptic_cell, 0.21)
     
    def _compute_predictions(self):
        ''' update pred_cells'''
        self.predictive_cells.clear()

        for segment in self.active_segments:
            self.predictive_cells.add(segment.cell_idx)

if __name__ == "__main__":
    CELLS_PER_COLUMN = 16

    tm = TemporalMemory(
        column_count=10,
        cells_per_column=CELLS_PER_COLUMN,
    )

    x = [
        np.array([0,1]),
        np.array([2,3]),
        np.array([4,5]),
        np.array([6,7]),
        np.array([8,9]),
        np.array([6,7]),
        np.array([4,5]),
        np.array([2,3]),
    ]

    for t, xt in enumerate(x*3):
        y = tm.compute(xt, learn=True)

        print(f"time step {t+1}")
        print(f"active columns: {y['active_columns']}")
        print(f"active cells: {y['active_cells']}")
        print(f"predictive cells: {y['predictive_cells']}")

