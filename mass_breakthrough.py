# import libraries
from geofluids import *

import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

N_SEED = 32

IMIN = 1
IMAX = 2

XMIN = 199
XMAX = 201
YMIN = 0
YMAX = 100

OUTPUT_PATH = 'bt_data/'


def multi_solve(input):
    s = input[1]
    print(f'Starting thread for seed: {input[0]}')
    lnk = Perm2D(s['mean_lnk'], s['variance_lnk'], s['dx'], s['correlation_length'], s['random_seed'])
    
    solver = RWPT_solver()
    solver.set_permeability_field(s['perm_field']**lnk)
    solver.solve_flow_field()
    solver.generate_velocity_interpolators()
    solver.set_longitudinal_dispersivity(s['long_disp'])
    solver.set_transversal_dispersivity(s['trans_disp'])
    solver.generate_dispersivitiy_interpolators()
    solver.set_num_particles(s['num_particles'])
    solver.set_specie_labels(s['species_1'], s['species_2'], s['species_3'])
    solver.set_time_steps(s['t_steps'])
    solver.set_save_interval(s['save_interval'])
    solver.set_initial_particle_position()
    
    pos_x, pos_y, label = solver.solve()
    
    pos_x_mask = (pos_x > XMIN) & (pos_x < XMAX) 
    pos_y_mask = (pos_y > YMIN) & (pos_y < YMAX)
    mask = pos_x_mask & pos_y_mask
    
    c = np.zeros((int(s['t_steps']/s['save_interval'])+1, 3))
    for i in range(len(label)):
        c[i, 0] = np.count_nonzero(label[i][mask[i]] == s['species_1'])
        c[i, 1] = np.count_nonzero(label[i][mask[i]] == s['species_2'])
        c[i, 2] = np.count_nonzero(label[i][mask[i]] == s['species_3'])
        
    print(f'Finishing thread for seed: {input[0]}')
    return c
    

if __name__ == '__main__':
    # load setup
    setup = pd.read_excel('param_table.xlsx', sheet_name='Parameters', header=0)[IMIN:IMAX]
    
    print(f'-------- Starting pipeline with {len(setup)} setups --------')
    for i in range(len(setup)):
        start=time.time()
        print(f'\n-------- Calculating setup {i} --------')
        p_setup = pd.concat([setup.iloc[i]]*N_SEED, axis=1).transpose().reset_index(drop=True)
        p_setup['random_seed'] = np.arange(N_SEED)
    
        with Pool() as mp_pool:
            mp = mp_pool.map(multi_solve, p_setup.iterrows())
            c = np.sum(mp, axis=0) * 1/((XMAX-XMIN)*(YMAX-YMIN)) * (1/N_SEED)
    
        # save data
        np.savez_compressed(f'{OUTPUT_PATH}setup_{i}', c=c)
        
        stop = time.time()
        print(f'\nCPU time :{(stop - start)}')
        print(f'Avg-CPU time :{(stop - start)/N_SEED}')
        print(f'-------- Finished setup {i} --------\n')
    print('-------- Finished pipeline --------')