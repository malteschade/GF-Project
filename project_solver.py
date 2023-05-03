# import libraries
from geofluids import *

import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

SHEET_ID = '1jU6eor_WTvCt3tLzQbysIg0AH-8zRwek_B0aUV76O1k'
SHEET_NAME = 'Parameters'
OUTPUT_PATH = 'output/02_blue/'

def load_setup():
    url = f'https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&sheet={SHEET_NAME}'
    return pd.read_csv(url).reset_index()

def multi_solve(input):
    s = input[1]

    if type(s['completed']) == str:
        print(f'Skipping solved row: {input[0]+2}')
        return
    
    print(f'Starting thread for row: {input[0]+2}')
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
    np.savez_compressed(f'{OUTPUT_PATH}index_{s["index"]+2}', pos_x=pos_x, pos_y=pos_y, label=label)
    
    print(f'Finishing thread for row: {input[0]+2}')
    

if __name__ == '__main__':
    # load setup
    setup = load_setup()
    
    # run solver
    start=time.time()
    with Pool() as mp_pool:
        mp = mp_pool.imap(multi_solve, setup.iterrows())
        for _ in mp:
            pass
    print(f'\nCPU time :{(time.time() - start)}')
