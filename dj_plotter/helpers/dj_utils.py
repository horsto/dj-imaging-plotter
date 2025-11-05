### Small helpers for datajoint related functions
import sys
import copy
from datetime import datetime
from hashlib import sha512
from collections import OrderedDict

from tqdm.auto import tqdm 
import numpy as np

import datajoint as dj 
#### LOAD DATABASE #########################################

# Load base schema
schema = dj.schema(dj.config['dj_imaging.database'])
schema.spawn_missing_classes()

# Load personal schema 
imhotte = dj.schema('user_horsto_imaging')
imhotte.spawn_missing_classes()


def make_multi_recording_object_dict(key):
    ''' 
    Loop over session / object configurations for one metasession and 
    collect session KEYs in dictionary for 
    - Base session 
    - Object session 1
    - Object session 2 (or object session 1 second object)


    This is used in the code for object vector cell identification (table OVC()) 
    and in the plotting class to identify sessions. 
    
    '''
    
    # Take care of object sessions
    # Careful, ArenaObjectPos must be fully populated, all objects must be registered
    recording_dict = (Recording.RecordingType * \
                    ArenaObjectPos & \
                    'session_name = "{}"'.format(key['session_name'])).fetch('KEY', order_by='recording_order ASC')
    
    if len(recording_dict) > 2:
        raise NotImplementedError(f'{len(recording_dict)} object sessions / objects found, but > 2 not supported')
    if len(recording_dict) < 2:
        raise NotImplementedError(f'Only one recording found. OV cell calculations need >1.')
    # Loop over sessions / objects and extract session dictionaries
    sessions = {}
    for sess in recording_dict:
        if 'object1' not in sessions.keys():
            sessions['object1'] = sess
        else:
            sessions['object2'] = sess
        
    # Add base session
    sessions['base'] = (Recording.RecordingType & \
                        'recording_name = "{}"'.format(key['recording_name']) & \
                        'recording_type = "Open Field"').fetch1('KEY')
        
    return sessions


def recording_title_string(recording_name):
    # Create a string for display in the title of napari window or elsewhere
    recording_hash, animal_name, timestamp = (Recording & f'recording_name = "{recording_name}"').fetch1('recording_name','animal_name','timestamp')
    timestamp = datetime.strftime(timestamp, '%d.%m.%Y')
    title_string = 'Recording {} | Animal {} | {}'.format(recording_hash, animal_name, timestamp)
    return title_string 


#### ADDITIONAL HELPERS 

def get_signal_indices(signal, reference):
    '''
    Given two arrays: 
    - signal
    - reference
    
    where signal is a subset of reference,
    return the indices for every entry in signal in reference.
    
    This is useful if one wants to look up for example the 
    exact times when a signal (filtered in SignalTracking)
    occurred. 
    
    e.g. 
    get_signal_indices(y_pos_signal, y_pos), where 'y_pos_signal'
    is the filtered y_pos retrieved from SignalTracking and 'y_pos' 
    is the original y_pos that was used in SignalTracking, 
    returns the indices of y_pos_signal in y_pos.    
    
    See: 
    https://stackoverflow.com/questions/8251541/
    numpy-for-every-element-in-one-array-find-the-index-in-another-array
    
    Parameter
    ---------
    signal     : np.array
                 Array of data points to look up 
                 in "reference"
    reference  : np.array
                 Reference array that "signal"
                 should be compared with
                 
    Returns
    -------
    indices    : np.array
                 Indices of "signal" in "reference"
    
    '''
    x = reference
    y = signal

    index = np.argsort(x)
    sorted_x = x[index]
    sorted_index = np.searchsorted(sorted_x, y)

    yindex = np.take(index, sorted_index, mode="clip")
    mask = x[yindex] != y

    result = np.ma.array(yindex, mask=mask)
    indices = result.data

    assert len(np.unique(indices)) == len(indices) == len(signal)

    return indices