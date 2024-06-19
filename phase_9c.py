from antelope import antelope_analysis, split_trials
from matplotlib import pyplot as plt
import numpy as np

@antelope_analysis
class Main:
    """
    Function filters which sessions are phase 9c.
    Then sorts by animal and calculates success rates.
    """

    name = 'phase_9c'
    inputs = [
        {'table':'Experiment.Session',
        'attribute':['primary_key', 'session_notes'],
        'name':['primary_keys','session_notes']},

        {'table': 'Experiment.Self',
            'attribute': 'animal_id'}
    ]
    returns = {'plot':plt.Figure}
    inherits = ['get_animal_name', 'phase_9c_session_stats']

    def run(primary_keys, session_notes, animal_id, get_animal_name, phase_9c_session_stats):

        # get all primary keys that are 9c and 1000ms and group by animal
        phase_9c = []
        for key, note, animal_id in zip(primary_keys, session_notes, animal_id):
            notedict = {split.split(':')[0]:split.split(':')[1] for split in note.split('; ')}
            if notedict['phase'] == '9c': # also check wait time is 1000ms and make phase 9c
                phase_9c.append({**key,'animal_id':animal_id})

        animal_dict = {}
        for key in phase_9c:
            animal_name = get_animal_name(key)['animal_name']
            if animal_name not in animal_dict:
                animal_dict[animal_name] = []
            del key['animal_id']
            animal_dict[animal_name].append(key)

        # calculate success rate
        session_stats = {}
        for animal, keys in animal_dict.items():
            for key in keys:
                stats = phase_9c_session_stats(key)
                if animal not in session_stats:
                    session_stats[animal] = []
                print(stats)

        ## plot success rates by animal
        #fig, ax = plt.subplots()
        #for animal, rates in success_rates.items():
        #    ax.plot(rates, label=animal)
        #ax.legend()

        return fig



@antelope_analysis
class AnimalName:
    """
    Function gets animal name from session primary key.
    """
    name = 'get_animal_name'
    inputs = [
        {'table':'Animal', 'attribute':'animal_name', 'name':'animal_name'}
    ]
    returns = {'animal_name':str}
    def run(animal_name):
        return animal_name


@antelope_analysis
class SessionStats:
    """
    Computes led, sensor, gocue and head angle for each trial for a given session.
    """

    name = 'phase_9c_session_stats'
    inputs = [
        {'table':'Session.IntervalEvents', 'attribute': ['data', 'timestamps', 'intervalevents_name'], 'name':['led_data', 'led_timestamps','led_names'], 'restriction': [f'intervalevents_name = "LED_{i}"' for i in range(1,7)]},
        {'table':'Session.IntervalEvents', 'attribute': ['data', 'timestamps', 'intervalevents_name'], 'name':['sensor_data', 'sensor_timestamps','sensor_names'], 'restriction': [f'intervalevents_name = "SENSOR{i}"' for i in range(1,7)]},
        {'table':'Session.IntervalEvents', 'attribute': ['data', 'timestamps'], 'name':['gocue_data', 'gocue_timestamps'], 'restriction': 'intervalevents_name = "GO_CUE"'},
        {'table':'Session.Kinematics', 'attribute': ['data', 'timestamps', 'kinematics_name'], 'name':['kinematics_data', 'kinematics_timestamps', 'kinematics_names'], 'restriction': [f'kinematics_name = "{i}_ear"' for i in ['right','left']]},
        {'table':'Session.Mask', 'attribute':['data', 'timestamps'], 'name':['mask_data', 'mask_timestamps'], 'restriction':{'mask_name':'trials'}}
    ]
    returns = {'success_rate':float}

    def run(led_data, led_timestamps, led_names, sensor_data, sensor_timestamps, sensor_names, gocue_data, gocue_timestamps, kinematics_data, kinematics_timestamps, kinematics_names, mask_data, mask_timestamps):

        # split all inputs into trials
        mask_data, mask_timestamps = mask_data[0], mask_timestamps[0]
        trials = {}
        for data, timestamps, name in zip(led_data, led_timestamps, led_names):
            for i in range(1,7):
                if name == f'LED_{i}':
                    trials[f'led{i}'] = split_trials((data, timestamps), (mask_data, mask_timestamps))
        for data, timestamps, name in zip(sensor_data, sensor_timestamps, sensor_names):
            for i in range(1,7):
                if name == f'SENSOR{i}':
                    trials[f'sensor{i}'] = split_trials((data, timestamps), (mask_data, mask_timestamps))
        trials['gocue'] = split_trials((gocue_data[0], gocue_timestamps[0]), (mask_data, mask_timestamps))
        for data, timestamps, name in zip(kinematics_data, kinematics_timestamps, kinematics_names):
            for i in ['right','left']:
                if name == f'{i}_ear':
                    trials[f'{i}_ear'] = split_trials((data, timestamps), (mask_data, mask_timestamps))

        # figure out which led, sensor, and gocue are in each trial
        # and calculate head angle
        trial_stats = []
        for trial in range(len(trials['led1'])):
            led = None
            sensor = None
            for i in range(1,7):
                if trials[f'led{i}'][trial][0].size > 0 and trials[f'led{i}'][trial][0][0] == 1:
                    led = i
                if trials[f'sensor{i}'][trial][0].size > 0 and trials[f'sensor{i}'][trial][0][0] == 1:
                    sensor = i
            gocue = (trials['gocue'][trial][0].size >= 1)

            right_ear = trials['right_ear'][trial][0]
            left_ear = trials['left_ear'][trial][0]
            right_x, right_y = right_ear[0,0], right_ear[0,1]
            left_x, left_y = left_ear[0,0], left_ear[0,1]
            delta_x = right_x - left_x
            delta_y = right_y - left_y
            head_angle = np.degrees(np.arctan2(-delta_y, delta_x))

            trial_stats.append({'led':led, 'sensor':sensor, 'gocue':gocue, 'head_angle':head_angle})

        return trial_stats
