from antelope import antelope_analysis, split_trials
from matplotlib import pyplot as plt

@antelope_analysis
class Main:
    """
    Function filters which sessions are phase 3c.
    Then sorts by animal and calculates success rates.
    """

    name = 'phase_3c'
    inputs = [
        {'table':'Experiment.Session',
        'attribute':['primary_key', 'session_notes'],
        'name':['primary_keys','session_notes'],
        'restriction':{'experimenter':'srogers'}},

        {'table': 'Experiment.Self',
            'attribute': 'animal_id'}
    ]
    returns = {'response':str}
    inherits = ['get_animal_name', 'phase_3c_success']

    def run(primary_keys, session_notes, animal_id, get_animal_name, phase_3c_success):

        # get all primary keys that are 3c and 1000ms and group by animal
        phase_3c = []
        for key, note, animal_id in zip(primary_keys, session_notes, animal_id):
            notedict = {split.split(':')[0]:split.split(':')[1] for split in note.split('; ')}
            if notedict['phase'] == '2': # also check wait time is 1000ms and make phase 3c
                phase_3c.append({**key,'animal_id':animal_id})

        # group by animal
        animal_dict = {}
        for key in phase_3c:
            animal_name = get_animal_name(key)['animal_name']
            if animal_name not in animal_dict:
                animal_dict[animal_name] = []
            del key['animal_id']
            animal_dict[animal_name].append(key)

        # calculate success rate
        success_rates = {}
        for animal, keys in animal_dict.items():
            for key in keys:
                try: # obviously remove later, just because some masks failed
                    success_rate = phase_3c_success(key)
                    if animal not in success_rates:
                        success_rates[animal] = []
                    success_rates[animal].append(success_rate['success_rate'])
                except Exception:
                    pass

        # plot success rates by animal
        plt.figure()
        for animal, rates in success_rates.items():
            plt.plot(rates, label=animal)



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
class Success:
    """
    Trial is succesful if valve1 opens.
    """

    name = 'phase_3c_success'
    inputs = [
        {'table':'Session.DigitalEvents', 'attribute': ['data', 'timestamps'], 'name':['valve_data', 'valve_timestamps'], 'restriction':{'digitalevents_name':'VALVE1'}},
        {'table':'Session.Mask', 'attribute':['data', 'timestamps'], 'name':['mask_data', 'mask_timestamps'], 'restriction':{'mask_name':'trials'}}
    ]
    returns = {'success_rate':float}

    def run(valve_data, valve_timestamps, mask_data, mask_timestamps):

        trials = split_trials((valve_data[0], valve_timestamps[0]), (mask_data[0], mask_timestamps[0]))

        success = [len(t[0]) > 0 for t in trials]
        if len(success) == 0:
            return 0
        else:
            success_rate = sum(success)/len(success)
            return success_rate
