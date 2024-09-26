import numpy as np
from antelope import antelope_analysis

@antelope_analysis
class StefanMask:
    """
    This is the masking function for the red hex rig.
    Trials start when any LED or BUZZER turns on, and end when a port is touched or the next trial begins.
    GOCUE also signifies a trial start for Stefan.
    """

    name = 'stefan_mask'
    query = 'World'
    data = 'IntervalEvents'
    returns = {'data':np.ndarray, 'timestamps':np.ndarray}

    def run(key):

        restriction = [f'intervalevents_name = "{i}{j}"' for i in ['LED_','SENSOR'] for j in range(1,7)]
        interval_data, interval_timestamps, interval_names = (IntervalEvents & key & restriction).fetch('data','timestamps','intervalevents_name')

        namedict = {'LED_':'led','SENSOR':'sensor'}
        for data, timestamps, name in zip(interval_data, interval_timestamps, interval_names):
            locals()[f'{namedict[name[:-1]]}{name[-1]}_data'] = data
            locals()[f'{namedict[name[:-1]]}{name[-1]}_timestamps'] = timestamps

        gocue_data, gocue_timestamps = (IntervalEvents & key & 'intervalevents_name = "GO_CUE"').fetch1('data','timestamps')

        # initially concatenate all on times and all off times
        on_times = np.array([])
        for cue in [f'led{i}' for i in range(1,7)] + ['gocue']:
            data = locals()[f'{cue}_data']
            timestamps = locals()[f'{cue}_timestamps']
            start = timestamps[data == 1]
            on_times = np.concatenate((on_times,start))

        off_times = np.array([])
        for off in [f'sensor{i}' for i in range(1,7)]:
            data = locals()[f'{off}_data']
            timestamps = locals()[f'{off}_timestamps']
            start = timestamps[data == 1]
            off_times = np.concatenate((off_times,start))

        if on_times.shape[0] > 0 and off_times.shape[0] > 0:

            on_times = np.sort(on_times)
            off_times = np.sort(off_times)
            on_times = on_times[on_times < off_times[-1]]

            # the following computes the first off time after each on time
            indices = np.searchsorted(off_times, on_times, side='right') # indices of the first off time after each on time
            filtered_off_times = off_times[indices] # the first off time after each on time
            filtered_off_times = np.append(np.minimum(filtered_off_times[:-1], on_times[1:]), filtered_off_times[-1]) # minimum of the first off time after each on time and the next on time

            # compute data and timestamps arrays
            timestamps = np.sort(np.concatenate((on_times, filtered_off_times)))
            data = np.zeros(timestamps.shape[0])
            data[np.searchsorted(timestamps, on_times)] = 1
            data[np.searchsorted(timestamps, filtered_off_times)] = -1

            # need to explicitly handle the case where the on time equals the last time
            data[data == 0] = 1

            return data, timestamps

        else:
            return np.array([]), np.array([])


@antelope_analysis
class DanMask:
    """
    This is the masking function for the red hex rig.
    Trials start when any LED or BUZZER turns on, and end when a port is touched or the next trial begins.
    """

    name = 'dan_mask'
    query = 'World'
    data = 'IntervalEvents'
    returns = {'data':np.ndarray, 'timestamps':np.ndarray}

    def run(key):

        restriction = [f'intervalevents_name = "{i}{j}"' for i in ['LED_','SENSOR'] for j in range(1,7)]
        interval_data, interval_timestamps, interval_names = (IntervalEvents & key & restriction).fetch('data','timestamps','intervalevents_name')

        namedict = {'LED_':'led','SENSOR':'sensor'}
        for data, timestamps, name in zip(interval_data, interval_timestamps, interval_names):
            locals()[f'{namedict[name[:-1]]}{name[-1]}_data'] = data
            locals()[f'{namedict[name[:-1]]}{name[-1]}_timestamps'] = timestamps

        # initially concatenate all on times and all off times
        on_times = np.array([])
        for cue in [f'led{i}' for i in range(1,7)]:
            data = locals()[f'{cue}_data']
            timestamps = locals()[f'{cue}_timestamps']
            start = timestamps[data == 1]
            on_times = np.concatenate((on_times,start))

        off_times = np.array([])
        for off in [f'sensor{i}' for i in range(1,7)]:
            data = locals()[f'{off}_data']
            timestamps = locals()[f'{off}_timestamps']
            start = timestamps[data == 1]
            off_times = np.concatenate((off_times,start))

        if on_times.shape[0] > 0 and off_times.shape[0] > 0:

            on_times = np.sort(on_times)
            off_times = np.sort(off_times)
            on_times = on_times[on_times < off_times[-1]]

            # the following computes the first off time after each on time
            indices = np.searchsorted(off_times, on_times, side='right') # indices of the first off time after each on time
            filtered_off_times = off_times[indices] # the first off time after each on time
            filtered_off_times = np.append(np.minimum(filtered_off_times[:-1], on_times[1:]), filtered_off_times[-1]) # minimum of the first off time after each on time and the next on time

            # compute data and timestamps arrays
            timestamps = np.sort(np.concatenate((on_times, filtered_off_times)))
            data = np.zeros(timestamps.shape[0])
            data[np.searchsorted(timestamps, on_times)] = 1
            data[np.searchsorted(timestamps, filtered_off_times)] = -1

            # need to explicitly handle the case where the on time equals the last time
            data[data == 0] = 1

            return data, timestamps

        else:
            return np.array([]), np.array([])
