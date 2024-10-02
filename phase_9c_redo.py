from antelope import antelope_analysis, split_trials
from matplotlib import pyplot as plt
import numpy as np
import datetime
import pandas as pd
from pathlib import Path
import datajoint as dj
from scipy.stats import ttest_ind
import statsmodels.api as sm
from statsmodels.multivariate.manova import MANOVA

plt.style.use('tableau-colorblind10')


@antelope_analysis
class AnglePlot:
    """
    Plots success, impulsive, and right port rates as a function of head angle for 13/05/2024 and 15/05/2024 sessions.
    """

    name = 'angle_plot'
    query = 'Experiment'
    data = ['Session', 'Self', 'Animal']
    calls = ['session_stats']
    returns = {
        'Success':plt.Figure, 'Impulsive':plt.Figure, 'Right Port':plt.Figure,
        'Success Behind':plt.Figure, 'Impulsive Behind':plt.Figure, 'Right Port Behind':plt.Figure,
        'Success In Front':plt.Figure, 'Impulsive In Front':plt.Figure, 'Right Port In Front':plt.Figure
               }
    key = {'experimenter':'dwelch', 'experiment_id':1}

    def run(key):

        # questions for dan - loads of sessions per animal on each day
        # do we want 1000ms?
        # session range?
        # currently just summing across all sessions across all animals
        # manova? : each angle separate, 3 dependent variables, days are independent variables

        # fetch data
        df = (Session * Self * Animal & key).proj('session_notes','session_timestamp','animal_name').fetch(format='frame')
        df.reset_index(inplace=True)

        # loop through two dates and compute success rates by angle
        bindf = []
        for date in [13, 15]:

            # filter to all that are 9c and 1000ms 13/05
            start_date = datetime.datetime(2024, 5, date, 0, 0, 0)
            end_date = datetime.datetime(2024, 5, date + 1, 0, 0, 0)
            date_df = df[(df['session_notes'].str.contains('phase:9c')) & (df['session_notes'].str.contains('wait:1000')) & (df['session_timestamp'] >= start_date) & (df['session_timestamp'] <= end_date)]

            # loop through array and append success stats
            for i, row in date_df.iterrows():
                key = {key:row.to_dict()[key] for key in ['experimenter','experiment_id','session_id']}
                session_stats_dict = session_stats(key, session_range=[0.25,1])['session_stats']
                for k, v in session_stats_dict.items():
                    if v['total']:
                        bindf.append({'angle':int(k), 
                                    'Success':v['success'] / v['total'], 
                                    'Impulsive':v['impulsive'] / v['total'], 
                                    'Right_Port':v['right_port'] / v['total'],
                                    'date':date})
                        if int(k) == 15: # hacky way to get circle closed
                            bindf.append({'angle':375, 
                                        'Success':v['success'] / v['total'], 
                                        'Impulsive':v['impulsive'] / v['total'], 
                                        'Right_Port':v['right_port'] / v['total'],
                                        'date':date})

        df = pd.DataFrame(bindf)

        # compute manova statistics for each angle
        manova_results = {}
        for angle in df['angle'].unique():
            temp_df = df[df['angle'] == angle]
            manova = MANOVA.from_formula('Success + Impulsive + Right_Port ~ date', data=temp_df)
            manova_results[angle] = manova.mv_test()['date']['stat']['Pr > F']['Wilks\' lambda']

        df = df.groupby(['angle','date']).mean().reset_index()
        df.rename(columns={'Right_Port':'Right Port'}, inplace=True)
        # loop through 3 plots
        plot_dict = {}
        labels = {13:'Day 1', 15:'Day 3'}
        for feature in ['Success', 'Impulsive', 'Right Port']:

            # radial plot
            fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
            for date in [13, 15]:
                temp_df = df[df['date'] == date]
                ax.plot(temp_df['angle'] * np.pi / 180, temp_df[feature], label=labels[date])
            ax.set_title(f'{feature} Rates')
            ax.grid(True)
            ax.set_ylim(0, 1)
            for angle, p_value in manova_results.items():
                print(angle, p_value)
                ax.annotate(sig_value(p_value), xy=(np.radians(angle), 0.9),
                            ha='center', va='center', fontsize=8)
            fig.legend()
            ax.set_aspect('equal')
            ax.set_theta_zero_location('N')
            plot_dict[feature] = fig

            # in front / behind plot
            for position in ['Behind', 'In Front']:

                temp_df = df[(df['angle'] > 90) & (df['angle'] < 270)] if position == 'Behind' else df[~((df['angle'] > 90) & (df['angle'] < 270))] # varies across angle and date
                day1 = temp_df[temp_df['date'] == 13]
                day3 = temp_df[temp_df['date'] == 15]
                t_stat, p_value = ttest_ind(day1[feature].values, day3[feature].values)
                fig, ax = plt.subplots()
                sig_label = sig_value(p_value)
                ax.annotate(sig_label, xy=(1.5, 1),
                            xytext=(0, 3), textcoords="offset points", ha='center', fontsize=12)
                ax.axhline(y=0.95,xmin=0.25,xmax=0.75, color='black')
                ax.bar(1, day1.mean()[feature], label='Day 1')
                ax.bar(2, day3.mean()[feature], label='Day 3')
                ax.set_ylim(0, 1.1)
                ax.set_yticks(np.arange(0, 1.2, 0.2))
                ax.set_xticks([])
                ax.set_title(f'{feature} {position}')
                fig.legend()
                plot_dict[f'{feature} {position}'] = fig

        return tuple(plot_dict.values())


def sig_value(p_value):
    if p_value < 0.001:
        sig_label = '***'
    elif p_value < 0.01:
        sig_label = '**'
    elif p_value < 0.05:
        sig_label = '*'
    else:
        sig_label = 'n.s.'
    return sig_label


@antelope_analysis
class SessionStats:
    """
    Computes led, sensor, gocue and head angle for each trial for a given session, returns sums binned by angle.
    """

    name = 'session_stats'
    query = 'Session'
    data = ['Mask', 'IntervalEvents', 'Kinematics']
    args = {'session_range':list, 'buffer':int} # list of two floats giving the range of trials to consider
    returns = {'session_stats':float}
    hidden = True

    def run(key, session_range=[0,0], buffer=5):

        # fetch data
        restriction = [f'intervalevents_name = "{i}{j}"' for i in ['LED_','SENSOR'] for j in range(1,7)] + ['intervalevents_name = "GO_CUE"']
        interval_data, interval_timestamps, interval_names, x_coordinates, y_coordinates = (IntervalEvents & key & restriction).fetch('data','timestamps','intervalevents_name','x_coordinate','y_coordinate')
        mask_data, mask_timestamps = (Mask & key & 'mask_name = "DanMask"').fetch1('data','timestamps')
        restriction = [f'kinematics_name = "{i}_ear"' for i in ['right','left']]
        kinematics_data, kinematics_timestamps, kinematics_names = (Kinematics & key & restriction).fetch('data','timestamps','kinematics_name')

        # split all inputs into trials
        trials = {}
        coords = {}
        for data, timestamps, name, x, y in zip(interval_data, interval_timestamps, interval_names, x_coordinates, y_coordinates):
            for i in range(1,7):
                if name == f'LED_{i}':
                    trials[f'led{i}'] = split_trials((data, timestamps), (mask_data, mask_timestamps))
                    coords[name] = (x,y)
                if name == f'SENSOR{i}':
                    trials[f'sensor{i}'] = split_trials((data, timestamps), (mask_data, mask_timestamps))
            if name == 'GO_CUE':
                trials['gocue'] = split_trials((data, timestamps), (mask_data, mask_timestamps))
        for data, timestamps, name in zip(kinematics_data, kinematics_timestamps, kinematics_names):
            for i in ['right','left']:
                if name == f'{i}_ear':
                    trials[f'{i}_ear'] = split_trials((data, timestamps), (mask_data, mask_timestamps))

        # figure out trial statistics and bin by head angle
        session_stats = {}
        for i in range(0, 360, 30):
            session_stats[i+15] = {'success':0, 'impulsive':0, 'right_port':0, 'total':0, 'timeout':0}
        length = len(trials['led1'])
        for trial in range(int(length*session_range[0]), int(length*session_range[1])):
            try:
                # extract data from trial
                led = None
                sensor = None
                for i in range(1,7):
                    if trials[f'led{i}'][trial][0].size > 0 and trials[f'led{i}'][trial][0][0] == 1:
                        led = i
                    if trials[f'sensor{i}'][trial][0].size > 0 and trials[f'sensor{i}'][trial][0][0] == 1:
                        sensor = i
                gocue = (trials['gocue'][trial][0].size >= 1)

                # compute head angle
                right_ear = trials['right_ear'][trial][0]
                left_ear = trials['left_ear'][trial][0]
                right_x, right_y = right_ear[:buffer,0], right_ear[:buffer,1]
                left_x, left_y = left_ear[:buffer,0], left_ear[:buffer,1]
                delta_x = right_x - left_x
                delta_y = right_y - left_y
                head_angle = np.mean(np.degrees(np.arctan2(-delta_y, delta_x))) + 90 # minus since camera coords

                # compute angle to LED
                led_x, led_y = coords[f'LED_{led}']
                head_centre_x, head_centre_y = (right_x + left_x) / 2, (right_y + left_y) / 2
                delta_x = led_x - head_centre_x
                delta_y = led_y - head_centre_y
                led_angle = np.mean(np.degrees(np.arctan2(-delta_y, delta_x))) # minus since camera coords

                # compute relative angle and normalise to be in range 0-360
                angle = (led_angle - head_angle) % 360

                if not np.isnan(angle):
                    key = int(angle // 30 * 30) + 15

                    # if no sensor, trial is a timeout and we skip
                    if sensor is None:
                        session_stats[key]['timeout'] += 1
                        continue

                    # determine if trial is successful, impulsive, and gets right port
                    success, impulsive, right_port = trial_success(led, sensor, gocue)

                    # bin by head angle
                    session_stats[key]['success'] += success
                    session_stats[key]['impulsive'] += impulsive
                    session_stats[key]['right_port'] += right_port
                    session_stats[key]['total'] += 1


            except IndexError:
                pass

        return session_stats


@antelope_analysis
class SessionStats:
    """
    Computes led, sensor, gocue and head angle for each trial for a given session, returns sums binned by angle.
    """

    name = 'session_stats'
    query = 'Session'
    data = ['Mask', 'IntervalEvents', 'Kinematics']
    args = {'session_range':list, 'buffer':int} # list of two floats giving the range of trials to consider
    returns = {'session_stats':float}
    hidden = True

    def run(key, session_range=[0,0], buffer=5):

        # fetch data
        restriction = [f'intervalevents_name = "{i}{j}"' for i in ['LED_','SENSOR'] for j in range(1,7)] + ['intervalevents_name = "GO_CUE"']
        interval_data, interval_timestamps, interval_names, x_coordinates, y_coordinates = (IntervalEvents & key & restriction).fetch('data','timestamps','intervalevents_name','x_coordinate','y_coordinate')
        mask_data, mask_timestamps = (Mask & key & 'mask_name = "DanMask"').fetch1('data','timestamps')
        restriction = [f'kinematics_name = "{i}_ear"' for i in ['right','left']]
        kinematics_data, kinematics_timestamps, kinematics_names = (Kinematics & key & restriction).fetch('data','timestamps','kinematics_name')

        # split all inputs into trials
        trials = {}
        coords = {}
        for data, timestamps, name, x, y in zip(interval_data, interval_timestamps, interval_names, x_coordinates, y_coordinates):
            for i in range(1,7):
                if name == f'LED_{i}':
                    trials[f'led{i}'] = split_trials((data, timestamps), (mask_data, mask_timestamps))
                    coords[name] = (x,y)
                if name == f'SENSOR{i}':
                    trials[f'sensor{i}'] = split_trials((data, timestamps), (mask_data, mask_timestamps))
            if name == 'GO_CUE':
                trials['gocue'] = split_trials((data, timestamps), (mask_data, mask_timestamps))
        for data, timestamps, name in zip(kinematics_data, kinematics_timestamps, kinematics_names):
            for i in ['right','left']:
                if name == f'{i}_ear':
                    trials[f'{i}_ear'] = split_trials((data, timestamps), (mask_data, mask_timestamps))

        # figure out trial statistics and bin by head angle
        session_stats = {}
        for i in range(0, 360, 30):
            session_stats[i+15] = {'success':0, 'impulsive':0, 'right_port':0, 'total':0, 'timeout':0}
        length = len(trials['led1'])
        for trial in range(int(length*session_range[0]), int(length*session_range[1])):
            try:
                # extract data from trial
                led = None
                sensor = None
                for i in range(1,7):
                    if trials[f'led{i}'][trial][0].size > 0 and trials[f'led{i}'][trial][0][0] == 1:
                        led = i
                    if trials[f'sensor{i}'][trial][0].size > 0 and trials[f'sensor{i}'][trial][0][0] == 1:
                        sensor = i
                gocue = (trials['gocue'][trial][0].size >= 1)

                # compute head angle
                right_ear = trials['right_ear'][trial][0]
                left_ear = trials['left_ear'][trial][0]
                right_x, right_y = right_ear[:buffer,0], right_ear[:buffer,1]
                left_x, left_y = left_ear[:buffer,0], left_ear[:buffer,1]
                delta_x = right_x - left_x
                delta_y = right_y - left_y
                head_angle = np.mean(np.degrees(np.arctan2(-delta_y, delta_x))) + 90 # minus since camera coords

                # compute angle to LED
                led_x, led_y = coords[f'LED_{led}']
                head_centre_x, head_centre_y = (right_x + left_x) / 2, (right_y + left_y) / 2
                delta_x = led_x - head_centre_x
                delta_y = led_y - head_centre_y
                led_angle = np.mean(np.degrees(np.arctan2(-delta_y, delta_x))) # minus since camera coords

                # compute relative angle and normalise to be in range 0-360
                angle = (led_angle - head_angle) % 360

                if not np.isnan(angle):
                    key = int(angle // 30 * 30) + 15

                    # if no sensor, trial is a timeout and we skip
                    if sensor is None:
                        session_stats[key]['timeout'] += 1
                        continue

                    # determine if trial is successful, impulsive, and gets right port
                    success, impulsive, right_port = trial_success(led, sensor, gocue)

                    # bin by head angle
                    session_stats[key]['success'] += success
                    session_stats[key]['impulsive'] += impulsive
                    session_stats[key]['right_port'] += right_port
                    session_stats[key]['total'] += 1


            except IndexError:
                pass

        return session_stats


def trial_success(led, sensor, gocue):
    """
    Function determines if a trial is successful, impulsive, and gets right port.
    Input: led (int), sensor (int), gocue (bool)
    Output: success, impulsive, right_port
    """

    # impulsive if the gocue goes off during the trial
    impulsive = not gocue

    # right_port if led equals sensor
    right_port = led == sensor

    # true success if impulsive and right port
    success = (not impulsive) and right_port

    return success, impulsive, right_port