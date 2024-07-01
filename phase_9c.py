from antelope import antelope_analysis, split_trials
from matplotlib import pyplot as plt
import numpy as np
import datetime
import pandas as pd
from pathlib import Path
import datajoint as dj
from scipy.stats import ttest_ind

plt.style.use('tableau-colorblind10')


@antelope_analysis
class ChemoTime:
    """
    Function filters which sessions are phase 9c.
    Then plots success rates by angle.
    """

    name = 'chemo_time_plot'
    query = 'Experiment'
    returns = {'plot':plt.Figure}

    def run(key):

        exclusion_list = [383,385,387,384,386,388,379,380] # corrupted session ids

        # filter to all that are between 17/05 and 19/50
        start_date = datetime.datetime(2024, 5, 17, 0, 0, 0)
        end_date = datetime.datetime(2024, 5, 21, 0, 0, 0)

        # fetch data
        controldf = (Session * Self * Animal & key & f'session_timestamp < "{end_date}"' & f'session_timestamp > "{start_date}"').proj('session_notes','session_timestamp','animal_name').fetch(format='frame')
        controldf.reset_index(inplace=True)
        controldf = controldf[~controldf['session_id'].isin(exclusion_list)]
        controldf['date'] = controldf['session_timestamp'].dt.date

        # filter to all that are between 17/05 and 19/50
        start_date = datetime.datetime(2024, 5, 21, 0, 0, 0)
        end_date = datetime.datetime(2024, 5, 25, 0, 0, 0)

        # fetch data
        activedf = (Session * Self * Animal & key & f'session_timestamp < "{end_date}"' & f'session_timestamp > "{start_date}"').proj('session_notes','session_timestamp','animal_name').fetch(format='frame')
        activedf.reset_index(inplace=True)
        activedf = activedf[~activedf['session_id'].isin(exclusion_list)]
        activedf['date'] = activedf['session_timestamp'].dt.date

        # loop through animals and dates and compute stats
        df = []
        for i, group in controldf.groupby(['animal_name','date']):
            group = group.sort_values('session_timestamp').reset_index(drop=True)
            for j, row in group.iterrows():
                if j == 0: # first 10 mins is control
                    continue
                key = {key:row.to_dict()[key] for key in ['experimenter','experiment_id','session_id']}
                stats = phase_9c_session_stats_no_angle(key)['session_stats']
                stat_dict = {'success':0, 'non_impulsive':0, 'right_port':0, 'total':0}
                for trial in stats:
                    for k, v in trial.items():
                        stat_dict[k] += v
                    stat_dict['total'] += 1
                df.append({'animal_name':i[0],'date':i[1],'session_number':j,**stat_dict})
        for i, group in activedf.groupby(['animal_name','date']):
            if i[1] == datetime.date(2024, 5, 22): # this day is corrupted
                continue
            group = group.sort_values('session_timestamp').reset_index(drop=True)
            for j, row in group.iterrows():
                if j == 0: # first 10 mins is control
                    continue
                key = {key:row.to_dict()[key] for key in ['experimenter','experiment_id','session_id']}
                stats = phase_9c_session_stats_no_angle(key)['session_stats']
                stat_dict = {'success':0, 'non_impulsive':0, 'right_port':0, 'total':0}
                for trial in stats:
                    for k, v in trial.items():
                        stat_dict[k] += v
                    stat_dict['total'] += 1
                df.append({'animal_name':i[0],'date':i[1],'session_number':j,**stat_dict})
        df = pd.DataFrame(df)

        # compute success rates
        df['success'] = df['success'] / df['total']
        df['non_impulsive'] = df['non_impulsive'] / df['total']
        df['right_port'] = df['right_port'] / df['total']

        # average and std over animals
        df = df.groupby(['date','session_number']).agg({'success':['mean','std'], 'non_impulsive':['mean','std'], 'right_port':['mean','std']}).reset_index()

        # plot
        fig, ax = plt.subplots()
        testdf = df[df['session_number'] == 2]
        controldf = df[df['session_number'] == 1]
        testdf['session_number'] = testdf['date'].rank(method='first').astype(int)
        controldf['session_number'] = controldf['date'].rank(method='first').astype(int)
        ax.errorbar(testdf['session_number'], testdf['success']['mean'], testdf['success']['std'], marker='x', label='Test', capsize=5)
        ax.errorbar(controldf['session_number'], controldf['success']['mean'], controldf['success']['std'], marker='o', label='Control', capsize=5)
        ax.set_title('Success Rate',y=1.05)
        ax.set_xlabel('Session Number')
        ax.set_ylabel('Rate')
        ax.legend()
        yticks = np.arange(0,1.1,0.1)
        ax.set_yticks(yticks)
        vlines = [3.5]
        ax.axvline(vlines, linestyle='--', linewidth=0.7)
        ax.text(2.2, 1.02, 'Saline',ha='center')
        ax.text(4.8, 1.02, 'DCZ',ha='center')

        fig.savefig(Path.home() / 'Documents' / 'dan_plots' / 'chemo_timeseries.png', dpi=300)


@antelope_analysis
class Chemo:
    """
    Function filters which sessions are phase 9c.
    Then plots success rates by angle.
    """

    name = 'chemo_plot'
    query = 'Experiment'
    returns = {'plot':plt.Figure}

    def run(key):

        exclusion_list = [383,385,387,384,386,388,379,380] # corrupted session ids

        # filter to all that are between 17/05 and 19/50
        start_date = datetime.datetime(2024, 5, 17, 0, 0, 0)
        end_date = datetime.datetime(2024, 5, 21, 0, 0, 0)

        # fetch data
        controldf = (Session * Self * Animal & key & f'session_timestamp < "{end_date}"' & f'session_timestamp > "{start_date}"').proj('session_notes','session_timestamp','animal_name').fetch(format='frame')
        controldf.reset_index(inplace=True)
        controldf = controldf[~controldf['session_id'].isin(exclusion_list)]
        controldf['date'] = controldf['session_timestamp'].dt.date

        base_ls, control_ls = [], []
        # loop by animal and date
        for i, group in controldf.groupby(['animal_name', 'date']):
            group = group.sort_values('session_timestamp').reset_index(drop=True)
            base_dict = {'success':0, 'non_impulsive':0, 'right_port':0, 'total':0, 'date':i[1], 'animal_name':i[0]}
            control_dict = {'success':0, 'non_impulsive':0, 'right_port':0, 'total':0, 'date':i[1], 'animal_name':i[0]}

            # compute success stats for second and third sessions
            basekey = {key:group.iloc[1].to_dict()[key] for key in ['experimenter','experiment_id','session_id']}
            controlkey = {key:group.iloc[2].to_dict()[key] for key in ['experimenter','experiment_id','session_id']}
            base_stats = phase_9c_session_stats_no_angle(basekey)['session_stats']
            control_stats = phase_9c_session_stats_no_angle(controlkey)['session_stats']

            # aggregate stats across dates
            for trial in base_stats:
                for k, v in trial.items():
                    base_dict[k] += int(v)
                base_dict['total'] += 1
            for trial in control_stats:
                for k, v in trial.items():
                    control_dict[k] += int(v)
                control_dict['total'] += 1
            base_ls.append(base_dict)
            control_ls.append(control_dict)

        control_df = pd.DataFrame(control_ls)
        control_df.reset_index(inplace=True)
        base_df = pd.DataFrame(base_ls)
        base_df.reset_index(inplace=True)

        # filter to all that are between 17/05 and 19/50
        start_date = datetime.datetime(2024, 5, 21, 0, 0, 0)
        end_date = datetime.datetime(2024, 5, 25, 0, 0, 0)

        # fetch data
        activedf = (Session * Self * Animal & key & f'session_timestamp < "{end_date}"' & f'session_timestamp > "{start_date}"').proj('session_notes','session_timestamp','animal_name').fetch(format='frame')
        activedf.reset_index(inplace=True)
        activedf = activedf[~activedf['session_id'].isin(exclusion_list)]
        activedf['date'] = activedf['session_timestamp'].dt.date

        active_ls, activebase_ls = [], []
        for i, group in activedf.groupby(['animal_name', 'date']):
            if i[1] == datetime.date(2024, 5, 22):
                # this day is corrupted
                continue
            group = group.sort_values('session_timestamp').reset_index(drop=True)
            active_dict = {'success':0, 'non_impulsive':0, 'right_port':0, 'total':0, 'date':i[1], 'animal_name':i[0]}
            activebase_dict = {'success':0, 'non_impulsive':0, 'right_port':0, 'total':0, 'date':i[1], 'animal_name':i[0]}
            activekey = {key:group.iloc[1].to_dict()[key] for key in ['experimenter','experiment_id','session_id']}
            activebasekey = {key:group.iloc[2].to_dict()[key] for key in ['experimenter','experiment_id','session_id']}
            active_stats = phase_9c_session_stats_no_angle(activekey)['session_stats']
            activebase_stats = phase_9c_session_stats_no_angle(activebasekey)['session_stats']
            for trial in active_stats:
                for k, v in trial.items():
                    active_dict[k] += int(v)
                active_dict['total'] += 1
            for trial in activebase_stats:
                for k, v in trial.items():
                    activebase_dict[k] += int(v)
                activebase_dict['total'] += 1
            active_ls.append(active_dict)
            activebase_ls.append(activebase_dict)

        active_df = pd.DataFrame(active_ls)
        active_df.reset_index(inplace=True)
        activebase_df = pd.DataFrame(activebase_ls)
        activebase_df.reset_index(inplace=True)

        # compute success rates
        control_df['success'] = control_df['success'] / control_df['total']
        control_df['non_impulsive'] = control_df['non_impulsive'] / control_df['total']
        control_df['right_port'] = control_df['right_port'] / control_df['total']
        base_df['success'] = base_df['success'] / base_df['total']
        base_df['non_impulsive'] = base_df['non_impulsive'] / base_df['total']
        base_df['right_port'] = base_df['right_port'] / base_df['total']
        active_df['success'] = active_df['success'] / active_df['total']
        active_df['non_impulsive'] = active_df['non_impulsive'] / active_df['total']
        active_df['right_port'] = active_df['right_port'] / active_df['total']
        activebase_df['success'] = activebase_df['success'] / activebase_df['total']
        activebase_df['non_impulsive'] = activebase_df['non_impulsive'] / activebase_df['total']
        activebase_df['right_port'] = activebase_df['right_port'] / activebase_df['total']

        for j in range(1,3):
            if j == 1: # cohort 1
                tmpcontrol_df = control_df[control_df['animal_name'].str[-2] == '4']
                tmpbase_df = base_df[base_df['animal_name'].str[-2] == '4']
                tmpactive_df = active_df[active_df['animal_name'].str[-2] == '4']
                tmpactivebase_df = activebase_df[activebase_df['animal_name'].str[-2] == '4']
            elif j == 2:
                tmpcontrol_df = control_df[control_df['animal_name'].str[-2] != '4']
                tmpbase_df = base_df[base_df['animal_name'].str[-2] != '4']
                tmpactive_df = active_df[active_df['animal_name'].str[-2] != '4']
                tmpactivebase_df = activebase_df[activebase_df['animal_name'].str[-2] != '4']

            tmpcontrol_df.sort_values(['date', 'animal_name'], inplace=True)
            tmpbase_df.sort_values(['date', 'animal_name'], inplace=True)
            tmpactive_df.sort_values(['date', 'animal_name'], inplace=True)
            tmpactivebase_df.sort_values(['date', 'animal_name'], inplace=True)
            tmpcontrol_df.reset_index(inplace=True)
            tmpbase_df.reset_index(inplace=True)
            tmpactive_df.reset_index(inplace=True)
            tmpactivebase_df.reset_index(inplace=True)

            def plot_df(df1, df2, stat, ticklabels, title, filename):
                fig, ax = plt.subplots()
                t_stat, p_val = ttest_ind(df1[stat], df2[stat])
                df1 = df1.groupby('animal_name').mean()
                df2 = df2.groupby('animal_name').mean()
                df1.reset_index(inplace=True)
                df2.reset_index(inplace=True)
                pos = (df1[stat].mean() + df2[stat].mean()) / 2
                ax.text(0.5, 0.975, 'n.s.', color='black', ha='center')
                ax.text(0.5, 0.1, f'p = {p_val:.2f}', color='black', ha='center')
                for i, row in df1.iterrows():
                    ax.plot([0,1], [row[stat], df2.iloc[i][stat]], label=row['animal_name'], color='grey')
                ax.bar([0,1], [df1[stat].mean(), df2[stat].mean()], yerr=[df1[stat].std(), df2[stat].std()], capsize=5, width=0.4, alpha=0.7)
                ax.set_xticks([0,1])
                ax.set_xticklabels(ticklabels)
                ax.set_title(title)
                fig.savefig(Path.home() / 'Documents' / 'dan_plots' / filename, dpi=300)

            plot_df(tmpbase_df, tmpcontrol_df, 'success', ['Pre Saline', 'After Saline'], 'Saline Success Rate', f'control_vs_base_cohort{j}.png')
            plot_df(tmpactivebase_df, tmpactive_df, 'success', ['Pre DCZ', 'After DCZ'], 'DCZ Success Rate', f'active_vs_activebase_cohort{j}.png')
            plot_df(tmpcontrol_df, tmpactive_df, 'success', ['Saline', 'DCZ'], 'Saline vs DCZ Success Rate', f'control_vs_active_cohort{j}.png')
            plot_df(tmpbase_df, tmpcontrol_df, 'non_impulsive', ['Pre Saline', 'After Saline'], 'Saline non_impulsive Rate', f'saline_control_impulsive_cohort{j}.png')
            plot_df(tmpactivebase_df, tmpactive_df, 'non_impulsive', ['Pre DCZ', 'After DCZ'], 'DCZ non_impulsive Rate', f'dcz_control_impulsive_cohort{j}.png')
            plot_df(tmpbase_df, tmpcontrol_df, 'right_port', ['Pre Saline', 'After Saline'], 'Saline right_port Rate', f'saline_control_port_cohort{j}.png')
            plot_df(tmpactivebase_df, tmpactive_df, 'right_port', ['Pre DCZ', 'After DCZ'], 'DCZ right_port Rate', f'dcz_control_port_cohort_{j}.png')


@antelope_analysis
class AnglePlot13:
    """
    Function filters which sessions are phase 9c.
    Then plots success rates by angle.
    """

    name = 'angle_plot_13'
    query = 'Experiment'
    returns = {'plot':plt.Figure}

    def run(key):

        # fetch data
        df = (Session * Self * Animal & key).proj('session_notes','session_timestamp','animal_name').fetch(format='frame')
        df.reset_index(inplace=True)
        
        # filter to all that are 9c and 1000ms 15/05
        start_date = datetime.datetime(2024, 5, 13, 0, 0, 0)
        end_date = datetime.datetime(2024, 5, 14, 0, 0, 0)
        df = df[(df['session_notes'].str.contains('phase:9c')) & (df['session_notes'].str.contains('wait:1000')) & (df['session_timestamp'] >= start_date) & (df['session_timestamp'] <= end_date)]

        bindf = []

        # loop through array and append success stats
        for i, group in df.groupby('animal_name'):
            group.sort_values('session_timestamp', inplace=True)
            group.reset_index(inplace=True)
            row = group.iloc[1]
            key = {key:row.to_dict()[key] for key in ['experimenter','experiment_id','session_id']}
            session_stats = phase_9c_session_stats(key, session_range=[0,0.5])['session_stats']
            for k, v in session_stats.items():
                bindf.append({'angle':int(k), 
                            'success':v['success'] / v['total'] if v['total'] > 0 else 0,
                            'non_impulsive':v['non_impulsive'] / v['total'] if v['total'] > 0 else 0,
                            'right_port':v['right_port'] / v['total'] if v['total'] > 0 else 0})
                if int(k) == 15: # hacky way to get circle closed
                    bindf.append({'angle':375, 
                                'success':v['success'] / v['total'], 
                                'non_impulsive':v['non_impulsive'] / v['total'], 
                                'right_port':v['right_port'] / v['total']})

        df = pd.DataFrame(bindf)
        df = df.groupby('angle').mean().reset_index()

        # create plot
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        ax.plot(df['angle'] * np.pi / 180, df['success'], label='Success')
        ax.plot(df['angle'] * np.pi / 180, df['non_impulsive'], label='Non-Impulsive')
        ax.plot(df['angle'] * np.pi / 180, df['right_port'], label='Correct Port')

        ax.set_title('13/05/24 first half success rates')  # Set title
        ax.grid(True)  # Show a grid
        fig.legend()  # Show the legend
        ax.set_aspect('equal')
        ax.set_theta_zero_location('N')

        fig.savefig(Path.home() / 'Documents' / 'dan_plots' / '13th_plot.png', dpi=300)


@antelope_analysis
class TimeoutAnglePlot:
    """
    Function filters which sessions are phase 9c.
    Then plots success rates by angle.
    """

    name = 'timeoutangle_plot'
    query = 'Experiment'
    returns = {'plot':plt.Figure}

    def run(key):

        # fetch data
        df = (Session * Self * Animal & key).proj('session_notes','session_timestamp','animal_name').fetch(format='frame')
        df.reset_index(inplace=True)
        
        # filter to all that are 9c and 1000ms 15/05
        start_date = datetime.datetime(2024, 5, 15, 0, 0, 0)
        end_date = datetime.datetime(2024, 5, 16, 0, 0, 0)
        df = df[(df['session_notes'].str.contains('phase:9c')) & (df['session_notes'].str.contains('wait:1000')) & (df['session_timestamp'] >= start_date) & (df['session_timestamp'] <= end_date)]

        bindf = []

        # loop through array and append success stats
        for i, row in df.iterrows():
            print(row)
            key = {key:row.to_dict()[key] for key in ['experimenter','experiment_id','session_id']}
            session_stats = phase_9c_session_stats(key, session_range=[0.25,1])['session_stats']
            for k, v in session_stats.items():
                bindf.append({'angle':int(k), 
                            'timeout':v['timeout'] / v['total']})
                if int(k) == 15: # hacky way to get circle closed
                    bindf.append({'angle':375, 
                                'timeout':v['timeout'] / v['total']})

        df = pd.DataFrame(bindf)
        df = df.groupby('angle').mean().reset_index()

        # create plot
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        ax.plot(df['angle'] * np.pi / 180, df['timeout'], label='Timeout')

        ax.set_title('15/05/24 timeout rates')  # Set title
        ax.grid(True)  # Show a grid
        ax.set_aspect('equal')
        ax.set_ylim(0, 1)
        ax.set_theta_zero_location('N')

        fig.savefig(Path.home() / 'Documents' / 'dan_plots' / 'timeout_plot.png', dpi=300)


@antelope_analysis
class AnglePlot:
    """
    Function filters which sessions are phase 9c.
    Then plots success rates by angle.
    """

    name = 'angle_plot'
    query = 'Experiment'
    returns = {'plot':plt.Figure}

    def run(key):

        # fetch data
        df = (Session * Self * Animal & key).proj('session_notes','session_timestamp','animal_name').fetch(format='frame')
        df.reset_index(inplace=True)
        
        # filter to all that are 9c and 1000ms 15/05
        start_date = datetime.datetime(2024, 5, 15, 0, 0, 0)
        end_date = datetime.datetime(2024, 5, 16, 0, 0, 0)
        df = df[(df['session_notes'].str.contains('phase:9c')) & (df['session_notes'].str.contains('wait:1000')) & (df['session_timestamp'] >= start_date) & (df['session_timestamp'] <= end_date)]

        bindf = []

        # loop through array and append success stats
        for i, row in df.iterrows():
            key = {key:row.to_dict()[key] for key in ['experimenter','experiment_id','session_id']}
            session_stats = phase_9c_session_stats(key, session_range=[0.25,1])['session_stats']
            for k, v in session_stats.items():
                bindf.append({'angle':int(k), 
                              'success':v['success'] / v['total'], 
                              'non_impulsive':v['non_impulsive'] / v['total'], 
                              'right_port':v['right_port'] / v['total']})
                if int(k) == 15: # hacky way to get circle closed
                    bindf.append({'angle':375, 
                                'success':v['success'] / v['total'], 
                                'non_impulsive':v['non_impulsive'] / v['total'], 
                                'right_port':v['right_port'] / v['total']})

        df = pd.DataFrame(bindf)
        df = df.groupby('angle').mean().reset_index()

        # create plot
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        ax.plot(df['angle'] * np.pi / 180, df['success'], label='Success')
        ax.plot(df['angle'] * np.pi / 180, df['non_impulsive'], label='Non-Impulsive')
        ax.plot(df['angle'] * np.pi / 180, df['right_port'], label='Correct Port')

        ax.set_title('15/05/24 success rates')  # Set title
        ax.grid(True)  # Show a grid
        fig.legend()  # Show the legend
        ax.set_aspect('equal')
        ax.set_theta_zero_location('N')

        fig.savefig(Path.home() / 'Documents' / 'dan_plots' / '15th_plot.png', dpi=300)


@antelope_analysis
class Training:
    """
    Function filters which sessions are phase 9c.
    Then sorts by animal and calculates success rates.
    """

    name = 'training'
    query = 'Experiment'
    returns = {'plot':plt.Figure}

    def run(key):

        # fetch data
        df = (Session * Self * Animal & key).proj('session_notes','session_timestamp','animal_name').fetch(format='frame')
        df.reset_index(inplace=True)
        
        # filter to all that are 9c and 1000ms 13/05 - 15/05
        start_date = datetime.datetime(2024, 5, 13, 0, 0, 0)
        end_date = datetime.datetime(2024, 5, 16, 0, 0, 0)
        df = df[(df['session_notes'].str.contains('phase:9c')) & (df['session_notes'].str.contains('wait:1000')) & (df['session_timestamp'] >= start_date) & (df['session_timestamp'] <= end_date)]
        df = df.sort_values('session_timestamp')

        # loop through animals
        animal_stats = []
        for animal_name, group in df.groupby('animal_name'):
            group = group.sort_values('session_timestamp').reset_index(drop=True)

            # loop and aggregate by date
            date_dict = {}
            for i, row in group.iterrows():

                # compute session stats
                key = {key:row.to_dict()[key] for key in ['experimenter','experiment_id','session_id']}
                session_stats = phase_9c_session_stats_no_angle(key)['session_stats']

                if len(session_stats) > 10:
                    if row['session_timestamp'].date() not in date_dict:
                        date_dict[row['session_timestamp'].date()] = session_stats
                    else:
                        date_dict[row['session_timestamp'].date()]+= session_stats

            # loop through dates and split virtual sessions in four, and aggregate statistics
            for date, stats in date_dict.items():
                total_trials = len(stats)
                quarter = total_trials // 4

                for i in range(4):
                    if i == 3:
                        quarter_stats = stats[i*quarter:]
                    else:
                        quarter_stats = stats[i*quarter:(i+1)*quarter]
                    total = len(quarter_stats)
                    success = sum([x['success'] for x in quarter_stats]) / total
                    non_impulsive = sum([x['non_impulsive'] for x in quarter_stats]) / total
                    right_port = sum([x['right_port'] for x in quarter_stats]) / total
                    animal_stats.append({'animal_name':animal_name, 'date':date, 'quarter':i+1, 'success':success, 'non_impulsive':non_impulsive, 'right_port':right_port})


        df = pd.DataFrame(animal_stats)

        # fill in blank data
        animals = df['animal_name'].unique()
        dates = df['date'].unique()
        quarters = df['quarter'].unique()
        template = pd.DataFrame([(animal, date, quarter) for animal in animals for date in dates for quarter in quarters],
                        columns=['animal_name', 'date', 'quarter'])
        df = pd.merge(template, df, on=['animal_name', 'date', 'quarter'], how='left')


        # plot all animals on same plot
        plots = ['success', 'non_impulsive', 'right_port']
        plot_titles = ['Success Rate', 'Non-Impulsive Rate', 'Right Port Rate']
        plot_dict = {}
        for i in range(3):
            plot_dict[plots[i]] = plt.subplots()
        for animal_name, group in df.groupby('animal_name'):
            group = group.sort_values('date').reset_index(drop=True)
            group['Session Number'] = group['date'].rank(method='first').astype(int)
            for i in range(3):
                plot_dict[plots[i]][1].plot(group['Session Number'], group[plots[i]], label=animal_name)
        x_ticks = [2.5 + i*4 for i in range(len(group) // 4)]
        y_ticks = np.arange(0,1.1,0.1)
        vlines = [4.5 + i*4 for i in range(len(group) // 4)]
        x_labels = list(range(1,4))
        for i in range(3):
            plot_dict[plots[i]][1].set_xticks(ticks=x_ticks, labels=x_labels)
            plot_dict[plots[i]][1].set_yticks(ticks=y_ticks)
            for line in vlines:
                plot_dict[plots[i]][1].axvline(x=line, linestyle='--', linewidth=0.7)
            plot_dict[plots[i]][1].set_title(plot_titles[i])
            plot_dict[plots[i]][1].set_xlabel('Session Number')
            plot_dict[plots[i]][1].set_ylabel('Rate')
            plot_dict[plots[i]][0].legend(loc='lower right', bbox_to_anchor=(0.9, 0.1))

        for key, plot in plot_dict.items():
            save = Path.home() / 'Documents' / 'dan_plots' / f'{key}.png'
            plot[0].savefig(save, dpi=300)


        # plot mean and std across animals
        fig, ax = plt.subplots()
        mean_df = df.groupby(['date','quarter']).agg({'success':['mean','std'], 'non_impulsive':['mean','std'], 'right_port':['mean','std']}).reset_index()
        mean_df['Session Number'] = mean_df['date'].rank(method='first').astype(int)
        mean_plot_dict = {}
        for i in range(3):
            mean_plot_dict[plots[i]] = plt.subplots()
            mean_plot_dict[plots[i]][1].plot(mean_df['Session Number'], mean_df[plots[i]]['mean'])
            mean_plot_dict[plots[i]][1].fill_between(mean_df['Session Number'], mean_df[plots[i]]['mean'] - mean_df[plots[i]]['std'], mean_df[plots[i]]['mean'] + mean_df[plots[i]]['std'], alpha=0.2)
            x_ticks = [2.5 + i*4 for i in range(len(mean_df) // 4)]
            vlines = [4.5 + i*4 for i in range(len(mean_df) // 4)]
            x_labels = list(range(1,4))
            y_ticks = np.arange(0,1.1,0.1)
            mean_plot_dict[plots[i]][1].set_yticks(ticks=y_ticks)
            mean_plot_dict[plots[i]][1].set_xticks(ticks=x_ticks, labels=x_labels)
            for line in vlines:
                mean_plot_dict[plots[i]][1].axvline(x=line, linestyle='--', linewidth=0.7)
            mean_plot_dict[plots[i]][1].set_title(plot_titles[i])
            mean_plot_dict[plots[i]][1].set_xlabel('Session Number')
            mean_plot_dict[plots[i]][1].set_ylabel('Rate')

        for key, plot in mean_plot_dict.items():
            save = Path.home() / 'Documents' / 'dan_plots' / f'{key}_mean.png'
            plot[0].savefig(save, dpi=300)



@antelope_analysis
class SessionStatsNoAngle:
    """
    Computes led, sensor and gocue for each trial for a given session.
    """

    name = 'session_stats_no_angle'
    query = 'Session'
    returns = {'session_stats':list}

    def run(key):

        # fetch data
        restriction = [f'intervalevents_name = "{i}{j}"' for i in ['LED_','SENSOR'] for j in range(1,7)] + ['intervalevents_name = "GO_CUE"']
        interval_data, interval_timestamps, interval_names, x_coordinates, y_coordinates = (IntervalEvents & key & restriction).fetch('data','timestamps','intervalevents_name','x_coordinate','y_coordinate')
        mask_data, mask_timestamps = (Mask & key & 'mask_name = "trials"').fetch1('data','timestamps')
        restriction = [f'kinematics_name = "{i}_ear"' for i in ['right','left']]
        kinematics_data, kinematics_timestamps, kinematics_names = (Kinematics & key & restriction).fetch('data','timestamps','kinematics_name')

        # split all inputs into trials
        trials = {}
        for data, timestamps, name, x, y in zip(interval_data, interval_timestamps, interval_names, x_coordinates, y_coordinates):
            for i in range(1,7):
                if name == f'LED_{i}':
                    trials[f'led{i}'] = split_trials((data, timestamps), (mask_data, mask_timestamps))
                if name == f'SENSOR{i}':
                    trials[f'sensor{i}'] = split_trials((data, timestamps), (mask_data, mask_timestamps))
            if name == 'GO_CUE':
                trials['gocue'] = split_trials((data, timestamps), (mask_data, mask_timestamps))
        for data, timestamps, name in zip(kinematics_data, kinematics_timestamps, kinematics_names):
            for i in ['right','left']:
                if name == f'{i}_ear':
                    trials[f'{i}_ear'] = split_trials((data, timestamps), (mask_data, mask_timestamps))

        # figure out trial statistics
        session_stats = []
        for trial in range(len(trials['led1'])):
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

                # if no sensor, trial is a timeout and we skip
                if sensor is None:
                    continue

                # determine if trial is successful, non-impulsive, and gets right port
                success, non_impulsive, right_port = trial_success(led, sensor, gocue)

                session_stats.append({'success':success, 'non_impulsive':non_impulsive, 'right_port':right_port})

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
    args = {'session_range':list, 'buffer':int} # list of two floats giving the range of trials to consider
    returns = {'session_stats':float}

    def run(key, session_range=[0,0], buffer=5):

        # fetch data
        restriction = [f'intervalevents_name = "{i}{j}"' for i in ['LED_','SENSOR'] for j in range(1,7)] + ['intervalevents_name = "GO_CUE"']
        interval_data, interval_timestamps, interval_names, x_coordinates, y_coordinates = (IntervalEvents & key & restriction).fetch('data','timestamps','intervalevents_name','x_coordinate','y_coordinate')
        mask_data, mask_timestamps = (Mask & key & 'mask_name = "trials"').fetch1('data','timestamps')
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
            session_stats[i+15] = {'success':0, 'non_impulsive':0, 'right_port':0, 'total':0, 'timeout':0}
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

                    # determine if trial is successful, non-impulsive, and gets right port
                    success, non_impulsive, right_port = trial_success(led, sensor, gocue)

                    # bin by head angle
                    session_stats[key]['success'] += success
                    session_stats[key]['non_impulsive'] += non_impulsive
                    session_stats[key]['right_port'] += right_port
                    session_stats[key]['total'] += 1


            except IndexError:
                pass

        return session_stats


def trial_success(led, sensor, gocue):
    """
    Function determines if a trial is successful, non-impulsive, and gets right port.
    Input: led (int), sensor (int), gocue (bool)
    Output: success, non-impulsive, right_port
    """

    # non-impulsive if the gocue goes off during the trial
    non_impulsive = gocue

    # right_port if led equals sensor
    right_port = led == sensor

    # true success if non-impulsive and right port
    success = non_impulsive and right_port

    return success, non_impulsive, right_port
