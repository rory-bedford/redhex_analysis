import matplotlib.pyplot as plt

custom_colors = {
    'primary': '#00cdf9',  # Cyan
    'secondary': '#ff00a6',  # Magenta
    'tertiary': '#ff8a00',  # Orange
    'background': '#ffffff',  # White background
    'grid': '#a0a0a0',  # Light gray for grid
}

plt.rcParams['axes.prop_cycle'] = plt.cycler(color=[
    custom_colors['primary'],
    custom_colors['secondary'],
    custom_colors['tertiary'],
])

plt.rcParams['axes.labelcolor'] = 'black'
plt.rcParams['xtick.color'] = 'black'
plt.rcParams['ytick.color'] = 'black'
