''' Present an interactive function explorer with slider widgets.
Scrub the sliders to change the properties of the ``sin`` curve, or
type into the title text box to update the title of the plot.
Use the ``bokeh serve`` command to run the example by executing:
    bokeh serve sliders.py
at your command prompt. Then navigate to the URL
    http://localhost:5006/sliders
in your browser.
'''
import numpy as np

from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Slider, TextInput, Text, LabelSet
from bokeh.plotting import figure
import multi_mt as mm

# Set up parameters
N_S_j = 3
current_N = 0

# Set up data
N = 200
x = np.linspace(-2., 5., N)
y = mm.combined(x[:, None], [], S_c=0)

source_prob = ColumnDataSource(data=dict(x=x, y=y[0].reshape(-1)))


# Set up plot
plot_prob = figure(height=500, width=700, title="Order Fill Probability",
                   tools="crosshair,pan,reset,save,wheel_zoom",
                   x_range=[-2, 5], y_range=[-0.2, 1.2], x_axis_label='S_i')
plot_prob.title.text_font_size = "25px"
plot_prob.title.align = "center"

plot_prob.line('x', 'y', source=source_prob, line_width=3, line_alpha=0.6)


source_reward = ColumnDataSource(data=dict(x=x, y=y[1].reshape(-1)))
source_reward_old = ColumnDataSource(data=dict(x=x, y=y[2].reshape(-1)))

plot_reward = figure(height=500, width=700, title="expected reward",
                     tools="crosshair,pan,reset,save,wheel_zoom",
                     x_range=[-2, 5], y_range=[-2.5, 2.5])
plot_reward.title.text_font_size = "25px"
plot_reward.title.align = "center"

plot_reward.line('x', 'y', source=source_reward, line_width=3,
                 line_alpha=0.6, legend_label="true cost $m_t$", line_color="green")
# plot_reward.line('x', 'y', source=source_reward_old,
#                  line_width=3, line_alpha=0.6, legend_label="$cost C_i""true cost m_t", line_color="orange")


# Set up widgets
# text = TextInput(title="title", value='my sine wave')
S_c = Slider(title="S_c", value=0.0, start=-1.0, end=3.0, step=0.05)
N_S = Slider(title="Number of S_js", value=0, start=0, end=3, step=1)

sigma = Slider(title="sigma", value=0.3, start=0, end=1.5, step=0.02)
tau = Slider(title="tau", value=0.2, start=0, end=1.5, step=0.02)
rho = Slider(title="rho", value=0.0, start=0, end=0.98, step=0.02)

S_sliders = []
for i in range(N_S_j):
    S_sliders.append(
        Slider(title=f"S_{i}", value=0.0, start=-1.0, end=3.0, step=0.05))
    S_sliders[-1].visible = False


source_labels = ColumnDataSource(data=dict(
    x=[0],
    y=[0],
    names=['S_c']
))

plot_prob.cross(x='x', y='y', size=20, source=source_labels)
plot_reward.cross(x='x', y='y', size=20, source=source_labels)

labels_obj = LabelSet(x='x', y='y', text='names',
                      x_offset=5, y_offset=-30, source=source_labels)
plot_prob.add_layout(labels_obj)
plot_reward.add_layout(labels_obj)

# Set up callbacks


def update_plot():
    S_j = []
    for i in range(N_S.value):
        S_j.append(S_sliders[i].value)

    x = np.linspace(-2., 5., N)
    y = mm.combined(x[:, None], S_j, S_c=S_c.value,
                    sigma=sigma.value, tau=tau.value, rho=rho.value)

    X_s = [S_c.value]
    Y_s = [0]
    names = ['S_c']
    for i in range(N_S.value):
        X_s.append(S_sliders[i].value)
        Y_s.append(0)
        names.append(f'S_{i}')

    source_prob.data = dict(x=x, y=y[0].reshape(-1))
    source_reward.data = dict(x=x, y=y[1].reshape(-1))
    source_reward_old.data = dict(x=x, y=y[2].reshape(-1))
    source_labels.data = dict(x=X_s, y=Y_s, names=names)
    # labels_obj = LabelSet(x='x', y='y', text='names',
    #                       x_offset=5, y_offset=-20, source=source_labels)
    # plot_prob.add_layout(labels_obj)
    # plot_reward.add_layout(labels_obj)


def update_N(attrname, old, new):
    for i in range(old, new):
        S_sliders[i].visible = True
    for i in range(old, new, -1):
        S_sliders[i-1].visible = False
    update_plot()


def update_data(attrname, old, new):
    update_plot()


N_S.on_change('value_throttled', update_N)
for w in [S_c, sigma, tau, rho]:
    w.on_change('value_throttled', update_data)
for w in S_sliders:
    w.on_change('value_throttled', update_data)


# Set up layouts and add to document
inputs = column(N_S, S_c, sigma, tau, rho, *
                S_sliders, margin=(20, 20, 20, 40))

curdoc().add_root(row(inputs, plot_prob, plot_reward, width=800))
curdoc().title = "Sliders"
