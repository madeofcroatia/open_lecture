import numpy as np
import bokeh.io
import bokeh.plotting
import bokeh.models
import bokeh.layouts
import bokeh.events

from math import floor
from bokeh.palettes import Plasma256
from itertools import cycle


"""class SimpleEconomy:
    def __init__(self, starting_capital, size, transaction):
        self.starting_capital = starting_capital
        self.size = size
        self.transaction = transaction

        self.max_wealth = self.starting_capital * self.size

        self.person_form = self.initiate()
        self.wealth_form = self.to_wealth_form()

    def initiate(self):
        person_form = self.starting_capital * np.ones(self.size).astype('int')
        return person_form

    def to_wealth_form(self):
        wealth_form = np.zeros(self.max_wealth).astype('int')
        for wealth in self.person_form:
            wealth_form[wealth] += 1

        return wealth_form

    def simulate(self):
        givers = filter(lambda i: self.current_state[i] > 0, range(self.size))
        for giver in givers:
            receiver = np.random.randint(0, n_people)
            self.wealth_form[self.person_form[receiver]] -= 1
            self.person_form[receiver] += 1
            self.wealth_form[self.person_form[receiver]] += 1
            self.wealth_form[self.person_form[giver]]  -= 1
            self.person_form[giver] -= 1
            self.wealth_form[self.person_form[giver]] += 1


class SocialClass:
    def __init__(self, soc_range, economy):
        self.soc_range = soc_range
        self.economy = economy

    def get_wealth(self):
        low = floor(self.soc_range[0] * self.economy.size)
        high = floor(self.soc_range[1] * self.economy.size)
        sorted_person_form = sorted(self.economy.person_form)

        class_wealth = sum(sorted_person_form[low:high])

        return class_wealth

    def get_proportion(self):
        return self.get_wealth()/self.economy.size


class AnimatedHistogram:
    def __init__(self, plot, data, x_lim=None, fill_color="navy", line_color="white"):
        if isinstance(x_lim, tuple):
            self.x_lim = x_lim
        else:
            self.x_lim = (0, len(data))

        self.x = np.linspace(self.x_lim[0], self.x_lim[1], self.x_lim[1])

        self.source = bokeh.models.ColumnDataSource(dict(data=data, x=self.x))
        self.histogram = bokeh.models.VBar(
            top='data',
            x='x',
            fill_color=fill_color,
            line_color=line_color
        )

        self.glyph = plot.add_glyph(self.source, self.glyph)

    def update(self, new_data):
        self.source.update(
            dict(data=new_data, x=self.x)
        )


class AnimatedTimeSeries:
    def __init__(self, plot, data, color):
        self.plot = plot
        self.color = color
        self.time = 0
        self.source = bokeh.models.ColumnDataSource(dict(y=data, x=[self.time]))
        self.line = self.initiate_line()

        self.glyph = plot.add_glyph(self.source, self.line)

    def initiate_line(self):
        line_model = bokeh.models.Line(
            y='y',
            x='x',
            line_color=self.color
        )
        return line_model

    def stream(self, new_data):
        self.time += 1
        self.source.stream(
            dict(y=new_data, x=[self.time])
        )


class SocialClassTimeSeries(AnimatedTimeSeries):
    def __init__(self, plot, soc_class, color):
        AnimatedTimeSeries.__init__(self, plot, data, color)
        self.plot = plot
        self.soc_class = soc_class
        self.color = color



class SocialClassLines:
    def __init__(self, plot, soc_classes):
        self.plot = plot
        self.soc_classes = soc_classes
        self.lines = self.make_lines()

    def init_line(self, soc_class, color):
        data = soc_class.get_proportion()
        line = AnimatedTimeSeries(self.plot, data, color)
        return line

    def make_lines(self):
        colors = cycle(Plasma256)
        lines = []
        for soc_class in self.soc_classes:
            color = next(colors)
            line = self.init_line(soc_class, color)
            lines.append(line)
        return lines

    def stream(self):"""



def plot_form(original):
    new_form = np.zeros(max_money)
    for i in original:
        new_form[i] += 1
    return new_form


def get_top(percent):
    people = sorted(enumerate(original_form), reverse=True)
    n = floor(n_people * percent)
    top = []
    wealth = 0
    for cash, i in people:
        top.append(i)
        wealth += cash
        n -= 1
        if n == 0:
            break
    return top


def get_bottom50():
    n = floor(n_people * 0.5)
    wealth = 0
    for cash in sorted(original_form):
        wealth += cash
        n -= 1
        if n == 0:
            break
    return 100 * wealth/max_money


time = 0
n_people = 100
starting_cash = 3
max_money = n_people * starting_cash

original_form = starting_cash * np.ones(n_people).astype('int')
wealth_form = plot_form(original_form)

document = bokeh.plotting.curdoc()
x = np.linspace(0, max_money, max_money)
paused = True
callback = None

source = bokeh.models.ColumnDataSource(
    dict(top=wealth_form, x=x)
)

person_source = bokeh.models.ColumnDataSource(
    dict(top=original_form, x=np.linspace(0, n_people, n_people))
)

top10_source = bokeh.models.ColumnDataSource(
    dict(top10_x=[0], top10_y=[sum(get_top(0.1))])
)

bottom50_source = bokeh.models.ColumnDataSource(
    dict(bottom50_x=[0], bottom50_y=[get_bottom50()])
)

top5_source = bokeh.models.ColumnDataSource(
    dict(top5_x=[0], top5_y=[sum(get_top(0.05))])
)

wealth_plot = bokeh.plotting.figure(
    width=1800,
    height=600,
    x_axis_label='wealth',
    y_axis_label='number of people',
    x_range=(-1, min(max_money + 1, 15*starting_cash)),
    y_range=(0, n_people)
)

person_plot = bokeh.plotting.figure(
    width=1000,
    height=600,
    x_axis_label='person',
    y_axis_label='wealth',
    x_range=(-1, n_people + 1),
    y_range=(0, starting_cash*20)
)

inequality_plot = bokeh.plotting.figure(
    width=600,
    height=600,
    x_axis_label="time",
    y_axis_label="wealth %",
    y_range=(0, 100)
)

wealth_dist = bokeh.models.VBar(
    top="top",
    x="x",
    fill_color="navy",
    line_color="white"
)

person_dist = bokeh.models.VBar(
    top="top",
    x="x",
    fill_color="navy",
    line_color="white"
)

top10 = bokeh.models.Line(
    x="top10_x",
    y="top10_y",
    line_color="#ff5733"
)

top5 = bokeh.models.Line(
    x="top5_x",
    y="top5_y",
    line_color="#b233ff"
)

bottom50 = bokeh.models.Line(
    x="bottom50_x",
    y="bottom50_y",
    line_color="#33ff36"
)


wealth_plot.add_glyph(source, wealth_dist)
person_plot.add_glyph(person_source, person_dist)
inequality_plot.add_glyph(top10_source, top10)
inequality_plot.add_glyph(top5_source, top5)
inequality_plot.add_glyph(bottom50_source, bottom50)

legend = bokeh.models.Legend(
    items=[
        bokeh.models.LegendItem(label="the wealth proportion of the richest 10%",
                                renderers=[inequality_plot.renderers[0]]),
        bokeh.models.LegendItem(label="the wealth proportion of the poorest 50%",
                                renderers=[inequality_plot.renderers[2]])
    ],
    location="top_left",
    click_policy="hide"
)

inequality_plot.add_layout(legend)


def on_tap():
    global paused, callback
    if paused:
        paused = False
        callback = document.add_periodic_callback(update, 10)
    else:
        paused = True
        document.remove_periodic_callback(callback)


def update_original_form():
    givers = filter(lambda i: original_form[i] > 0, range(n_people))
    for giver in givers:
        receiver = np.random.randint(0, n_people)
        original_form[receiver] += 1
        original_form[giver] -= 1

    person_source.update(
        data=dict(top=original_form, x=np.linspace(0, n_people, n_people))
    )


def update_wealth_form():
    global wealth_form
    wealth_form = plot_form(original_form)

    source.update(
        data=dict(top=wealth_form, x=x)
    )


def update_inequality():
    global time
    time += 1
    red_top = get_top(0.1)
    new_top10 = dict(
        top10_x=[time],
        top10_y=[sum(red_top)]
    )
    yellow_top = get_top(0.05)
    new_top5 = dict(
        top5_x=[time],
        top5_y=[sum(yellow_top)]
    )
    new_bottom50 = dict(
        bottom50_x=[time],
        bottom50_y=[get_bottom50()]
    )
    top10_source.stream(new_top10)
    top5_source.stream(new_top5)
    bottom50_source.stream(new_bottom50)


def update():
    update_original_form()
    update_wealth_form()
    update_inequality()


wealth_plot.on_event(bokeh.events.Tap, on_tap)
layout = bokeh.layouts.column(wealth_plot, inequality_plot, person_plot)
document.add_root(layout)
