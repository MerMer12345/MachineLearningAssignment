
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# from tkinter import *
# Explicit imports to satisfy Flake8
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage

from matplotlib.figure import Figure
from matplotlib.pyplot import figure

OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path("assets/frame0")


def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)


window = Tk()

window.geometry("1300x800")
window.configure(bg = "#1B1833")


canvas = Canvas(
    window,
    bg = "#1B1833",
    height = 800,
    width = 1300,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge"
)

canvas.place(x = 0, y = 0)
image_image_1 = PhotoImage(
    file=relative_to_assets("image_1.png"))
image_1 = canvas.create_image(
    180.0,
    116.0,
    image=image_image_1
)

canvas.create_text(
    41.0,
    100.0,
    anchor="nw",
    text="Name:",
    fill="#FFFFFF",
    font=("InriaSerif Regular", 20 * -1)
)

input_field = Entry(window)
input_field_window = canvas.create_window(485.0, 116.0, window=input_field)


canvas.create_text(
    273.0,
    100.0,
    anchor="nw",
    text="Monthly Income:",
    fill="#FFFFFF",
    font=("InriaSerif Regular", 20 * -1)
)

image_image_3 = PhotoImage(
    file=relative_to_assets("image_3.png"))
image_3 = canvas.create_image(
    749.0,
    116.0,
    image=image_image_3
)

canvas.create_text(
    557.0,
    100.0,
    anchor="nw",
    text="Electricity Bill:",
    fill="#FFFFFF",
    font=("InriaSerif Regular", 20 * -1)
)

image_image_4 = PhotoImage(
    file=relative_to_assets("image_4.png"))
image_4 = canvas.create_image(
    167.0,
    166.0,
    image=image_image_4
)

canvas.create_text(
    41.0,
    150.0,
    anchor="nw",
    text="Netflix:",
    fill="#FFFFFF",
    font=("InriaSerif Regular", 20 * -1)
)

image_image_5 = PhotoImage(
    file=relative_to_assets("image_5.png"))
image_5 = canvas.create_image(
    294.0,
    216.0,
    image=image_image_5
)

canvas.create_text(
    41.0,
    200.0,
    anchor="nw",
    text="Savings for Property:",
    fill="#FFFFFF",
    font=("InriaSerif Regular", 20 * -1)
)

image_image_6 = PhotoImage(
    file=relative_to_assets("image_6.png"))
image_6 = canvas.create_image(
    568.0,
    216.0,
    image=image_image_6
)

canvas.create_text(
    357.0,
    200.0,
    anchor="nw",
    text="Monthly Outing:",
    fill="#FFFFFF",
    font=("InriaSerif Regular", 20 * -1)
)

image_image_7 = PhotoImage(
    file=relative_to_assets("image_7.png"))
image_7 = canvas.create_image(
    1041.0,
    216.0,
    image=image_image_7
)

canvas.create_text(
    926.0,
    200.0,
    anchor="nw",
    text="Date:",
    fill="#FFFFFF",
    font=("InriaSerif Regular", 20 * -1)
)

image_image_8 = PhotoImage(
    file=relative_to_assets("image_8.png"))
image_8 = canvas.create_image(
    845.0,
    216.0,
    image=image_image_8
)

canvas.create_text(
    638.0,
    200.0,
    anchor="nw",
    text="Other Expenses:",
    fill="#FFFFFF",
    font=("InriaSerif Regular", 20 * -1)
)

image_image_9 = PhotoImage(
    file=relative_to_assets("image_9.png"))
image_9 = canvas.create_image(
    425.0,
    166.0,
    image=image_image_9
)

canvas.create_text(
    229.0,
    150.0,
    anchor="nw",
    text="Amazon Prime:",
    fill="#FFFFFF",
    font=("InriaSerif Regular", 20 * -1)
)

image_image_10 = PhotoImage(
    file=relative_to_assets("image_10.png"))
image_10 = canvas.create_image(
    857.0,
    166.0,
    image=image_image_10
)

canvas.create_text(
    700.0,
    150.0,
    anchor="nw",
    text="Groceries:",
    fill="#FFFFFF",
    font=("InriaSerif Regular", 20 * -1)
)

image_image_11 = PhotoImage(
    file=relative_to_assets("image_11.png"))
image_11 = canvas.create_image(
    638.0,
    166.0,
    image=image_image_11
)

canvas.create_text(
    481.0,
    150.0,
    anchor="nw",
    text="Sky Sports:",
    fill="#FFFFFF",
    font=("InriaSerif Regular", 20 * -1)
)

image_image_12 = PhotoImage(
    file=relative_to_assets("image_12.png"))
image_12 = canvas.create_image(
    1141.0,
    166.0,
    image=image_image_12
)

canvas.create_text(
    926.0,
    150.0,
    anchor="nw",
    text="Transportation:",
    fill="#FFFFFF",
    font=("InriaSerif Regular", 20 * -1)
)

canvas.create_rectangle(
    0.0,
    0.0,
    1300.0,
    80.0,
    fill="#F29F58",
    outline="")

image_image_13 = PhotoImage(
    file=relative_to_assets("image_13.png"))
image_13 = canvas.create_image(
    41.0,
    40.0,
    image=image_image_13
)

canvas.create_text(
    80.0,
    29.0,
    anchor="nw",
    text="Future Finance",
    fill="#000000",
    font=("InriaSerif Bold", 20 * -1)
)

image_image_14 = PhotoImage(
    file=relative_to_assets("image_14.png"))
image_14 = canvas.create_image(
    221.0,
    497.0,
    image=image_image_14
)

image_image_15 = PhotoImage(
    file=relative_to_assets("image_15.png"))
image_15 = canvas.create_image(
    845.0,
    497.0,
    image=image_image_15
)

canvas.create_text(
    54.0,
    763.0,
    anchor="nw",
    text="if you skip this purchase, you will have more money next month",
    fill="#FFFFFF",
    font=("InriaSerif Regular", 20 * -1)
)

image_image_16 = PhotoImage(
    file=relative_to_assets("image_16.png"))
image_16 = canvas.create_image(
    947.0,
    116.0,
    image=image_image_16
)

canvas.create_text(
    818.0,
    100.0,
    anchor="nw",
    text="Gas Bill:",
    fill="#FFFFFF",
    font=("InriaSerif Regular", 20 * -1)
)

image_image_17 = PhotoImage(
    file=relative_to_assets("image_17.png"))
image_17 = canvas.create_image(
    1175.0,
    116.0,
    image=image_image_17
)

canvas.create_text(
    1016.0,
    100.0,
    anchor="nw",
    text="Water Bill:",
    fill="#FFFFFF",
    font=("InriaSerif Regular", 20 * -1)
)

df = pd.read_csv('../personal_finance_employees_V1.csv')

df_corrdata = {
    'Employee' : df['Employee'],
    'Bills': df['Electricity Bill (£)'] + df['Gas Bill (£)'] + df['Water Bill (£)'],
    'Entertainment': df['Amazon Prime (£)'] + df['Netflix (£)'] + df['Sky Sports (£)'],
    'Transport': df['Transportation (£)'],
    'Savings': df['Savings for Property (£)']
}
df_corrdata = pd.DataFrame(df_corrdata)


# Filter for Employee_1
subframe = df_corrdata[df_corrdata['Employee'] == 'Employee_2']

subframe = subframe.drop(columns=['Employee'])

# Prepare data for the pie chart
expenses = subframe.iloc[0]  # Select the first (and only) row of the subframe


labels = expenses.index
sizes = expenses.values

# Create the pie chart
fig_1 = plt.figure(figsize=(3.1, 3.1), facecolor="#441752")
ax_1 = plt.pie(sizes, labels=labels, startangle=140, autopct='%1.1f%%')

canvas = FigureCanvasTkAgg(figure=fig_1, master=window)
canvas.draw()
canvas.get_tk_widget().place(x=50, y=350)


window.resizable(False, False)
window.mainloop()
