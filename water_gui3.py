from sympy import *
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
import io
import sys
import matplotlib.lines as mlines

class ConsoleOutput(io.StringIO):  # create console output inside of tkinter window
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget

    def write(self, message):
        self.text_widget.insert(tk.END, message)
        self.text_widget.see(tk.END)


class rhs_linf:  # linear operator class to make linear rhs with arbitrary constant.
    def __init__(self, c):
        self.c = c

    def __call__(self, u, v):
        return self.c * u


def Euler_solver(f, x_0, y_0, x_end, divs):  # Euler solver
    print("Starting Euler solver...")
    x = np.linspace(x_0, x_end, divs)
    y = np.zeros(divs)
    y[0] = y_0
    for k in range(divs - 1):
        y[k + 1] = f(x[k], y[k]) * (x[k + 1] - x[k]) + y[k]
        print(f"Step {k}, y[{k + 1}] = {y[k + 1]}, x[{k + 1}] = {x[k + 1]}")
    return x, y


def RK2_solver(f, x_0, y_0, x_end, divs):  # Runge-Kutta II order solver
    print("Starting Runge-Kutta II order solver...")
    x = np.linspace(x_0, x_end, divs)
    y = np.zeros(divs)
    y[0] = y_0
    for k in range(divs - 1):
        dx = x[k + 1] - x[k]
        k1 = f(x[k], y[k])
        k2 = f(x[k] + dx * 0.5, y[k] + (dx * 0.5) * k1)
        y[k + 1] = y[k] + (dx / 2.0) * (k1 + k2)
        print(f"Step {k}, y[{k + 1}] = {y[k + 1]}, x[{k + 1}] = {x[k + 1]}")
    return x, y


def RK4_solver(f, x_0, y_0, x_end, divs):  # Runge-Kutta IV order solver
    print("Starting Runge-Kutta IV order solver...")
    x = np.linspace(x_0, x_end, divs)
    y = np.zeros(divs)
    y[0] = y_0
    for k in range(divs - 1):
        dx = x[k + 1] - x[k]
        k1 = f(x[k], y[k])
        k2 = f(x[k] + dx * 0.5, y[k] + (dx * 0.5) * k1)
        k3 = f(x[k] + dx * 0.5, y[k] + (dx * 0.5) * k2)
        k4 = f(x[k] + dx, y[k] + dx * k3)
        y[k + 1] = y[k] + (dx / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        print(f"Step {k}, y[{k + 1}] = {y[k + 1]}, x[{k + 1}] = {x[k + 1]}")
    return x, y


def add_tooltip(widget, text):
    tooltip = ttk.Label(root, text=text, background="white")
    tooltip.place_forget()

    def show(event):
        tooltip.lift()
        tooltip.place(x=event.x_root - root.winfo_x() + 10, y=event.y_root - root.winfo_y() + 10)

    def hide(event):
        tooltip.place_forget()

    widget.bind("<Enter>", show)
    widget.bind("<Leave>", hide)


class WaterSurfaceApp:  # gui class
    def __init__(self, root):
        self.root = root
        self.root.title("Water Surface Simulation")
        self.fig = None
        self.ax = None
        self.canvas_fig = None
        self.legend_drawn = False
        self.setup_ui()

    def setup_ui(self):
        # Default values
        self.DEFAULT_VALUES = {
            'r': "1.0",
            'V_0': "2.0",
            'x_divs': "100",
            'omega': "6.28318",
            'solver': "Runge-Kutta II"
        }

        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(side=tk.LEFT, fill=tk.Y)

        plot_frame = ttk.Frame(self.root, padding="10")
        plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        console_frame = ttk.Frame(self.root, padding="10")
        console_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        # Input parameters r, V, divs, omega
        ttk.Label(control_frame, text="Parameters").grid(row=0, column=0, columnspan=2, pady=5)

        ttk.Label(control_frame, text="Radius (r):").grid(row=1, column=0, sticky=tk.W)
        self.r_entry = ttk.Entry(control_frame)
        self.r_entry.grid(row=1, column=1)
        self.r_entry.insert(0, self.DEFAULT_VALUES['r'])
        add_tooltip(self.r_entry, "Enter cylinder radius")

        ttk.Label(control_frame, text="Volume (V_0):").grid(row=2, column=0, sticky=tk.W)
        self.V0_entry = ttk.Entry(control_frame)
        self.V0_entry.grid(row=2, column=1)
        self.V0_entry.insert(0, self.DEFAULT_VALUES['V_0'])
        add_tooltip(self.V0_entry, "Enter volume")

        ttk.Label(control_frame, text="Divisions (x_divs):").grid(row=3, column=0, sticky=tk.W)
        self.divs_entry = ttk.Entry(control_frame)
        self.divs_entry.grid(row=3, column=1)
        self.divs_entry.insert(0, self.DEFAULT_VALUES['x_divs'])
        add_tooltip(self.divs_entry, "Enter mesh resolution (number of divsions")

        ttk.Label(control_frame, text="Angular velocity (ω):").grid(row=4, column=0, sticky=tk.W)
        self.omega_entry = ttk.Entry(control_frame)
        self.omega_entry.grid(row=4, column=1)
        self.omega_entry.insert(0, self.DEFAULT_VALUES['omega'])
        add_tooltip(self.omega_entry, "Enter anular velocity")

        self.g = 9.8

        #solvers
        ttk.Label(control_frame, text="Solver Method:").grid(row=5, column=0, sticky=tk.W, pady=(10, 0))
        self.solver_var = tk.StringVar()
        self.solver_combobox = ttk.Combobox(control_frame, textvariable=self.solver_var,
                                            values=["Euler", "Runge-Kutta II", "Runge-Kutta IV"])
        self.solver_combobox.grid(row=5, column=1, pady=(10, 0))
        self.solver_combobox.set(self.DEFAULT_VALUES['solver'])

        #errors
        self.error_label = ttk.Label(control_frame, text="Maximum Error: ")
        self.error_label.grid(row=6, column=0, columnspan=2, pady=(10, 5))
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=7, column=0, columnspan=2, pady=10)

        calc_button = ttk.Button(button_frame, text="Calculate", command=self.on_calculate)
        calc_button.pack(side=tk.LEFT, padx=5)

        #defaults
        defaults_button = ttk.Button(button_frame, text="Restore Defaults", command=self.restore_defaults)
        defaults_button.pack(side=tk.LEFT, padx=5)

        # Clear
        clear_button = ttk.Button(button_frame, text="Clear Plot", command=self.clear_plot)
        clear_button.pack(side=tk.LEFT, padx=5)

        self.plot_canvas = tk.Canvas(plot_frame)
        self.plot_canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.initialize_plot()

        #console
        self.console_text = tk.Text(console_frame, height=10, wrap=tk.WORD)
        self.console_scroll = ttk.Scrollbar(console_frame, command=self.console_text.yview)
        self.console_text.configure(yscrollcommand=self.console_scroll.set)
        self.console_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.console_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Redirect
        sys.stdout = ConsoleOutput(self.console_text)

    def initialize_plot(self):
        if self.fig is None:
            self.fig = plt.figure(figsize=(6, 4))
            self.ax = self.fig.add_subplot(111)
            self.ax.grid()
            self.ax.axis('equal')
            self.ax.set_title("Water Surface Profile")
            self.ax.set_xlabel("x")
            self.ax.set_ylabel("y")

            # Embed plot in Tkinter window
            self.canvas_fig = FigureCanvasTkAgg(self.fig, master=self.plot_canvas)
            self.canvas_fig.draw()
            self.canvas_fig.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def clear_plot(self):
        if self.ax:
            self.ax.clear()
            self.ax.grid()
            self.ax.axis('equal')
            self.ax.set_title("Water Surface Profile")
            self.ax.set_xlabel("x")
            self.ax.set_ylabel("y")
            self.canvas_fig.draw()
            self.legend_drawn = False
            print("Plot cleared")

    def restore_defaults(self):
        self.r_entry.delete(0, tk.END)
        self.r_entry.insert(0, self.DEFAULT_VALUES['r'])
        self.V0_entry.delete(0, tk.END)
        self.V0_entry.insert(0, self.DEFAULT_VALUES['V_0'])
        self.divs_entry.delete(0, tk.END)
        self.divs_entry.insert(0, self.DEFAULT_VALUES['x_divs'])
        self.omega_entry.delete(0, tk.END)
        self.omega_entry.insert(0, self.DEFAULT_VALUES['omega'])
        self.solver_combobox.set(self.DEFAULT_VALUES['solver'])
        print("Defaults restored")

    def on_calculate(self):
        try:
            r = float(self.r_entry.get())
            V_0 = float(self.V0_entry.get())
            x_divs = int(self.divs_entry.get())
            omega = float(self.omega_entry.get())
            solver_choice = self.solver_var.get()

            if solver_choice == "Euler":
                solver = Euler_solver
                lcolor = "Blue"
            elif solver_choice == "Runge-Kutta II":
                solver = RK2_solver
                lcolor = "Red"
            else:
                solver = RK4_solver
                lcolor = "Black"

            self.calculate(r, V_0, self.g, omega, x_divs, solver, lcolor)
        except ValueError as e:
            self.console_text.insert(tk.END, f"Error: {str(e)}\n")

    def calculate(self, r, V_0, g, omega, x_divs, solver, lcolor):  # main method
        print("Calculating water surface line...")
        x_0 = 0.0
        x_r = r
        analytic_solution = self.analytic(r, V_0, g, x_divs, omega)
        y_subcrit = analytic_solution[0]
        y_overcrit = analytic_solution[1]
        omega_cr = analytic_solution[2]

        rhs = rhs_linf(omega**2 / g)

        if self.fig is None:
            self.initialize_plot()

        if omega < omega_cr:
            print("omega < omega_critical")
            y_0 = V_0 / (np.pi * r**2) - (omega**2 * r**2) / (4 * g)
            y_analytic = y_subcrit
            linecolor = lcolor
        else:
            print("omega >= omega_critical")
            y_0 = (np.sqrt(V_0) * omega) / (np.sqrt(np.pi) * np.sqrt(g)) - (omega**2 * r**2) / (2 * g)
            y_analytic = y_overcrit
            linecolor = lcolor

        x_num, y_num = solver(rhs, x_0, y_0, x_r, x_divs)
        err = abs(y_analytic - y_num)
        max_err = max(err)
        print(f"Maximum error = {max_err}")
        self.error_label.config(text=f"Maximum Error: {max_err:.6f}")

        y_num = np.where(y_num < 0.0, 0.0, y_num)  # remove y < 0
        y_analytic = np.where(y_analytic < 0.0, 0.0, y_analytic)

        label = f"ω={omega:.2f}, V={V_0:.1f}, {self.solver_var.get()}"
        self.ax.plot(np.hstack((-x_num[::-1], x_num)), np.hstack((y_num[::-1], y_num)),
                     color=linecolor, label=f'Numerical ({label})')
        self.ax.plot(np.hstack((-x_num[::-1], x_num)), np.hstack((y_analytic[::-1], y_analytic)),
                     color='green', linestyle='dotted', label=f'Analytical ({label})')
        #create empty lines
        line1 = mlines.Line2D([], [], color='red', label='RK2')
        line2 = mlines.Line2D([], [], color='blue', label='Euler')
        line3 = mlines.Line2D([], [], color='green', linestyle='dotted',  label='Analytical')
        line4 = mlines.Line2D([], [], color='black', label='RK4')

        legend = self.ax.legend(handles=[line1, line2, line3, line4], loc='upper center') #put legend

        self.canvas_fig.draw()

    def analytic(self, r_inp, V_0_inp, g_inp, x_divs, omega_inp):  # analytical solution
        print("Solving analytical...")
        y_subcrit = np.zeros((x_divs))
        y_overcrit = np.zeros((x_divs))
        x_linsp = np.linspace(0.0, r_inp, x_divs)

        x, omega, y0, C1 = symbols("x omega y0 C1")
        g = symbols("g", positive=True)

        rhs = omega**2 * x / g
        sol = integrate(rhs, x) + C1
        eq1 = Eq(sol.subs(x, 0), y0)  # y(0) = C1
        sol_with_y0 = sol.subs(C1, y0)  # y(x) = y0 + integral(rhs)

        print(f"y(x) = {sol_with_y0}")

        V0, R = symbols("V0 R", positive=True)
        vol = 2 * pi * integrate(sol_with_y0 * x, (x, 0, R))
        print(f"V = {vol}")

        eq2 = Eq(vol, V0)
        sol_V0_y0 = solveset(eq2, y0)
        y1 = simplify(tuple(sol_V0_y0)[0])
        sol_V0 = sol_with_y0.subs(y0, y1)  # subcritical case

        print("Solved. y(x) function for subcritic case has following form:")
        print(f"y(x) = {sol_V0}")

        print("Evaluating omega_cr...")
        eq3 = Eq(sol_V0.subs(x, 0), 0)
        sol3 = solveset(eq3, omega)
        omega0 = tuple(sol3)[1]  # omega_cr expression
        print(f"omega_cr = {omega0}")

        omega_cr = omega0.subs({V0: V_0_inp, R: r_inp, omega: omega_inp, g: g_inp})
        print(f"omega_cr = {omega_cr}")

        for i in range(x_divs):
            y_subcrit[i] = sol_V0.subs({x: x_linsp[i], V0: V_0_inp, R: r_inp, omega: omega_inp, g: g_inp})

        print("y_subcrit values from analytical solution:")
        print(y_subcrit)

        eq4 = Eq(sol_with_y0, 0)
        sol4 = solveset(eq4, x)
        x0 = tuple(sol4)[0]  # x_0 : y(x_0) = 0

        vol1 = 2 * pi * integrate(sol_with_y0 * x, (x, x0, R))  # overcritical case volume
        eq5 = Eq(vol1, V0)
        sol5 = solveset(eq5, y0)
        y2 = tuple(sol5)[0]
        sol_with_y0_crit = sol_with_y0.subs(y0, y2)  # overcritical case

        print("Solved. y(x) function for overcritic case has following form:")
        print(f"y(x) = {sol_with_y0_crit}")

        for i in range(x_divs):
            y_overcrit[i] = sol_with_y0_crit.subs({x: x_linsp[i], V0: V_0_inp, R: r_inp, omega: omega_inp, g: g_inp})

        print("y_overcrit values from analytical solution:")
        print(y_overcrit)

        print("Analytic solution finished.")
        return [y_subcrit, y_overcrit, omega_cr]


if __name__ == "__main__":
    root = tk.Tk()
    app = WaterSurfaceApp(root)  # init app with gui
    root.mainloop()
