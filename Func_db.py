"""
@functions: parameter calculation
@author: Zhen ZHAO
@date: June 20, 2018
"""
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

import os
import sqlite3
import pickle
import argparse

_design_idea = """
In this functions, we provide an database editor/viewer.
All other functions retrieve the station and source data from this database.
However, for efficiency, we save the whole data as a pickle file, 
which can save the info-selection time. Subsequently, all the other functions
are actually based on the pickle file. Thus,

- if you use our provided DB Editor, every would be fine and the pickle file will be updated automatically. Run it through "python Func_cal.py -g"

- if you use some other DB editor to modify the db file directly, please remember to update the pickle accordingly. Run it through "python Func_cal.py -p"

Get help by "python Func_cal.py -h"
"""


class DbEditor(object):
    def __init__(self, window=None, db_file='', pkl_file=''):
        self.db_name = db_file
        self.pkl_name = pkl_file
        self.master = window
        self.myDb = DbModel(self.db_name, self.pkl_name)

        self.test_num = self.master.register(test)

        self.delete_item_src = tk.StringVar('')
        self.delete_item_sat = tk.StringVar('')
        self.delete_item_telem = tk.StringVar('')
        self.delete_item_vlbi = tk.StringVar('')

        self._gui_int()
        center_window(self.master, 700, 400)

    def _gui_int(self):
        # one tab for each table, 4 in total
        tab_control = ttk.Notebook(self.master)
        tab_control.pack(expand=1, fill="both")

        # # # # # # # # # # #
        # 1. table_src
        # # # # # # # # # # #
        tab_src = tk.Frame(tab_control)
        tab_control.add(tab_src, text='Source Table')

        # 1.1 table_src input
        input_frm_src = tk.Frame(tab_src)  # , bg='red', height=20
        input_frm_src.pack(side=tk.TOP, fill=tk.X, expand=1,  anchor='nw', padx=20, pady=5)

        self.var_src_name = tk.StringVar()
        tk.Label(input_frm_src, text='Source name [string]:').grid(row=0, column=1, sticky=tk.E)
        tk.Entry(input_frm_src, bg="#282B2B", fg="white", width=15, textvariable=self.var_src_name,
                 validate="key", validatecommand=(self.test_num, '%P')).grid(row=0, column=2, sticky=tk.W)

        self.var_src_ra = tk.StringVar()
        tk.Label(input_frm_src, text='Source Ra [hms]:').grid(row=0, column=3, sticky=tk.E)
        tk.Entry(input_frm_src, bg="#282B2B", fg="white", width=15, textvariable=self.var_src_ra).grid(row=0, column=4, sticky=tk.W)

        self.var_src_dec = tk.StringVar()
        tk.Label(input_frm_src, text='Source Dec [hms]:').grid(row=1, column=1, sticky=tk.E)
        tk.Entry(input_frm_src, bg="#282B2B", fg="white", width=15, textvariable=self.var_src_dec).grid(row=1, column=2, sticky=tk.W)

        tk.Button(input_frm_src, text="Import Data", command=self.load_into_table_src).grid(row=2, column=1, sticky=tk.E)
        tk.Button(input_frm_src, text="Delete", command=self.delete_table_src).grid(row=2, column=3, sticky=tk.E)
        tk.Button(input_frm_src, text="Reset", command=self.reset_table_src).grid(row=2, column=4, sticky=tk.E)
        tk.Button(input_frm_src, text="Insert", command=self.insert_table_src).grid(row=2, column=5, sticky=tk.E)
        for i in range(6):
            input_frm_src.grid_columnconfigure(i, weight=1)

        # 1.2 table_src view
        ttk.Separator(tab_src).pack(side=tk.TOP, fill=tk.X, expand=1)
        view_frm_src = tk.Frame(tab_src, bg='blue', height=20)  # , bg='blue', height=20
        view_frm_src.pack(side=tk.TOP, fill=tk.X, expand=1, anchor='n', padx=20, pady=5)

        src_vbar = ttk.Scrollbar(view_frm_src, orient=tk.VERTICAL)
        src_hbar = ttk.Scrollbar(view_frm_src, orient=tk.HORIZONTAL)
        self.tree_view_src = ttk.Treeview(view_frm_src,
                                          columns=('c1', 'c2', 'c3'),
                                          show="headings",
                                          yscrollcommand=src_vbar.set,
                                          xscrollcommand=src_hbar.set)
        self.tree_view_src.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        src_vbar.config(command=self.tree_view_src.yview)
        src_hbar.config(command=self.tree_view_src.xview)

        self.tree_view_src.column('c1', width=100, anchor='center')
        self.tree_view_src.column('c2', width=100, anchor='center')
        self.tree_view_src.column('c3', width=100, anchor='center')
        self.tree_view_src.heading('c1', text='src_name')
        self.tree_view_src.heading('c2', text='src_ra')
        self.tree_view_src.heading('c3', text='src_dec')

        # # # # # # # # # # #
        # 2. table_sat
        # # # # # # # # # # #
        tab_sat = ttk.Frame(tab_control)
        tab_control.add(tab_sat, text='Satellite Table')

        # 2.1 table_sat input
        input_frm_sat = tk.Frame(tab_sat)  # , bg='red', height=20
        input_frm_sat.pack(side=tk.TOP, fill=tk.X, expand=1, anchor='nw', padx=20, pady=5)

        self.var_sat_name = tk.StringVar()
        tk.Label(input_frm_sat, text='Satellite name [string]:').grid(row=0, column=1, sticky=tk.E)
        tk.Entry(input_frm_sat, bg="#282B2B", fg="white", width=15, textvariable=self.var_sat_name,
                 validate="key", validatecommand=(self.test_num, '%P')).grid(row=0, column=2, sticky=tk.W)

        self.var_sat_apo = tk.StringVar()
        tk.Label(input_frm_sat, text='Satellite apo. [deg]:').grid(row=0, column=3, sticky=tk.E)
        tk.Entry(input_frm_sat, bg="#282B2B", fg="white", width=15, textvariable=self.var_sat_apo,
                 validate="key", validatecommand=(self.test_num, '%P')).grid(row=0, column=4, sticky=tk.W)

        self.var_sat_peri = tk.StringVar()
        tk.Label(input_frm_sat, text='Satellite peri. [deg]:').grid(row=1, column=1, sticky=tk.E)
        tk.Entry(input_frm_sat, bg="#282B2B", fg="white", width=15, textvariable=self.var_sat_peri,
                 validate="key", validatecommand=(self.test_num, '%P')).grid(row=1, column=2, sticky=tk.W)

        self.var_sat_incl = tk.StringVar()
        tk.Label(input_frm_sat, text='Satellite incl. [deg]:').grid(row=1, column=3, sticky=tk.E)
        tk.Entry(input_frm_sat, bg="#282B2B", fg="white", width=15, textvariable=self.var_sat_incl,
                 validate="key", validatecommand=(self.test_num, '%P')).grid(row=1, column=4, sticky=tk.W)

        self.var_sat_o1 = tk.StringVar()
        tk.Label(input_frm_sat, text='Satellite omega [deg]:').grid(row=2, column=1, sticky=tk.E)
        tk.Entry(input_frm_sat, bg="#282B2B", fg="white", width=15, textvariable=self.var_sat_o1,
                 validate="key", validatecommand=(self.test_num, '%P')).grid(row=2, column=2, sticky=tk.W)

        self.var_sat_o2 = tk.StringVar()
        tk.Label(input_frm_sat, text='Satellite Omega [deg]:').grid(row=2, column=3, sticky=tk.E)
        tk.Entry(input_frm_sat, bg="#282B2B", fg="white", width=15, textvariable=self.var_sat_o2,
                 validate="key", validatecommand=(self.test_num, '%P')).grid(row=2, column=4, sticky=tk.W)

        self.var_sat_m0 = tk.StringVar()
        tk.Label(input_frm_sat, text='Satellite M0 [deg]:').grid(row=3, column=1, sticky=tk.E)
        tk.Entry(input_frm_sat, bg="#282B2B", fg="white", width=15, textvariable=self.var_sat_m0,
                 validate="key", validatecommand=(self.test_num, '%P')).grid(row=3, column=2, sticky=tk.W)

        self.var_sat_t = tk.StringVar()
        tk.Label(input_frm_sat, text='Satellite T0 [string]:').grid(row=3, column=3, sticky=tk.E)
        tk.Entry(input_frm_sat, bg="#282B2B", fg="white", width=15, textvariable=self.var_sat_t,
                 validate="key", validatecommand=(self.test_num, '%P')).grid(row=3, column=4, sticky=tk.W)

        tk.Button(input_frm_sat, text="Import Data", command=self.load_into_table_sat).grid(row=4, column=1, sticky=tk.E)
        tk.Button(input_frm_sat, text="Delete", command=self.delete_table_sat).grid(row=4, column=3, sticky=tk.E)
        tk.Button(input_frm_sat, text="Reset", command=self.reset_table_sat).grid(row=4, column=4, sticky=tk.E)
        tk.Button(input_frm_sat, text="Insert", command=self.insert_table_sat).grid(row=4, column=5, sticky=tk.E)
        for i in range(6):
            input_frm_sat.grid_columnconfigure(i, weight=1)

        # 2.2 table_sat view
        ttk.Separator(tab_sat).pack(side=tk.TOP, fill=tk.X, expand=1)
        view_frm_sat = tk.Frame(tab_sat)  # , bg='blue', height=20
        view_frm_sat.pack(side=tk.TOP, fill=tk.X, expand=1, anchor='nw')

        sat_vbar = ttk.Scrollbar(view_frm_sat, orient=tk.VERTICAL)
        sat_hbar = ttk.Scrollbar(view_frm_sat, orient=tk.HORIZONTAL)
        self.tree_view_sat = ttk.Treeview(view_frm_sat,
                                          columns=('c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8'),
                                          show="headings",
                                          yscrollcommand=sat_vbar.set,
                                          xscrollcommand=sat_hbar.set)
        self.tree_view_sat.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        sat_vbar.config(command=self.tree_view_sat.yview)
        sat_hbar.config(command=self.tree_view_sat.xview)

        self.tree_view_sat.column('c1', width=100, anchor='center')
        self.tree_view_sat.column('c2', width=100, anchor='center')
        self.tree_view_sat.column('c3', width=100, anchor='center')
        self.tree_view_sat.column('c4', width=100, anchor='center')
        self.tree_view_sat.column('c5', width=100, anchor='center')
        self.tree_view_sat.column('c6', width=100, anchor='center')
        self.tree_view_sat.column('c7', width=100, anchor='center')
        self.tree_view_sat.column('c8', width=100, anchor='center')

        self.tree_view_sat.heading('c1', text='sat_name')
        self.tree_view_sat.heading('c2', text='sat_apo')
        self.tree_view_sat.heading('c3', text='sat_peri')
        self.tree_view_sat.heading('c4', text='sat_incl')
        self.tree_view_sat.heading('c5', text='sat_omega')
        self.tree_view_sat.heading('c6', text='sat_Omega')
        self.tree_view_sat.heading('c7', text='sat_M0')
        self.tree_view_sat.heading('c8', text='sat_t0')

        # # # # # # # # # # #
        # 3. table_telem    #
        # # # # # # # # # # #
        tab_telem = ttk.Frame(tab_control)
        tab_control.add(tab_telem, text='Telemetry Table')

        # 3.1 table_telem input
        input_frm_telem = tk.Frame(tab_telem)  # , bg='red', height=20
        input_frm_telem.pack(side=tk.TOP, fill=tk.X, expand=1, anchor='nw', padx=20, pady=5)

        self.var_telem_name = tk.StringVar()
        tk.Label(input_frm_telem, text='Telemetry name [string]:').grid(row=0, column=1, sticky=tk.E)
        tk.Entry(input_frm_telem, bg="#282B2B", fg="white", width=15, textvariable=self.var_telem_name,
                 validate="key", validatecommand=(self.test_num, '%P')).grid(row=0, column=2, sticky=tk.W)

        self.var_telem_x = tk.StringVar()
        tk.Label(input_frm_telem, text='Telemetry X [km]:').grid(row=0, column=3, sticky=tk.E)
        tk.Entry(input_frm_telem, bg="#282B2B", fg="white", width=15, textvariable=self.var_telem_x,
                 validate="key", validatecommand=(self.test_num, '%P')).grid(row=0, column=4, sticky=tk.W)

        self.var_telem_y = tk.StringVar()
        tk.Label(input_frm_telem, text='Telemetry Y [km]:').grid(row=1, column=1, sticky=tk.E)
        tk.Entry(input_frm_telem, bg="#282B2B", fg="white", width=15, textvariable=self.var_telem_y,
                 validate="key", validatecommand=(self.test_num, '%P')).grid(row=1, column=2, sticky=tk.W)

        self.var_telem_z = tk.StringVar()
        tk.Label(input_frm_telem, text='Telemetry Z [km]:').grid(row=1, column=3, sticky=tk.E)
        tk.Entry(input_frm_telem, bg="#282B2B", fg="white", width=15, textvariable=self.var_telem_z,
                 validate="key", validatecommand=(self.test_num, '%P')).grid(row=1, column=4, sticky=tk.W)

        self.var_telem_el = tk.StringVar()
        tk.Label(input_frm_telem, text='Telemetry El [deg]:').grid(row=2, column=1, sticky=tk.E)
        tk.Entry(input_frm_telem, bg="#282B2B", fg="white", width=15, textvariable=self.var_telem_el,
                 validate="key", validatecommand=(self.test_num, '%P')).grid(row=2, column=2, sticky=tk.W)

        tk.Button(input_frm_telem, text="Import Data", command=self.load_into_table_telem).grid(row=3, column=1, sticky=tk.E)
        tk.Button(input_frm_telem, text="Delete", command=self.delete_table_telem).grid(row=3, column=3, sticky=tk.E)
        tk.Button(input_frm_telem, text="Reset", command=self.reset_table_telem).grid(row=3, column=4, sticky=tk.E)
        tk.Button(input_frm_telem, text="Insert", command=self.insert_table_telem).grid(row=3, column=5, sticky=tk.E)
        for i in range(6):
            input_frm_telem.grid_columnconfigure(i, weight=1)

        # 3.2 table_telem view
        ttk.Separator(tab_telem).pack(side=tk.TOP, fill=tk.X, expand=1)
        view_frm_telem = tk.Frame(tab_telem, bg='blue', height=20)  # , bg='blue', height=20
        view_frm_telem.pack(side=tk.TOP, fill=tk.X, expand=1, anchor='nw')

        telem_vbar = ttk.Scrollbar(view_frm_telem, orient=tk.VERTICAL)
        telem_hbar = ttk.Scrollbar(view_frm_telem, orient=tk.HORIZONTAL)
        self.tree_view_telem = ttk.Treeview(view_frm_telem,
                                            columns=('c1', 'c2', 'c3', 'c4', 'c5'),
                                            show="headings",
                                            yscrollcommand=telem_vbar.set,
                                            xscrollcommand=telem_hbar.set)
        self.tree_view_telem.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        telem_vbar.config(command=self.tree_view_telem.yview)
        telem_hbar.config(command=self.tree_view_telem.xview)

        self.tree_view_telem.column('c1', width=100, anchor='center')
        self.tree_view_telem.column('c2', width=100, anchor='center')
        self.tree_view_telem.column('c3', width=100, anchor='center')
        self.tree_view_telem.column('c4', width=100, anchor='center')
        self.tree_view_telem.column('c5', width=100, anchor='center')

        self.tree_view_telem.heading('c1', text='telem_name')
        self.tree_view_telem.heading('c2', text='telem_x')
        self.tree_view_telem.heading('c3', text='telem_y')
        self.tree_view_telem.heading('c4', text='telem_z')
        self.tree_view_telem.heading('c5', text='telem_el')

        # # # # # # # # # # #
        # 4. table_vlbi
        # # # # # # # # # # #
        tab_vlbi = ttk.Frame(tab_control)
        tab_control.add(tab_vlbi, text='VLBI_Stat Table')

        # 4.1 table_vlbi input
        input_frm_vlbi = tk.Frame(tab_vlbi)  # , bg='red', height=20
        input_frm_vlbi.pack(side=tk.TOP, fill=tk.X, expand=1, anchor='nw', padx=20, pady=5)

        self.var_vlbi_name = tk.StringVar()
        tk.Label(input_frm_vlbi, text='VLBI_Stat name [string]:').grid(row=0, column=1, sticky=tk.E)
        tk.Entry(input_frm_vlbi, bg="#282B2B", fg="white", width=15, textvariable=self.var_vlbi_name,
                 validate="key", validatecommand=(self.test_num, '%P')).grid(row=0, column=2, sticky=tk.W)

        self.var_vlbi_x = tk.StringVar()
        tk.Label(input_frm_vlbi, text='VLBI_Stat X [km]:').grid(row=0, column=3, sticky=tk.E)
        tk.Entry(input_frm_vlbi, bg="#282B2B", fg="white", width=15, textvariable=self.var_vlbi_x,
                 validate="key", validatecommand=(self.test_num, '%P')).grid(row=0, column=4, sticky=tk.W)

        self.var_vlbi_y = tk.StringVar()
        tk.Label(input_frm_vlbi, text='VLBI_Stat Y [km]:').grid(row=1, column=1, sticky=tk.E)
        tk.Entry(input_frm_vlbi, bg="#282B2B", fg="white", width=15, textvariable=self.var_vlbi_y,
                 validate="key", validatecommand=(self.test_num, '%P')).grid(row=1, column=2, sticky=tk.W)

        self.var_vlbi_z = tk.StringVar()
        tk.Label(input_frm_vlbi, text='VLBI_Stat Z [km]:').grid(row=1, column=3, sticky=tk.E)
        tk.Entry(input_frm_vlbi, bg="#282B2B", fg="white", width=15, textvariable=self.var_vlbi_z,
                 validate="key", validatecommand=(self.test_num, '%P')).grid(row=1, column=4, sticky=tk.W)

        self.var_vlbi_el = tk.StringVar()
        tk.Label(input_frm_vlbi, text='VLBI_Stat El [deg]:').grid(row=2, column=1, sticky=tk.E)
        tk.Entry(input_frm_vlbi, bg="#282B2B", fg="white", width=15, textvariable=self.var_vlbi_el,
                 validate="key", validatecommand=(self.test_num, '%P')).grid(row=2, column=2, sticky=tk.W)

        self.var_vlbi_type = tk.StringVar()
        tk.Label(input_frm_vlbi, text='VLBI_Stat Type [0-4]:').grid(row=2, column=3, sticky=tk.E)
        tk.Entry(input_frm_vlbi, bg="#282B2B", fg="white", width=15, textvariable=self.var_vlbi_type,
                 validate="key", validatecommand=(self.test_num, '%P')).grid(row=2, column=4, sticky=tk.W)

        tk.Button(input_frm_vlbi, text="Import Data", command=self.load_into_table_vlbi).grid(row=3, column=1, sticky=tk.E)
        tk.Button(input_frm_vlbi, text="Delete", command=self.delete_table_vlbi).grid(row=3, column=3, sticky=tk.E)
        tk.Button(input_frm_vlbi, text="Reset", command=self.reset_table_vlbi).grid(row=3, column=4, sticky=tk.E)
        tk.Button(input_frm_vlbi, text="Insert", command=self.insert_table_vlbi).grid(row=3, column=5, sticky=tk.E)
        for i in range(6):
            input_frm_vlbi.grid_columnconfigure(i, weight=1)
        # 4.2 table_vlbi view
        ttk.Separator(tab_vlbi).pack(side=tk.TOP, fill=tk.X, expand=1)
        view_frm_vlbi = tk.Frame(tab_vlbi, bg='blue', height=20)  # , bg='blue', height=20
        view_frm_vlbi.pack(side=tk.TOP, fill=tk.X, expand=1, anchor='nw')

        vlbi_vbar = ttk.Scrollbar(view_frm_vlbi, orient=tk.VERTICAL)
        vlbi_hbar = ttk.Scrollbar(view_frm_vlbi, orient=tk.HORIZONTAL)
        self.tree_view_vlbi = ttk.Treeview(view_frm_vlbi,
                                           columns=('c1', 'c2', 'c3', 'c4', 'c5', 'c6'),
                                           show="headings",
                                           yscrollcommand=vlbi_vbar.set,
                                           xscrollcommand=vlbi_hbar.set)
        self.tree_view_vlbi.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        vlbi_vbar.config(command=self.tree_view_vlbi.yview)
        vlbi_hbar.config(command=self.tree_view_vlbi.xview)

        self.tree_view_vlbi.column('c1', width=100, anchor='center')
        self.tree_view_vlbi.column('c2', width=100, anchor='center')
        self.tree_view_vlbi.column('c3', width=100, anchor='center')
        self.tree_view_vlbi.column('c4', width=100, anchor='center')
        self.tree_view_vlbi.column('c5', width=100, anchor='center')
        self.tree_view_vlbi.column('c6', width=100, anchor='center')

        self.tree_view_vlbi.heading('c1', text='vlbi_name')
        self.tree_view_vlbi.heading('c2', text='vlbi_x')
        self.tree_view_vlbi.heading('c3', text='vlbi_y')
        self.tree_view_vlbi.heading('c4', text='vlbi_z')
        self.tree_view_vlbi.heading('c5', text='vlbi_el')
        self.tree_view_vlbi.heading('c6', text='vlbi_type')

        # 5. update all data (using multiprocessing)
        self._update_table_src()
        self._update_table_sat()
        self._update_table_telem()
        self._update_table_vlbi()

        # 6. bind tree view with event
        self.tree_view_src.bind('<ButtonRelease-1>', self._tree_view_src_click)
        self.tree_view_sat.bind('<ButtonRelease-1>', self._tree_view_sat_click)
        self.tree_view_telem.bind('<ButtonRelease-1>', self._tree_view_telem_click)
        self.tree_view_vlbi.bind('<ButtonRelease-1>', self._tree_view_vlbi_click)

    # load through txt
    def load_into_table_src(self):
        pass

    def load_into_table_sat(self):
        pass

    def load_into_table_telem(self):
        pass

    def load_into_table_vlbi(self):
        pass

    # get select position
    def _tree_view_src_click(self, event):
        if not self.tree_view_src.selection():
            return
        item = self.tree_view_src.selection()[0]
        self.delete_item_src.set(self.tree_view_src.item(item, 'values')[0])

    def _tree_view_sat_click(self, event):
        if not self.tree_view_sat.selection():
            return
        item = self.tree_view_sat.selection()[0]
        self.delete_item_sat.set(self.tree_view_sat.item(item, 'values')[0])

    def _tree_view_telem_click(self, event):
        if not self.tree_view_telem.selection():
            return
        item = self.tree_view_telem.selection()[0]
        self.delete_item_telem.set(self.tree_view_telem.item(item, 'values')[0])

    def _tree_view_vlbi_click(self, event):
        if not self.tree_view_vlbi.selection():
            return
        item = self.tree_view_vlbi.selection()[0]
        self.delete_item_vlbi.set(self.tree_view_vlbi.item(item, 'values')[0])

    # reset
    def reset_table_src(self):
        self.var_src_dec.set('')
        self.var_src_name.set('')
        self.var_src_ra.set('')

    def reset_table_sat(self):
        self.var_sat_apo.set('')
        self.var_sat_incl.set('')
        self.var_sat_name.set('')
        self.var_sat_o1.set('')
        self.var_sat_o2.set('')
        self.var_sat_peri.set('')
        self.var_sat_t.set('')
        self.var_sat_m0.set('')

    def reset_table_telem(self):
        self.var_telem_el.set('')
        self.var_telem_name.set('')
        self.var_telem_x.set('')
        self.var_telem_y.set('')
        self.var_telem_z.set('')

    def reset_table_vlbi(self):
        self.var_vlbi_el.set('')
        self.var_vlbi_name.set('')
        self.var_vlbi_type.set('')
        self.var_vlbi_x.set('')
        self.var_vlbi_y.set('')
        self.var_vlbi_z.set('')

    # insert
    def insert_table_src(self):
        tmp_name = self.var_src_name.get()
        tmp_ra = self.var_src_ra.get()
        tmp_dec = self.var_src_dec.get()
        if tmp_name == '' or tmp_ra == '' or tmp_dec == '':
            # print("bad input")
            return
        else:
            self.myDb.insert_src_data(tmp_name, tmp_ra, tmp_dec)

        self._update_table_src()

    def insert_table_sat(self):
        tmp_apo = self.var_sat_apo.get()
        tmp_incl = self.var_sat_incl.get()
        tmp_name = self.var_sat_name.get()
        tmp_o1 = self.var_sat_o1.get()
        tmp_o2 = self.var_sat_o2.get()
        tmp_peri = self.var_sat_peri.get()
        tmp_m0 = self.var_sat_m0.get()
        tmp_t = self.var_sat_t.get()
        if tmp_apo == '' or tmp_incl == '' or tmp_name == '' or tmp_o1 == '' or tmp_o2 == '' or tmp_peri == '' or tmp_t == '' or tmp_m0 == '':
            return
        else:
            self.myDb.insert_sat_data(tmp_name, tmp_apo, tmp_peri, tmp_incl, tmp_o1, tmp_o2, tmp_m0, tmp_t)

        self._update_table_sat()

    def insert_table_telem(self):
        tmp_el = self.var_telem_el.get()
        tmp_name = self.var_telem_name.get()
        tmp_x = self.var_telem_x.get()
        tmp_y = self.var_telem_y.get()
        tmp_z = self.var_telem_z.get()
        if tmp_el == '' or tmp_name == '' or tmp_x == '' or tmp_y == '' or tmp_z == '':
            return
        else:
            self.myDb.insert_telem_data(tmp_name, tmp_x, tmp_y, tmp_z, tmp_el)

        self._update_table_telem()

    def insert_table_vlbi(self):
        tmp_el = self.var_vlbi_el.get()
        tmp_name = self.var_vlbi_name.get()
        tmp_type = self.var_vlbi_type.get()
        tmp_x = self.var_vlbi_x.get()
        tmp_y = self.var_vlbi_y.get()
        tmp_z = self.var_vlbi_z.get()
        if tmp_el == '' or tmp_name == '' or tmp_type == '' or tmp_x == '' or tmp_y == '' or tmp_z == '':
            return
        else:
            self.myDb.insert_vlbi_data(tmp_name,tmp_x, tmp_y, tmp_z, tmp_el, tmp_type)

        self._update_table_vlbi()

    # delete
    def delete_table_src(self):
        name = self.delete_item_src.get()
        # print(name)
        if name == '':
            # tk.messagebox.showerror(title='Sorry', message='Please a record')
            return
        else:
            self.myDb.delete_src_data(name)
            self.delete_item_src.set('')
            self._update_table_src()

    def delete_table_sat(self):
        name = self.delete_item_sat.get()
        if name == '':
            # tk.messagebox.showerror(title='Sorry', message='Please a record')
            return
        else:
            self.myDb.delete_sat_data(name)
            self.delete_item_sat.set('')
            self._update_table_sat()

    def delete_table_telem(self):
        name = self.delete_item_telem.get()
        if name == '':
            # tk.messagebox.showerror(title='Sorry', message='Please a record')
            return
        else:
            self.myDb.delete_telem_data(name)
            self.delete_item_telem.set('')
            self._update_table_telem()

    def delete_table_vlbi(self):
        name = self.delete_item_vlbi.get()
        if name == '':
            # tk.messagebox.showerror(title='Sorry', message='Please a record')
            return
        else:
            self.myDb.delete_vlbi_data(name)
            self.delete_item_vlbi.set('')
            self._update_table_vlbi()

    # update
    def _update_table_src(self):
        # 1. delete old
        for row in self.tree_view_src.get_children():
            self.tree_view_src.delete(row)
        # 2. read new
        temp = self.myDb.read_src_all()
        # 3. insert new
        for i, item in enumerate(temp):
            self.tree_view_src.insert('', i, values=item[:])

    def _update_table_sat(self):
        # 1. delete old
        for row in self.tree_view_sat.get_children():
            self.tree_view_sat.delete(row)
        # 2. read new
        temp = self.myDb.read_sat_all()
        # 3. insert new
        for i, item in enumerate(temp):
            self.tree_view_sat.insert('', i, values=item[:])

    def _update_table_telem(self):
        # 1. delete old
        for row in self.tree_view_telem.get_children():
            self.tree_view_telem.delete(row)
        # 2. read new
        temp = self.myDb.read_telem_all()
        # 3. insert new
        for i, item in enumerate(temp):
            self.tree_view_telem.insert('', i, values=item[:])

    def _update_table_vlbi(self):
        # 1. delete old
        for row in self.tree_view_vlbi.get_children():
            self.tree_view_vlbi.delete(row)
        # 2. read new
        temp = self.myDb.read_vlbi_all()
        # 3. insert new
        for i, item in enumerate(temp):
            self.tree_view_vlbi.insert('', i, values=item[:])

    # update pkl file
    def update_db_pkl(self):
        self.myDb.write_to_pickle()


class DbModel(object):
    def __init__(self, db_file, pickle_file):
        # 4个tableName: table_src, table_vlbi, table_telem, table_sat
        db_dir = os.path.join(os.getcwd(), 'DATABASE')
        self.db_path = os.path.join(db_dir, db_file)
        self.pkl_path = os.path.join(db_dir, pickle_file)

        # # conn and cur
        # self.conn = None
        # self.is_conn_obtained = False
        # self.cursor = None
        # self.is_cursor_obtained = False

        # 留接口，将4个表中的数据，保存成字典类型，并且写入pickle文件
        self.src_dict = {}
        self.sat_dict = {}
        self.telem_dict = {}
        self.vlbi_dict = {}

        self.vlbi_vlba_dict = {}
        self.vlbi_evn_dict = {}
        self.vlbi_eavn_dict = {}
        self.vlbi_lba_dict = {}
        self.vlbi_other_dict = {}

    def do_insert_delete_sql(self, sql_statement):
        if (sql_statement is None) or (sql_statement == ''):
            return
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute(sql_statement)
            conn.commit()

    def do_select_sql(self, sql_statement):
        if (sql_statement is None) or (sql_statement == ''):
            return None
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute(sql_statement)
            result = cur.fetchall()
            return result

    def create_table_src(self):
        pass

    def create_table_vlbi(self):
        pass

    def create_table_sat(self):
        pass

    def create_table_telem(self):
        pass

    # def _get_conn(self):
    #     if os.path.exists(self.db_path) and os.path.isfile(self.db_path):
    #         self.conn = sqlite3.connect(self.db_path)
    #         self.is_conn_obtained = True
    #
    # def _get_cursor(self):
    #     if not self.is_conn_obtained:
    #         self._get_conn()
    #     if self.conn is not None:
    #         self.cursor = self.conn.cursor()
    #         self.is_cursor_obtained = True
    #
    # def _close_all(self):
    #     '''关闭数据库游标对象和数据库连接对象'''
    #     try:
    #         if self.cursor is not None:
    #             self.cursor.close()
    #     finally:
    #         if self.conn is not None:
    #             self.conn.close()

    # # # # # # # # # # # # #
    # 4个表，对于4种 增加，删除
    # # # # # # # # # # # # #
    def insert_src_data(self, name, ra, dec):
        # 此处应添加 插入检测
        sql = 'INSERT INTO table_src (src_name, src_ra, src_dec) VALUES("' \
              + name + '","' + ra + '","' + dec + '")'
        self.do_insert_delete_sql(sql)

    def delete_src_data(self, name):
        sql = 'DELETE FROM table_src WHERE src_name="' + name + '"'
        # print(sql)
        self.do_insert_delete_sql(sql)

    def insert_vlbi_data(self, name, x, y, z, el, type):
        # 此处应添加 插入检测
        sql = 'INSERT INTO table_vlbi (vlbi_name, vlbi_x, vlbi_y, vlbi_z, vlbi_el, vlbi_type) VALUES("' \
              + name + '",' + x + "," + y + ',' + z + ',' + el + ',' + type + ')'
        self.do_insert_delete_sql(sql)

    def delete_vlbi_data(self, name):
        sql = 'DELETE FROM table_vlbi WHERE vlbi_name="' + name + '"'
        self.do_insert_delete_sql(sql)

    def insert_sat_data(self, name, apo, peri, incl, omega, Omega, sat_m0, t):
        # 此处应添加 插入检测
        sql = 'INSERT INTO table_sat (sat_name, sat_apo, sat_peri, sat_incl, sat_o1, sat_o2, sat_m0, sat_t) VALUES("' \
              + name + '",' + apo + "," + peri + ',' + incl + ',' + omega + ',' + Omega + ',' + sat_m0 + ',"' + t + '")'
        self.do_insert_delete_sql(sql)

    def delete_sat_data(self, name):
        sql = 'DELETE FROM table_sat WHERE sat_name="' + name + '"'
        self.do_insert_delete_sql(sql)

    def insert_telem_data(self, name, x, y, z, el):
        # 此处应添加 插入检测
        sql = 'INSERT INTO table_telem (telem_name, telem_x, telem_y, telem_z, telem_el) VALUES("' \
              + name + '",' + x + "," + y + ',' + z + ',' + el + ')'
        self.do_insert_delete_sql(sql)

    def delete_telem_data(self, name):
        sql = 'DELETE FROM table_telem WHERE telem_name="' + name + '"'
        self.do_insert_delete_sql(sql)

    # # # # # # # # # # # # # # # # #
    # 查询4个表的内容，并保存到字典变量中
    # # # # # # # # # # # # # # # # #
    def read_src_all(self):
        sql = 'SELECT * FROM table_src'
        result = self.do_select_sql(sql)
        return result

    def _read_src_to_dict(self):
        result = self.read_src_all()
        if result is None:
            print('\tDatabase has no record at this time.')
        else:
            for record in result:
                if len(record) != 0:
                    self.src_dict[record[0]] = record
        # print(self.src_dict)

    def read_sat_all(self):
        sql = 'SELECT * FROM table_sat'
        result = self.do_select_sql(sql)
        return result

    def _read_sat_to_dict(self):
        result = self.read_sat_all()
        if result is None:
            print('\tDatabase has no record at this time.')
        else:
            for record in result:
                if len(record) != 0:
                    self.sat_dict[record[0]] = record
        # print(self.sat_dict)

    def read_telem_all(self):
        sql = 'SELECT * FROM table_telem'
        result = self.do_select_sql(sql)
        return result

    def _read_telem_to_dict(self):
        result = self.read_telem_all()
        if result is None:
            print('\tDatabase has no record at this time.')
        else:
            for record in result:
                if len(record) != 0:
                    self.telem_dict[record[0]] = record
        # print(self.telem_dict)

    def read_vlbi_all(self):
        sql = 'SELECT * FROM table_vlbi'
        result = self.do_select_sql(sql)
        return result

    def _read_vlbi_to_dict_all(self):
        result = self.read_vlbi_all()
        if result is None:
            print('\tDatabase has no record at this time.')
        else:
            for record in result:
                if len(record) != 0:
                    self.vlbi_dict[record[0]] = record

    def _read_vlbi_to_dict(self):
        self._read_vlbi_vlba()
        self._read_vlbi_evn()
        self._read_vlbi_eavn()
        self._read_vlbi_lba()
        self._read_vlbi_other()

    def _read_vlbi_vlba(self):
        sql = 'SELECT * FROM table_vlbi WHERE vlbi_type = 0'
        result = self.do_select_sql(sql)
        if result is None:
            print('\tDatabase has no record at this time.')
        else:
            for record in result:
                if len(record) != 0:
                    self.vlbi_vlba_dict[record[0]] = record
        # print(self.vlbi_vlba_dict)

    def _read_vlbi_evn(self):
        sql = 'SELECT * FROM table_vlbi WHERE vlbi_type = 1'
        result = self.do_select_sql(sql)
        if result is None:
            print('\tDatabase has no record at this time.')
        else:
            for record in result:
                if len(record) != 0:
                    self.vlbi_evn_dict[record[0]] = record
        # print(self.vlbi_evn_dict)

    def _read_vlbi_eavn(self):
        sql = 'SELECT * FROM table_vlbi WHERE vlbi_type = 2'
        result = self.do_select_sql(sql)
        if result is None:
            print('\tDatabase has no record at this time.')
        else:
            for record in result:
                if len(record) != 0:
                    self.vlbi_eavn_dict[record[0]] = record
        # print(self.vlbi_eavn_dict)

    def _read_vlbi_lba(self):
        sql = 'SELECT * FROM table_vlbi WHERE vlbi_type = 3'
        result = self.do_select_sql(sql)
        if result is None:
            print('\tDatabase has no record at this time.')
        else:
            for record in result:
                if len(record) != 0:
                    self.vlbi_lba_dict[record[0]] = record
        # print(self.vlbi_lba_dict)

    def _read_vlbi_other(self):
        sql = 'SELECT * FROM table_vlbi WHERE vlbi_type = 4'
        result = self.do_select_sql(sql)
        if result is None:
            print('\tDatabase has no record at this time.')
        else:
            for record in result:
                if len(record) != 0:
                    self.vlbi_other_dict[record[0]] = record
        # print(self.vlbi_other_dict)

    # # # # # # # # # # # # # # # # #
    # 将字典变量dump到pickle文件中
    # # # # # # # # # # # # # # # # #
    # dump 和 load 的顺序保持一致
    def write_to_pickle(self):
        self._read_src_to_dict()
        self._read_vlbi_to_dict()
        self._read_sat_to_dict()
        self._read_telem_to_dict()
        # add new with all vlbi
        self._read_vlbi_to_dict_all()
        if os.path.exists(self.pkl_path):
            os.remove(self.pkl_path)
        with open(self.pkl_path, 'wb') as fw:
            pickle.dump(self.src_dict, fw)
            pickle.dump(self.sat_dict, fw)
            pickle.dump(self.telem_dict, fw)
            pickle.dump(self.vlbi_vlba_dict, fw)
            pickle.dump(self.vlbi_evn_dict, fw)
            pickle.dump(self.vlbi_eavn_dict, fw)
            pickle.dump(self.vlbi_lba_dict, fw)
            pickle.dump(self.vlbi_other_dict, fw)
            pickle.dump(self.vlbi_dict, fw)

    def test_read_data(self):
        self.write_to_pickle()
        with open(self.pkl_path, 'rb') as fr:
            dict_src = pickle.load(fr)
            dict_sat = pickle.load(fr)
            dict_telem = pickle.load(fr)
            dict_vlba = pickle.load(fr)
            print(dict_sat)
            # print(dict_src)
            # print(dict_vlba)


def test(content):
    return True
    # if content.isdigit() or is_float(content):
    # if content.isdigit() or '.' in content:
    #     return True
    # else:
    #     return False


def is_float(s):
    s = str(s)
    if s.count('.') == 1:
        new_s = s.split('.')
        left_num = new_s[0]
        right_num = new_s[1]
        if right_num.isdigit():
            if left_num.isdigit():
                return True
            elif left_num.count('-') == 1 and left_num.startswith('-'):
                tmp_num = left_num.split('-')[-1]
                if tmp_num.isdigit():
                    return True

    return False


def center_window(root, width, height):
    screenwidth = root.winfo_screenwidth()
    screenheight = root.winfo_screenheight()
    position_x = max((screenwidth - width) / 2 - width, 0)
    position_y = max((screenheight - height) / 2 - 80, 0)
    size = '%dx%d+%d+%d' % (width, height, position_x, position_y)
    # print(size)
    root.geometry(size)


def update_pickle():
    db_file = 'database.db'
    pkl_file = 'database.pkl'

    # test DbModel
    myDb = DbModel(db_file, pkl_file)
    # myDb.test_read_data()
    myDb.write_to_pickle()


def test_dbGUI():
    db_file = 'database.db'
    pkl_file = 'database.pkl'

    # test DbEditor
    my_window = tk.Tk()
    # my_window.resizable(False, False)
    my_window.title("Database Editor")
    DbEditor(my_window, db_file, pkl_file)
    my_window.mainloop()


def parse_args():
    parser = argparse.ArgumentParser(description="Run the Database Viewer and update the pickle file")
    group = parser.add_mutually_exclusive_group()

    group.add_argument('-g',
                       '--show_gui',
                       action="store_true",
                       help='Choose to use my developed db editor to view/update data')

    group.add_argument('-p',
                       '--update_pkl',
                       action="store_true",
                       help='Remember to update the pickle file if you use other database editor')

    parser.add_argument('-i',
                        '--show_info',
                        action="store_true",
                        help='Show info about our design and why we use pickle', )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # args.show_gui = True
    if args.show_info:
        print(_design_idea)

    if args.update_pkl:
        update_pickle()

    if args.show_gui:
        test_dbGUI()
