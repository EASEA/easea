/**
 * @author Clément Ligneul <clement.ligneul@etu.unistra.fr>
 */

import { QTabWidget, QIcon, QMainWindow, QWidget, WidgetEventTypes, QMenu } from '@nodegui/nodegui';
import { spawn, ChildProcess } from 'child_process';
import { Compile } from './compile_tab';
import { general_css } from './style';
import { Run_tab } from './run_tab';
import * as util from './utilities';
import { QMenuBar, QAction, QGridLayout } from "@nodegui/nodegui";
import { Plot_result } from './plot_tab';
import fs from 'fs';
import os from 'os';
import { exit } from 'process';


/** arrays of running child processes **/
export var running_proc: ChildProcess[] = [];

/** arrays of plotting child processes **/
export var running_plot: ChildProcess[] = [];

/** arrays of running islands **/
export var running_islands: ChildProcess[] = []; 

/** main window */
const global_win = new QMainWindow();
global_win.setWindowTitle("EASEA Compiler");

/** main widget */
const centralWidget = new QWidget();

const general_layout = new QGridLayout();

centralWidget.setLayout(general_layout);

// menu
const main_menu = new QMenuBar();
const sub_menu = new QMenu();
sub_menu.setTitle('Help');

const help_action = new QAction();
help_action.setText('Documentation')
help_action.addEventListener('triggered', () => {
    var command = '';
    if (os.type() === 'Linux') {
        command = 'sensible-browser';
    } else if (os.type() === 'Darwin') {
        command = 'open';
    }
    spawn(command, [process.cwd() + '/documentation/doc_en.html'], { detached: true });
});

const EASEA_action = new QAction();
EASEA_action.setText('EASEA website');
EASEA_action.addEventListener('triggered', () => {
    var command = '';
    if (os.type() === 'Linux') {
        command = 'sensible-browser';
    } else if (os.type() === 'Darwin') {
        command = 'open';
    }
    spawn(command, ['http://easea.unistra.fr/index.php/EASEA_platform'], { detached: true });
});

sub_menu.addAction(help_action);
sub_menu.addAction(EASEA_action);

main_menu.addMenu(sub_menu);
global_win.setMenuBar(main_menu);

// objects
const compile = new QWidget();
export const compile_obj = new Compile('', '');
compile.setLayout(compile_obj.generate_compile_tab());

const run = new QWidget();
export const run_obj = new Run_tab('', '');
run.setLayout(run_obj.generate());

const plot = new QWidget();
export const plot_obj = new Plot_result();
plot.setLayout(plot_obj.generate());


export const tab_menu = new QTabWidget();
tab_menu.addTab(compile, new QIcon(), 'Compile');
tab_menu.addTab(run, new QIcon(), 'Run');
tab_menu.addTab(plot, new QIcon(), 'Result Plot')

general_layout.addWidget(tab_menu);

global_win.setCentralWidget(centralWidget);

global_win.setLayout(general_layout);
global_win.setStyleSheet(general_css);

// called at the end 
global_win.addEventListener(WidgetEventTypes.Close, () => {
    util.kill_all(running_proc);
    fs.rmSync('/tmp/plotting', { recursive: true, force: true });
});

// global_win.adjustSize();
// global_win.setFixedSize(global_win.size().width(), global_win.size().height());
// global_win.setFixedSize(1060, 900);

fs.mkdir('/tmp/plotting/', (err) => {
    if (err && err.code !== 'EEXIST') {
        console.log(err.message);
        console.log(err);
        exit(1);
    }
});

global_win.show();

(global as any).global_win = global_win;    // for garbage collector
