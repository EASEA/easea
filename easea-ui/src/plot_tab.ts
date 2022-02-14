/**
 * @author Clément Ligneul <clement.ligneul@etu.unistra.fr>
 */

import { AcceptMode, AlignmentFlag, QBoxLayout, QFileDialog, QGridLayout, QLabel, QPixmap, QPushButton, QScrollArea, QWidget, WidgetEventTypes } from "@nodegui/nodegui";
import fs, { existsSync } from 'fs';
import { spawn, ChildProcess } from "child_process";
import { running_plot } from "./index";
import { run_obj } from './index'
import os from 'os'
import { Win_alert } from "./win_alert";
import { Update_graph_win } from "./update_graph_win";


const SCRIPT_2D_PATH = 'src/plot.py';
const SCRIPT_3D_PATH = 'src/3D_plot.py';

export class Plot_result {
    widget: QWidget;
    layout: QGridLayout;
    image_path: string;
    image: QPixmap;
    image_label: QLabel;
    btn_widget: QWidget;
    save_static_btn: QPushButton;
    save_interactive_btn: QPushButton;
    update_graph: QPushButton;
    scroll_image: QScrollArea;
    update_label: QLabel;
    graph_option: Update_graph_win;

    constructor() {
        this.widget = new QWidget();
        this.layout = new QGridLayout();
        this.image_path = '';
        this.image = new QPixmap();
        this.image_label = new QLabel();
        this.save_static_btn = new QPushButton();
        this.save_interactive_btn = new QPushButton();
        this.update_graph = new QPushButton();
        this.scroll_image = new QScrollArea();
        this.btn_widget = new QWidget();
        this.update_label = new QLabel();
        this.graph_option = new Update_graph_win();
    }

    update_plot(path: string, plot_size: number, type: string, csv_path: string, title: string, x_name: string, y_name: string, z_name: string, color?: string) {
        var run: ChildProcess;

        var nb_gen = run_obj.option_obj.nb_gen
        var plot_path = '';

        if (type === '2D') {
            if (existsSync(SCRIPT_2D_PATH)) {
                plot_path = 'src/plot.py';
            } else if (existsSync('plot.py')) {
                plot_path = 'plot.py';
            } else {
                this.image_label.setFixedSize(200, 30);
                this.image_label.setText('Error : plot script not found');
                console.log('Error : plot script not found');
            }
        } else if (type === '3D') {
            if (existsSync(SCRIPT_3D_PATH)) {
                plot_path = 'src/3D_plot.py';
            } else if (existsSync('3D_plot.py')) {
                plot_path = '3D_plot.py';
            } else {
                this.image_label.setFixedSize(200, 30);
                this.image_label.setText('Error : plot script not found');
                console.log('Error : plot script not found');
            }
        }

        if (isNaN(nb_gen)) {
            var val = this.get_generations(run_obj.ez_file_address);
            console.log('\n\n--------------- Plotting --------------- \n' + 'nb gen detected in file : ' + val);
            if (val === -1) {
                nb_gen = 30;
            } else {
                nb_gen = val;
                run_obj.total_generations = val;
            }
        } else {
            run_obj.total_generations = nb_gen;
        }

        console.log('updating plot...');
        this.btn_widget.setEnabled(false);

        if (type === '2D') {
            if(color ) {
                run = spawn('python3', [plot_path, nb_gen.toString(), plot_size.toString(), title, x_name, y_name, color], { timeout: 20000 });
            } else {
                run = spawn('python3', [plot_path, nb_gen.toString(), plot_size.toString(), title], { timeout: 20000 });
            }
        } else if (type === '3D') {
            run = spawn('python3', [plot_path, nb_gen.toString(), plot_size.toString(), csv_path, title, x_name, y_name, z_name], { timeout: 20000 });
        } else {
            return;
        }

        running_plot.push(run);

        run.stdout?.on('data', (data) => {
            console.log('Plot script : ');
            console.log(data.toString());
        });

        run.stderr?.on('data', (data) => {
            console.log('\nError (Python) in plot script : ');
            console.log(data.toString());
        });

        run.on('exit', (code, signal) => {
            const index = running_plot.indexOf(run, 0);
            if (index > -1)
                running_plot.length === 1 ? running_plot.pop() : running_plot.splice(index, 1);

            if (code === 0) {
                this.image_path = path;
                if (this.image.load(path)) {
                    this.image_label.setPixmap(this.image.scaled(800,600));
                    this.image_label.setFixedSize(900, 600);
                    this.btn_widget.setEnabled(true);
                } else {
                    this.image_label.setFixedSize(210, 30);
                    this.image_label.setText('Error : graph not found');
                    console.log('Error : graph not found');
                }
            } else {
                this.image_label.setFixedSize(210, 30);
                this.image_label.setText('This graph cannot be displayed');
            }

            run_obj.progress_bar.setValue(run_obj.progress_bar.value() + (100 / (2 * run_obj.batch_size)));

            if (running_plot.length === 0 && signal === null) {
                run_obj.running_label.setText('Finished');
                run_obj.running_animation_movie.stop();
                run_obj.running_animation_label.hide();
                run_obj.progress_bar.setValue(100);
                run_obj.enable_buttons(true);
            }

            if (signal?.toString() === 'SIGTERM') {
                run_obj.enable_buttons(true);
                run_obj.running_label.setText('Interrupted');
                run_obj.running_animation_movie.stop();
                run_obj.running_animation_label.hide();
                run_obj.running_widget.show();
            }

            this.btn_widget.setEnabled(true);
            this.update_label.hide();
        });
    }

    // read the number of generations in the ez file
    get_generations(filename: string): number {
        var text_file = fs.readFileSync(filename, 'utf-8');
        var lines = text_file.split('\n');

        for (var i = 0; i < lines.length; i++) {
            // remove all spaces at the beginning
            while (lines[i][0] === ' ' || lines[i][0] === '\t')
                lines[i] = lines[i].substring(1, lines[i].length);

            var col = lines[i].split(' ');

            if (col[0].toLowerCase() === 'number' && col[1].toLowerCase() === 'of') {
                if (col[2].toLowerCase() === 'generations:') {
                    return Number(col[3]);
                } else if (col[2].toLowerCase() === 'generations') {
                    return Number(col[4]);
                }
            }
        }
        return -1;
    }

    generate() {
        this.widget.setLayout(this.layout);

        // save buttons
        const btn_layout = new QBoxLayout(0);
        this.btn_widget.setLayout(btn_layout);

        this.save_static_btn.setText('Save static plot');
        this.save_static_btn.setFixedSize(130, 30);
        this.save_static_btn.addEventListener('clicked', () => {
            const fileDialog = new QFileDialog();
            fileDialog.setNameFilter('*.svg');
            fileDialog.setAcceptMode(AcceptMode.AcceptSave);

            if (fileDialog.exec()) {
                var path_file = fileDialog.selectedFiles().toString();
                if (!path_file.endsWith('.svg'))
                    path_file += '.svg';
                try {
                    fs.copyFileSync('/tmp/plotting/fig.svg', path_file);
                } catch (e) {
                    if (e) {
                        new Win_alert(e + "", 'Save Static Plot');
                        return;
                    }
                }
                new Win_alert('Plot Saved', 'Save Static Plot');
            }
        });

        this.save_interactive_btn.setText('Save interactive plot')
        this.save_interactive_btn.setFixedSize(150, 30);
        this.save_interactive_btn.addEventListener('clicked', () => {
            const fileDialog = new QFileDialog();
            fileDialog.setNameFilter('*.html');
            fileDialog.setAcceptMode(AcceptMode.AcceptSave);

            if (fileDialog.exec()) {
                var path_file = fileDialog.selectedFiles().toString();
                try {
                    if (!path_file.endsWith('.html'))
                        path_file += '.html';

                    fs.copyFileSync('/tmp/plotting/fig.html', path_file);
                } catch (e) {
                    if (e)
                        new Win_alert(e + "", 'Save Interactive Plot');
                    return;
                }
                new Win_alert('Plot Saved', 'Save Interactive Plot');
            }
        });

        // updating label
        this.update_label.setText('updating ...');
        this.update_label.setAlignment(AlignmentFlag.AlignCenter);
        this.update_label.hide();

        // update graph button
        this.update_graph.setFixedSize(150, 30);
        this.update_graph.setText('Update Plot');
        this.update_graph.addEventListener('clicked', () => {
            this.graph_option.execution();
        });

        btn_layout.addWidget(this.save_static_btn);
        btn_layout.addWidget(this.update_graph);
        btn_layout.addWidget(this.save_interactive_btn);
        this.btn_widget.setEnabled(false);

        // static graph
        this.image_label.setPixmap(this.image);
        this.image_label.setFixedSize(210, 30);
        this.image_label.setText('No graph to display');
        this.image_label.setAlignment(AlignmentFlag.AlignCenter);

        this.image_label.addEventListener(WidgetEventTypes.MouseButtonDblClick, () => {
            if (os.type() === 'Linux') {
                spawn('sensible-browser', ['/tmp/plotting/fig.html']);
            } else if (os.type() === 'Darwin') {
                spawn('open', ['/tmp/plotting/fig.html']);
            }
        })

        this.scroll_image.setWidget(this.image_label);
        this.scroll_image.setAlignment(AlignmentFlag.AlignCenter);

        this.layout.addWidget(this.scroll_image, 0, 0);
        this.layout.addWidget(this.update_label, 1, 0);
        this.layout.addWidget(this.btn_widget, 2, 0);

        return this.layout;
    }
}