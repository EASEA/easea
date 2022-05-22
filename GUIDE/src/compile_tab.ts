/**
 * @author Clément Ligneul <clement.ligneul@etu.unistra.fr>
 */

import { Pseudo_term } from './pseudo_term';
import { Win_alert } from './win_alert';
import { plot_obj, run_obj } from './index';
import * as util from './utilities';
import { FileMode, QCheckBox, QWidget, QLabel, QFileDialog, QGridLayout, QPushButton, QBoxLayout, AlignmentFlag } from '@nodegui/nodegui';
import { cwd, exit } from 'process';
import fs from 'fs';
import * as paths from './paths';


// buttons dimensions
const btn_width = 110;
const btn_height = 27; 

export class Compile {
    ez_file_address: string;
    dir_path: string;
    ready: Number;
    options: string;
    cuda = false;
    gp = false;
    cuda_gp = false;
    nsgaii = false;
    nsgaiii = false;
    asrea = false;
    ibea = false;
    cdas = false;
    memetic = false;
    cmaes = false;
    verbose = false;
    true_line = false;


    constructor(ez_file: string, dir: string) {
        this.ez_file_address = ez_file;
        this.dir_path = dir;
        this.ready = 0;
        this.options = '';
    }

    generate_compile_tab() {
        var global_layout = new QBoxLayout(2);

        // output terminal
        const widget_console = new QWidget();
        const centered_console = new QGridLayout();
        const output_compile = new Pseudo_term('Output :');
        centered_console.addWidget(output_compile.window, 0, 0);
        widget_console.setLayout(centered_console);

        // grid of options
        const option_layout = new QGridLayout();
        const option_widget = new QWidget();
        option_widget.setLayout(option_layout);

        // separators
        const label_mode = this.generate_separator('Compilation Options');
        const misc_label = this.generate_separator('Miscellaneous Options');
        const file_load_separator = this.generate_separator('File Options');

        const check_gp = new QCheckBox();
        check_gp.setText('gp');
        check_gp.addEventListener('stateChanged', () => {
            this.gp = !this.gp;
        });

        const check_cuda = new QCheckBox();
        check_cuda.setText('cuda');
        check_cuda.addEventListener('stateChanged', () => {
            this.cuda = !this.cuda;
        });

        const check_cuda_gp = new QCheckBox();
        check_cuda_gp.setText('cuda_gp');
        check_cuda_gp.addEventListener('stateChanged', () => {
            this.cuda_gp = !this.cuda_gp;
        });

        const check_nsgaii = new QCheckBox();
        check_nsgaii.setText('nsgaii');
        check_nsgaii.addEventListener('stateChanged', () => {
            this.nsgaii = !this.nsgaii;
            if (check_nsgaii.isChecked()) {
                run_obj.nb_plot_box.text_edit.clear();
                run_obj.nb_plot_box.widget.setEnabled(false);
            } else {
                run_obj.nb_plot_box.widget.setEnabled(true);
            }
        });

        const check_nsgaiii = new QCheckBox();
        check_nsgaiii.setText('nsgaiii');
        check_nsgaiii.addEventListener('stateChanged', () => {
            this.nsgaiii = !this.nsgaiii;
            if (check_nsgaiii.isChecked()) {
                run_obj.nb_plot_box.text_edit.clear();
                run_obj.nb_plot_box.widget.setEnabled(false);
            } else {
                run_obj.nb_plot_box.widget.setEnabled(true);
            }
        });

        const check_asrea = new QCheckBox();
        check_asrea.setText('asrea');
        check_asrea.addEventListener('stateChanged', () => {
            this.asrea = !this.asrea;
            if (check_asrea.isChecked()) {
                run_obj.nb_plot_box.text_edit.clear();
                run_obj.nb_plot_box.widget.setEnabled(false);
            } else {
                run_obj.nb_plot_box.widget.setEnabled(true);
            }
        });

        const check_ibea = new QCheckBox();
        check_ibea.setText('ibea');
        check_ibea.addEventListener('stateChanged', () => {
            this.ibea = !this.ibea;
            if (check_ibea.isChecked()) {
                run_obj.nb_plot_box.text_edit.clear();
                run_obj.nb_plot_box.widget.setEnabled(false);
            } else {
                run_obj.nb_plot_box.widget.setEnabled(true);
            }
        });

        const check_cdas = new QCheckBox();
        check_cdas.setText('cdas');
        check_cdas.addEventListener('stateChanged', () => {
            this.cdas = !this.cdas;
            if (check_cdas.isChecked()) {
                run_obj.nb_plot_box.text_edit.clear();
                run_obj.nb_plot_box.widget.setEnabled(false);
            } else {
                run_obj.nb_plot_box.widget.setEnabled(true);
            }
        });

        const check_memetic = new QCheckBox();
        check_memetic.setText('memetic');
        check_memetic.addEventListener('stateChanged', () => {
            this.memetic = !this.memetic;
        });

        const check_cmaes = new QCheckBox();
        check_cmaes.setText('cmaes');
        check_cmaes.addEventListener('stateChanged', () => {
            this.cmaes = !this.cmaes;
        });

        const check_v = new QCheckBox();
        check_v.setText('verbose mode');
        check_v.addEventListener('stateChanged', () => {
            this.verbose = !this.verbose;
        });

        const check_tl = new QCheckBox();
        check_tl.setText('true line mode');
        check_tl.addEventListener('stateChanged', () => {
            this.true_line = !this.true_line;
        });

        option_layout.addWidget(check_gp, 1, 0);
        option_layout.addWidget(check_nsgaii, 1, 1);
        option_layout.addWidget(check_nsgaiii, 1, 2);
        option_layout.addWidget(check_asrea, 2, 0);
        option_layout.addWidget(check_ibea, 2, 1);
        option_layout.addWidget(check_cdas, 2, 2);
        option_layout.addWidget(check_memetic, 3, 0);
        option_layout.addWidget(check_cmaes, 3, 1);

        // cuda options if nvcc is installed
        const cuda_test = new Pseudo_term('Cuda test');
        if (cuda_test.run_command('nvcc', ['--version']) !== -1) {
            option_layout.addWidget(check_cuda, 3, 2);
            option_layout.addWidget(check_cuda_gp, 3, 3);
        }

        option_layout.setHorizontalSpacing(150);
        option_layout.setVerticalSpacing(15);

        const option_widget_misc = new QWidget();
        const option_layout_misc = new QGridLayout();
        option_widget_misc.setLayout(option_layout_misc);
        option_layout_misc.addWidget(check_v, 0, 1);
        option_layout_misc.addWidget(check_tl, 0, 2);

        option_layout_misc.setSpacing(100);

        // compile button
        const compile_btn = new QPushButton();
        compile_btn.setText("Compile");
        compile_btn.setFixedSize(btn_width, btn_height);
        compile_btn.addEventListener('clicked', () => {
            if (this.ez_file_address === '') {
                new Win_alert("Please load a file");
            } else {
                // avoid problems by disabling all buttons
                option_widget.setEnabled(false);
                compile_btn.setEnabled(false);
                make_clean_btn.setEnabled(false);
                file_select.setEnabled(false);

                var dir_path = util.get_path(this.ez_file_address);

                // options to compile
                var params = [util.get_file_name(this.ez_file_address)];

                if (this.cuda)
                    params.push('-cuda');

                if (this.gp)
                    params.push('-gp');

                if (this.cuda_gp)
                    params.push('-cuda_gp');

                if (this.nsgaii)
                    params.push('-nsgaii');

                if (this.nsgaiii)
                    params.push('-nsgaiii');

                if (this.asrea)
                    params.push('-asrea');

                if (this.ibea)
                    params.push('-ibea');

                if (this.cdas)
                    params.push('-cdas');

                if (this.memetic)
                    params.push('-memetic');

                if (this.cmaes)
                    params.push('-cmaes');

                if (this.verbose)
                    params.push('-v');

                if (this.true_line)
                    params.push('-tl');

                // makefile address
                var ez_makefile = this.ez_file_address.substring(0, this.ez_file_address.length - 2);
                ez_makefile = ez_makefile.concat('mak');

                var run = output_compile.compile('easena', params, dir_path);

                output_compile.action_label.setText('Compiling ...');
                output_compile.action_animation.setFileName(cwd() + '/src/assets/loader2.gif');
                output_compile.action_animation.start();
                output_compile.action_widget.show();

                run.stdout.on('data', (data) => {
                    output_compile.text.insertPlainText(data.toString());
                });

                run.stderr.on('data', (data) => {
                    output_compile.text.insertPlainText(data.toString());
                });

                run.on('exit', (code) => {
                    if (code === 0) {
                        output_compile.text.insertPlainText('\n');
                        var make = output_compile.compile('make', ['-f', ez_makefile], dir_path);
                        make.stdout.on('data', (data) => {
                            output_compile.text.insertPlainText(data.toString());
                        });

                        make.on('close', (code, signal) => {
                            make_clean_btn.setEnabled(true);
                            file_select.setEnabled(true);
                            compile_btn.setEnabled(true);
                            option_widget.setEnabled(true);
                            output_compile.action_animation.stop();
                            // movie_loader.stop();
                            if(code === 0) {
                                plot_obj.graph_option.nb_plots = 1;
                                this.ready = 1;
                                run_obj.setReady(1);
                                if (this.nsgaii || this.nsgaiii || this.asrea || this.ibea || this.cdas) {
                                    run_obj.plot_type = '3D';
                                    plot_obj.graph_option.axe_z_box.widget.show();
                                    plot_obj.graph_option.plots.widget.hide();
                                    plot_obj.graph_option.color_palet.hide();
                                    run_obj.activate_island_model.setEnabled(false);
                                    run_obj.activate_island_model.setChecked(false);
                                } else {
                                    run_obj.plot_type = '2D';
                                    plot_obj.graph_option.axe_z_box.widget.hide();
                                    plot_obj.graph_option.plots.widget.show();
                                    plot_obj.graph_option.color_palet.show();
                                    run_obj.activate_island_model.setEnabled(true);
                                }
                                output_compile.action_label.setText('Compilation succeed');
                                output_compile.action_animation.setFileName(cwd() + '/src/assets/ok_icon.png');
                            } else {
                                output_compile.action_label.setText('Compilation failed');
                                output_compile.action_animation.setFileName(cwd() + '/src/assets/fail_icon.png');
                            }
                        });

                    } else {
                        this.ready = 0;
                        make_clean_btn.setEnabled(true);
                        file_select.setEnabled(true);
                        compile_btn.setEnabled(true);
                        option_widget.setEnabled(true);
                        output_compile.action_label.setText('Compilation failed');
                        output_compile.action_animation.stop();
                        output_compile.action_animation.setFileName(cwd() + '/src/assets/fail_icon.png');
                    }
                });
            }
            this.options = '';
        });

        // make clean button
        const make_clean_btn = new QPushButton();
        make_clean_btn.setText("Make clean");
        make_clean_btn.setFixedSize(btn_width, btn_height);
        make_clean_btn.addEventListener('clicked', () => {
            if (this.ez_file_address === '') {
                new Win_alert("Please load a file");
            } else {
                var ez_makefile = this.ez_file_address.substring(0, this.ez_file_address.length - 2);
                ez_makefile = ez_makefile.concat('mak');
                var make_clean = output_compile.compile('make', ['-C', this.dir_path, '-f', ez_makefile, 'easeaclean']);

                make_clean.stdout.on('data', (data) => {
                    output_compile.text.insertPlainText(data.toString());
                });

                make_clean.on('close', (code) => {
                    if(code === 0){
                        this.ready = 0;
                        run_obj.setReady(0);
                        output_compile.action_label.setText('Make clean OK');
                        output_compile.action_animation.setFileName(cwd() + '/src/assets/ok_icon.png');
                    } else {
                        output_compile.action_label.setText('Make clean failed');
                        output_compile.action_animation.setFileName(cwd() + '/src/assets/fail_icon.png');
                    }
                });

            }

        });

        const label_loaded = new QLabel();
        label_loaded.setText("File loaded : ");
        label_loaded.setStyleSheet("margin-left: 15px");

        const file_loaded = new QLabel();
        file_loaded.setText('Empty');

        // load file dialog
        const file_select = new QPushButton();
        file_select.setText("Load a file");
        file_select.setFixedSize(btn_width, btn_height);

        file_select.setStyleSheet("margin-left: 20px");
        file_select.addEventListener('clicked', () => {
            const fileDialog = new QFileDialog();
            fileDialog.setFileMode(FileMode.ExistingFile);
            fileDialog.setNameFilter('EASEA files (*.ez)');

            if (fileDialog.exec()) {
                var file = fileDialog.selectedFiles().toString();
                if (file) {
                    this.ez_file_address = file;
                    this.dir_path = util.get_path(this.ez_file_address);
                    run_obj.setDirPath(this.dir_path);
                    file_loaded.setText(util.get_file_name(this.ez_file_address));
                    run_obj.setEzFileAddress(this.ez_file_address);
                }
            }
        });

        // load file box
        const widget_load = new QWidget();
        const hbox_load = new QBoxLayout(0);

        hbox_load.addWidget(label_loaded);
        hbox_load.addWidget(file_loaded);
        widget_load.setLayout(hbox_load);
        
        // // compiling label
        // const compiling_widget = new QWidget();
        // const compiling_layout = new QBoxLayout(0);
        // compiling_widget.setLayout(compiling_layout);

        // const compiling_label = new QLabel();
        // compiling_layout.addWidget(compiling_label);

        // const compiling_animation = new QLabel();
        // const movie_loader = new QMovie();
        // // movie_loader.setFileName(cwd() + '/src/assets/loader2.gif');
        // compiling_animation.setMovie(movie_loader);
        // compiling_layout.addWidget(compiling_animation);
        // compiling_widget.hide();

        // compile box
        const widget_comp = new QWidget();
        const hbox_comp = new QBoxLayout(0);
        hbox_comp.addWidget(compile_btn);
        hbox_comp.addWidget(make_clean_btn);
        widget_comp.setLayout(hbox_comp);

        global_layout.addWidget(file_load_separator, undefined, AlignmentFlag.AlignCenter);
        global_layout.addWidget(widget_load, undefined, AlignmentFlag.AlignCenter);
        global_layout.addWidget(file_select, undefined, AlignmentFlag.AlignCenter);
        global_layout.addWidget(label_mode, undefined, AlignmentFlag.AlignCenter);
        global_layout.addWidget(option_widget, undefined, AlignmentFlag.AlignCenter);
        global_layout.addWidget(misc_label, undefined, AlignmentFlag.AlignCenter);;
        global_layout.addWidget(option_widget_misc, undefined, AlignmentFlag.AlignCenter);
        global_layout.addWidget(widget_comp, undefined, AlignmentFlag.AlignCenter);
        // global_layout.addWidget(compiling_widget, undefined, AlignmentFlag.AlignCenter);
        global_layout.addWidget(widget_console, undefined, AlignmentFlag.AlignCenter);

        return global_layout;
    }

    setReady(val: Number) {
        this.ready = val;
    }

    generate_separator(text:string) {
        const sep = new QLabel();
        sep.setText(text);
        sep.setAlignment(AlignmentFlag.AlignCenter);
        sep.setFixedSize(500,50);
        sep.setStyleSheet(`
            font-size: 15pt;
            border-bottom: 0.5px solid;
            background-color:rgb(226,226,226);
        `);

        return sep;
    }
}