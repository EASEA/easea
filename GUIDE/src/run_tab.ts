/**
 * @author Clément Ligneul <clement.ligneul@etu.unistra.fr>
 */

import { Pseudo_term } from './pseudo_term';
import { Win_alert } from './win_alert';
import * as util from './utilities';
import { compile_obj, plot_obj, running_islands, running_proc, tab_menu } from './index';
import { QGridLayout, QLabel, QWidget, QBoxLayout, QPushButton, QCheckBox, Direction, QProgressBar, QMovie, AlignmentFlag } from '@nodegui/nodegui';
import { Run_options } from './run_options_win';
import { Parent_options_win } from './parents_options_win';
import { Offspring_options_win } from './offspring_options_win';
import { Island_options_win } from './island_options_win';
import * as fs_extra from 'fs-extra';
import { Advanced_option_widget } from './advanced_option_widget';
import { running_plot } from './index';
import { ChildProcess } from 'child_process';
import { Results_win } from './results_win';
import ip from 'ip';
import fs, { copyFileSync, cpSync, mkdirSync } from 'fs';
import { cwd, exit } from 'process';
import * as paths from './paths';

// first process
export let initial_proc: ChildProcess;

export class Run_tab {
    ez_file_address: string;
    dir_path: string;
    dir_save: string = '';
    batch_size: number;
    ready: Number;  // avoid to run without compiling
    batch_display: QLabel;
    options: string;
    option_obj: Run_options = new Run_options();
    island_obj: Island_options_win = new Island_options_win();
    parent_obj: Parent_options_win = new Parent_options_win();
    off_obj: Offspring_options_win = new Offspring_options_win();
    island_model: boolean = false;
    labels_box: QBoxLayout = new QBoxLayout(Direction.TopToBottom);
    running_widget: QWidget = new QWidget();
    running_label: QLabel = new QLabel();
    running_animation_label: QLabel = new QLabel();
    running_animation_movie: QMovie = new QMovie();
    finished_label: QLabel = new QLabel();
    nb_plots: number = 1;
    progress_bar: QProgressBar = new QProgressBar();
    plot_type: string = '2D';
    total_generations: number = 0;
    results_button: QPushButton = new QPushButton();
    run_results: string[] = [];
    runned_proc: number = 1;
    island_run_array: number[][][] = [];       // stats on island runs (dim 0 = group num, dim 1 = island num, dim 2 = ready for (run_num) )
    remaining_proc: number = 0;
    total_runs: number = 0;
    batch_id: number = 0;
    plot_last_gen_only: boolean = true;

    //thresholds array
    best_fit_result_thresh: number[][] = [];    // [0] = value [1] = gen
    worst_fit_result_thresh: number[][] = [];
    avg_fit_result_thresh: number[][] = [];
    std_dev_result_thresh: number[][] = [];

    // buttons
    activate_island_model: QCheckBox = new QCheckBox();
    island_btn: QPushButton = new QPushButton();
    general_option_btn: QPushButton = new QPushButton();
    parent_btn: QPushButton = new QPushButton();
    offsp_btn: QPushButton = new QPushButton();
    nb_plot_box: Advanced_option_widget = new Advanced_option_widget('Plot', 0, '1');
    run_btn: QPushButton = new QPushButton();
    stop_proc_btn: QPushButton = new QPushButton()


    constructor(ez_file: string, dir: string) {
        this.ez_file_address = ez_file;
        this.dir_path = dir;
        this.batch_size = 1;
        this.ready = 0;
        this.batch_display = new QLabel();
        this.options = '';
        this.finished_label.hide();
    }

    generate() {
        const global_layout = new QGridLayout();

        // progress bar
        this.progress_bar.setFixedSize(205, 30);
        this.progress_bar.setMaximum(100);
        this.progress_bar.setValue(0);
        this.progress_bar.hide();

        // running widget
        this.running_label.setAlignment(AlignmentFlag.AlignCenter);
        const running_layout = new QBoxLayout(0);
        this.running_widget.setLayout(running_layout);
        
        this.running_animation_movie.setFileName(cwd() + '/src/assets/loader2.gif');
        this.running_animation_label.setMovie(this.running_animation_movie);
        
        running_layout.addWidget(this.running_label);
        running_layout.addWidget(this.running_animation_label);
        this.running_widget.hide();

        // output run console
        const output_run = new Pseudo_term('Output :');
        const widget_console = new QWidget();
        const layout_console = new QGridLayout();
        layout_console.addWidget(output_run.window);
        widget_console.setLayout(layout_console);
        output_run.window.setFixedSize(1000, 500);
        output_run.text.setFixedSize(970, 435);

        // batch size
        this.batch_display.setText('Batch size : ' + this.batch_size);
        this.batch_display.setFixedSize(200, 30);

        // ip address
        const ip_label = new QLabel();
        ip_label.setText('Actual IP address : ' + ip.address());
        ip_label.setFixedSize(300,30);

        // activate island model
        this.activate_island_model.setText('Activate Island Model');
        this.activate_island_model.addEventListener('stateChanged', () => {
            this.island_model = this.activate_island_model.isChecked();
        });

        // nb plots choice
        this.nb_plot_box.widget.setFixedSize(160, 50);
        this.nb_plot_box.text_edit.setFixedSize(50, 30);
        const nb_plot_box_label = new QLabel();
        nb_plot_box_label.setText('graph(s)');
        this.nb_plot_box.layout.addWidget(nb_plot_box_label);
        this.nb_plot_box.text_edit.addEventListener('textChanged', () => {
            let text = this.nb_plot_box.text_edit.text();
            let val = Number(text);

            text === '' ? this.nb_plots = 1 : this.nb_plots = val;
        });

        // general options
        const run_option = new Run_options();
        this.general_option_btn.setText("General Options");
        this.general_option_btn.setFixedSize(155, 33);
        this.general_option_btn.addEventListener('clicked', () => {
            this.option_obj = run_option.execution();

            if (this.option_obj.batch_size > 0) {
                this.batch_size = Number(this.option_obj.batch_size);
                this.batch_display.setText("Batch size : " + this.batch_size.toString());
            }
        });

        // parent options
        const parent_options = new Parent_options_win();
        this.parent_btn.setText("Parents Options");
        this.parent_btn.setFixedSize(155, 33);
        this.parent_btn.addEventListener('clicked', () => {
            this.parent_obj = parent_options.execution();
        });

        // offspring options
        const offsp_option = new Offspring_options_win();
        this.offsp_btn.setText("Offspring Options");
        this.offsp_btn.setFixedSize(205, 33);
        this.offsp_btn.setStyleSheet('margin-left: 30px');
        this.offsp_btn.addEventListener('clicked', () => {
            this.off_obj = offsp_option.execution();
        });

        // remote Island options
        const island_option = new Island_options_win();
        this.island_btn.setText("Island Options");
        this.island_btn.setFixedSize(155, 33);
        this.island_btn.addEventListener('clicked', () => {
            this.island_obj = island_option.execution();
        });

        this.running_widget.hide();

        //see results
        this.results_button = new QPushButton();
        this.results_button.setFixedSize(155, 33);
        this.results_button.setText('See Results');
        this.results_button.addEventListener('clicked', () => {
            const results = new Results_win();
            results.generate();
            results.execute();
        });
        this.results_button.hide();


        // run button
        this.run_btn.setText("Run !");
        this.run_btn.setFixedSize(155, 33);
        this.run_btn.setStyleSheet('background-color: #5CFF00;');
        this.run_btn.addEventListener('clicked', () => {

            // empty the tables
            this.island_run_array.length = 0;

            // erase previous graph
            plot_obj.image_label.setFixedSize(210, 30);
            plot_obj.image_label.setText('No graph to display');
            plot_obj.btn_widget.setEnabled(false);
                
            
            if (this.ready === 0) {
                new Win_alert('Please compile your file before run', 'Run error');
                return;
            }

            if (this.nb_plots <= 0) {
                new Win_alert('The number of plots must be positive', 'Run error');
                return;
            }

            if ((this.nb_plots !== 1 && this.nb_plots % 1 !== 0)) {
                new Win_alert('The number of plots must be an integer', 'Run error');
                return;
            }

            if (isNaN(this.nb_plots)) {
                new Win_alert('The number of plots is invalid', 'Run error');
                return;
            }

            if (this.island_obj.local === false) {
                new Win_alert('Remote islands not yet implemented');
                return;
            }

            if (this.nb_plots > this.option_obj.nb_gen) {
                new Win_alert('The number of plots is invalid \n(nb generations \u2265 nb plots required)', 'Run error');
                return;
            }

            if (this.plot_type === '3D' && this.batch_size > 1) {
                new Win_alert('Batch mode is not yet available for multi-objective programs', 'Run error');
                return;
            }

            this.run_results.length = 0;

            this.enable_buttons(false);
            this.progress_bar.reset();
            this.progress_bar.setValue(0);
            this.progress_bar.show();
            this.results_button.hide();
            this.plot_last_gen_only=true;
            plot_obj.graph_option.check_gen.setChecked(true);

            if (this.ez_file_address) {
                let cmd = this.ez_file_address.substring(0, this.ez_file_address.length - 3);

                if (this.batch_size > 0) {
                
                    // reset running animation & output
                    this.running_animation_movie.start();
                    this.running_animation_label.show();

                    // delete existing plot files
                    fs_extra.emptyDirSync(this.dir_path + paths.dir_tmp_path);

                    // create directories for results
                    try{
                        if(!fs.existsSync(this.dir_path + paths.ui_dir_path))
                            fs.mkdirSync(this.dir_path + paths.ui_dir_path);
                        if(!fs.existsSync(this.dir_path + paths.dir_tmp_path))
                            fs.mkdirSync(this.dir_path + paths.dir_tmp_path);
                        if(!fs.existsSync(this.dir_path + paths.dir_results_path))
                            fs.mkdirSync(this.dir_path + paths.dir_results_path);

                    } catch(e) {
                        console.log(e);
                        console.log(e);
                        exit(1);
                    }

                    // initialization
                    let nb_files:number = 0;
                    this.run_results.length = 0;
                    this.remaining_proc = 0;
                    this.total_runs = 0;

                    this.worst_fit_result_thresh.length = 0;
                    this.best_fit_result_thresh.length = 0;
                    this.avg_fit_result_thresh.length = 0;
                    this.std_dev_result_thresh.length = 0;

                    for(let x = 0; x <= this.batch_size; x++){
                        this.best_fit_result_thresh[x] = [];
                        this.worst_fit_result_thresh[x] = [];
                        this.avg_fit_result_thresh[x] = [];
                        this.std_dev_result_thresh[x] = [];
                    }

                    if(this.island_obj.local && this.activate_island_model.isChecked()){
                        nb_files = Math.floor(this.island_obj.nb_islands / this.island_obj.nb_isl_per_run);

                        this.total_runs = this.batch_size * this.island_obj.nb_isl_per_run;

                        // IP file + island runs array generation
                        let port_cpt:number = 2929;
                        let cpt_rank = 0;
                        
                        // prepare local islands array
                        for(let i = 1; i <= nb_files; i++){
                            let file_name = this.dir_path + paths.dir_tmp_path + 'ip_file' + i + '.txt'
                            let fd = fs.openSync(file_name, 'ax+');

                            this.island_run_array[i-1] = [];
                            
                            for(let j = 0; j < this.island_obj.nb_isl_per_run; j++){
                                fs.writeFileSync(fd, '127.0.0.1:' + port_cpt + '\n', {encoding:'utf-8'});
                                this.island_run_array[i-1][j] = [];
                                this.island_run_array[i-1][j][0] = i-1;      // available for run
                                this.island_run_array[i-1][j][1] = i-1;      // island num
                                this.island_run_array[i-1][j][2] = port_cpt; // port associated
                                this.island_run_array[i-1][j][3] = cpt_rank; // rank
                                this.island_run_array[i-1][j][4] = 0;        // actual pid (for stop button)
                                
                                port_cpt++;
                                cpt_rank++;
                            }

                            fs.closeSync(fd);
                        }
                    }

                    // batch loop
                    this.batch_id = Date.now();
                    
                    for (let i = 0; i < this.batch_size; i++) {

                        // file result
                        let ms = Date.now();
                        let date_ob = new Date(ms);
                        let month = ("0" + (date_ob.getMonth() + 1)).slice(-2);
                        let day = date_ob.getDate();
                        let year = date_ob.getFullYear();
                        let hours = ((date_ob.getHours()<10?'0':'') + date_ob.getHours());
                        let minutes = ((date_ob.getMinutes()<10?'0':'') + date_ob.getMinutes());
                        let seconds = ((date_ob.getSeconds()<10?'0':'') + date_ob.getSeconds());
                        let time = year + '-' + month + '-' + day + '_' + hours + '-' + minutes + '-' + seconds;
                        let path = compile_obj.ez_file_address.split('/');
                        let ez_filname = path[path.length-1];

                        // dir creation
                        if(i===0){
                            this.dir_save = this.dir_path  + paths.dir_results_path + ez_filname.substring(0, ez_filname.length - 3) + '/' + time + '/';
                            try{
                                mkdirSync(this.dir_save, {recursive:true});
                            } catch (e){
                                if(e !== 'EEXIST')
                                console.log("Error during result dir generation : " + e);
                            }
                            // save logs in results dir
                            // if(this.plot_type == "2D" && !this.island_model){
                            //     try{
                            //         let a = this.ez_file_address.substring(0, this.ez_file_address.length - 3);
                            //         let n = a.split(`/`)
                            //         cmd = this.dir_save + n[n.length-1]
                            //         copyFileSync(a, this.dir_save + n[n.length-1]);
                            //     } catch (e) {
                            //         console.log(e)
                            //     }
                            // }
                        }

                        // file creation
                        let file = this.dir_save + ez_filname.substring(0, ez_filname.length - 3) + '_' + time;
                        if(!this.island_model){
                            file = file + '_run_' + (i+1) +'.log';

                        }
                        
                        // prepare options
                        this.build_run_options(i);

                        // runs start here
                        this.run_results = [];

                        // classic runs
                        if(!this.island_model){
                            
                            if(i === 0){
                                var run = output_run.run(cmd, this.nb_plots, true, file, this.options, this.dir_path, i + 1);
                            } else {
                                var run = output_run.run(cmd, this.nb_plots, false, file, this.options, this.dir_path, i + 1);
                            }

                            if (i === 0)
                                initial_proc = run;

                            run.on('close', (code: Number, signal: Number) => {
                                if (signal?.toString() === 'SIGTERM' || code !== 0) {
                                    this.enable_buttons(true);
                                    this.running_label.setText('Interrupted');
                                    this.running_animation_movie.stop();
                                    this.running_animation_label.hide();
                                    this.running_widget.show();
                                    running_plot.length = 0;
                                } else if (running_proc.length !== 0) {
                                    this.running_label.setText('Runs in progress ...');
                                    this.running_widget.show();
                                }

                                if (code === 0 && running_proc.length === 0 && this.plot_type === '2D') {
                                    this.runned_proc = this.batch_size;
                                    this.results_button.show();
                                }
                            });

                        // local island model runs
                        } else if (this.island_obj.local) {
                            running_islands.length = 0;

                            for(let k = 0; k < nb_files; k++){
                                if(this.island_run_array[k][0][0] < this.batch_size){

                                    console.log('---------------- Run subprocess ' + (k) + ' ----------------')

                                    for(let l = 0; l < this.island_run_array[k].length; l++){
                                        if(l === 0 && k === 0){
                                            this.run_local_islands(k, l, file, output_run, this.island_run_array[k][l][3], nb_files, true);
                                        } else {
                                            this.run_local_islands(k, l, file, output_run, this.island_run_array[k][l][3], nb_files, false);
                                        }
                                    }
                                }
                            }

                        // remote island model runs
                        } else if (!this.island_obj.local) {
                            this.enable_buttons(true);
                            console.log('Debug : ok remote');
                            return;

                        } else {
                            new Win_alert('Error during runs: unknown run type (not local or remote)');
                            this.enable_buttons(true);
                            return;
                        }


                        this.running_label.setText('Runs in progress ...');
                        this.running_widget.show();
                        
                        if (this.island_model === true) {
                            this.finished_label.setText('Completed Runs : ' + 0 + '/' + this.total_runs);
                            this.finished_label.show();
                            break;
                        }
                        
                        this.finished_label.setText('Completed Runs : ' + (this.batch_size - running_proc.length) + '/' + this.batch_size);
                        this.finished_label.show();
                    }
                } else {
                    new Win_alert("Incorrect batch size");
                }

            } else {
                new Win_alert("Please load a file");
            }
        });

        // stop all processes
        this.stop_proc_btn.setText("Stop all runs");
        this.stop_proc_btn.setFixedSize(205, 33);
        this.stop_proc_btn.setStyleSheet('background-color: red; margin-left: 30px');
        this.stop_proc_btn.addEventListener('clicked', () => {
            util.kill_all(running_proc);
            util.kill_all(running_plot);
            util.kill_all(running_islands);
        });

        // run box
        const widget_run = new QWidget();
        const hbox_run = new QGridLayout();
        hbox_run.addWidget(this.batch_display, 0, 0);
        hbox_run.addWidget(ip_label, 0, 2);
        hbox_run.addWidget(this.activate_island_model, 1, 0);
        hbox_run.addWidget(this.island_btn, 1, 1);
        hbox_run.addWidget(this.general_option_btn, 2, 0);
        hbox_run.addWidget(this.parent_btn, 2, 1);
        hbox_run.addWidget(this.offsp_btn, 2, 2);
        hbox_run.addWidget(this.nb_plot_box.widget, 3, 0);
        hbox_run.addWidget(this.run_btn, 3, 1);
        hbox_run.addWidget(this.stop_proc_btn, 3, 2);
        hbox_run.addWidget(this.running_widget, 4, 1);
        hbox_run.addWidget(this.finished_label, 4, 0);
        hbox_run.addWidget(this.progress_bar, 4, 2);
        hbox_run.addWidget(this.results_button, 5, 1);


        widget_run.setLayout(hbox_run);

        this.island_btn.hide();

        this.activate_island_model.addEventListener('stateChanged', () => {
            if (this.activate_island_model.isChecked()) {
                this.island_btn.show();
            } else {
                this.island_btn.hide();
            }
        });

        global_layout.addWidget(widget_run, 0, 0);
        global_layout.addWidget(widget_console, 1, 0);
        global_layout.setRowStretch(0, 2);

        return global_layout;
    }

    // enable interface buttons
    enable_buttons(v: boolean) {
        if (v) {
            this.activate_island_model.setEnabled(true);
            this.run_btn.setEnabled(true);
            this.parent_btn.setEnabled(true);
            this.island_btn.setEnabled(true);
            this.offsp_btn.setEnabled(true);
            this.general_option_btn.setEnabled(true);
            if (!compile_obj.nsgaii && !compile_obj.nsgaiii && !compile_obj.cdas)
                this.nb_plot_box.widget.setEnabled(true);
            tab_menu.tabs[0].setEnabled(true);
        } else {
            this.activate_island_model.setEnabled(false);
            this.run_btn.setEnabled(false);
            this.parent_btn.setEnabled(false);
            this.island_btn.setEnabled(false);
            this.offsp_btn.setEnabled(false);
            this.general_option_btn.setEnabled(false);
            this.nb_plot_box.widget.setEnabled(false);
            tab_menu.tabs[0].setEnabled(false);
        }
    }

    setEzFileAddress(path: string) {
        this.ez_file_address = path;
    }

    setDirPath(path: string) {
        this.dir_path = path;
    }

    setReady(val: number) {
        this.ready = val;
    }

    run_local_islands(island_group: number, island_num: number, file: string, output_run: Pseudo_term, rank: number, nb_distributed_island: number, print: boolean) {
        console.log('running ipfile' + (this.island_run_array[island_group][island_num][1]) + '.txt | on port : ' + this.island_run_array[island_group][island_num][2] + ' | rank = ' + rank);
        
        const cmd = this.ez_file_address.substring(0, this.ez_file_address.length - 3);
        this.build_run_options(rank);
        this.options = this.options.concat(' --serverPort ' + this.island_run_array[island_group][island_num][2]);
        this.options = this.options.concat(' --ipFile ' + this.dir_path + paths.dir_tmp_path + 'ip_file' + (this.island_run_array[island_group][island_num][1] + 1) + '.txt');
        
        var local_run = output_run.run(cmd, this.nb_plots, print, file, this.options, this.dir_path, (rank + 1));
        
        if(rank === 0)
            initial_proc = local_run

        local_run.on('close', (code: Number, signal: Number) => {
            if (signal?.toString() === 'SIGTERM' || code !== 0) {
                this.enable_buttons(true);
                this.running_label.setText('Interrupted');
                this.running_animation_movie.stop();
                this.running_animation_label.hide();
                this.running_widget.show();
                running_plot.length = 0;
            } else if (running_proc.length !== 0) {
                this.running_label.setText('Runs in progress ...');
                this.running_widget.show();
            }

            if (code === 0) {
                let ok = 1;
                this.island_run_array[island_group][island_num][0] += nb_distributed_island;
                let next_run = this.island_run_array[island_group][island_num][0];

                for(let x = 0; x < this.island_run_array[island_group].length; x++){
                    if(this.island_run_array[island_group][x][0] != next_run){
                        ok = 0;
                        break;
                    }
                }

                if(next_run < this.batch_size){    
                    if(ok){
                        console.log('---------------- Running subprocess ' + (next_run) + ' ----------------')

                        for(let i = 0; i < this.island_run_array[island_group].length; i++){
                            // let new_file = file.split('_');
                            // new_file[new_file.length-1] = i + '.txt';
                            // file = new_file.join('_');

                            if(i === 0){
                                print = true;
                            } else {
                                print = false;
                            }
                            this.run_local_islands(island_group, i, file, output_run, i+(next_run*this.island_obj.nb_isl_per_run), nb_distributed_island, print);
                            
                        }
                    }
                } else {                    
                    // all runs are finished
                    if(this.remaining_proc === this.total_runs && ok){
                        this.running_label.setText('Finished');

                        this.running_animation_movie.stop();
                        this.running_animation_label.hide();

                        this.runned_proc = this.total_runs;
                        this.enable_buttons(true);
                        this.results_button.show();
                    }
                }

            // all processes are stopped
            } else {
                util.kill_all(running_proc);
                util.kill_all(running_plot);
                console.log('process chain stopped');
            }
        });
    }

    // write the run options in this.options
    build_run_options(rank: number) {
        this.options = '';

        let seed_cpt = rank;

        // general options
        if (rank !== 0 || !this.option_obj.plot_stats) {
            this.options = this.options.concat(' --plotStats 0')
        } else {
            this.options = this.options.concat(' --plotStats 1')
        }

        if (!isNaN(this.option_obj.compression))
            this.options = this.options.concat(' --compression ' + this.option_obj.compression);

        if (!isNaN(this.option_obj.pop_size))
            this.options = this.options.concat(' --popSize ' + this.option_obj.pop_size);

        if (!isNaN(this.option_obj.elite_type))
            this.options = this.options.concat(' --eliteType ' + this.option_obj.elite_type);

        if (!isNaN(this.option_obj.nb_elite))
            this.options = this.options.concat(' --elite ' + this.option_obj.nb_elite.toString());

        if (!isNaN(this.option_obj.nb_gen))
            this.options = this.options.concat(' --nbGen ' + this.option_obj.nb_gen);

        if (!isNaN(this.option_obj.time_limit))
            this.options = this.options.concat(' --timeLimit ' + this.option_obj.time_limit);

        if (this.option_obj.select_op !== 'Tournament')
            this.options = this.options.concat(' --selectionOperator=' + this.option_obj.select_op);

        if (!isNaN(this.option_obj.select_pressure))
            this.options = this.options.concat(' --selectionPressure=' + this.option_obj.select_pressure);

        if (this.option_obj.reduce_final_op !== 'Tournament')
            this.options = this.options.concat(' --reduceFinalOperator=' + this.option_obj.reduce_final_op);

        if (!isNaN(this.option_obj.reduce_final_pressure))
            this.options = this.options.concat(' --reduceFinalPressure=' + this.option_obj.reduce_final_pressure);

        if (!isNaN(this.option_obj.optimize_it))
            this.options = this.options.concat(' --optimiseIterations ' + this.option_obj.optimize_it);

        if (!isNaN(this.option_obj.baldwinism))
            this.options = this.options.concat(' --baldwinism ' + this.option_obj.baldwinism);

        if (this.option_obj.output_file)
            this.options = this.options.concat(' --outputfile ' + this.option_obj.output_file);

        if (this.option_obj.input_file)
            this.options = this.options.concat(' --inputfile ' + this.option_obj.input_file);

        if (this.option_obj.generate_csv_file)
            this.options = this.options.concat(' --generateCSVFile 1');

        if (this.option_obj.generate_plot_script)
            this.options = this.options.concat(' --generatePlotScript 1');

        if (this.option_obj.generate_r_script)
            this.options = this.options.concat(' --generateRScript 1');

        if (this.option_obj.print_init_pop)
            this.options = this.options.concat(' --printInitialPopulation 1');

        if (this.option_obj.print_final_pop)
            this.options = this.options.concat(' --printFinalPopulation 1');

        if (rank === 0 && !this.option_obj.print_stats)
            this.options = this.options.concat(' --printStats 0');

        if (this.option_obj.save_pop)
            this.options = this.options.concat(' --savePopulation 1');

        if (this.option_obj.start_file)
            this.options = this.options.concat(' --startFromFile 1');

        if (!isNaN(this.option_obj.fstgpu_param))
            this.options = this.options.concat(' --fstgpu ' + this.option_obj.fstgpu_param);

        if (!isNaN(this.option_obj.lstgpu_param))
            this.options = this.options.concat(' --lstgpu ' + this.option_obj.lstgpu_param);

        if (!isNaN(this.option_obj.thread_number) && this.option_obj.thread_number > 1)
            this.options = this.options.concat(' --nbCPUThreads ' + this.option_obj.thread_number);

        if (this.option_obj.u1)
            this.options = this.options.concat(' --u1 ' + this.option_obj.u1);

        if (this.option_obj.u2)
            this.options = this.options.concat(' --u2 ' + this.option_obj.u2);

        if (this.option_obj.u3)
            this.options = this.options.concat(' --u3 ' + this.option_obj.u3);

        if (this.option_obj.u4)
            this.options = this.options.concat(' --u4 ' + this.option_obj.u4);

        if (this.option_obj.u5)
            this.options = this.options.concat(' --u5 ' + this.option_obj.u5);

        if (this.island_model){
            this.options = this.options.concat(' --remoteIslandModel 1');
        } else {
            this.options = this.options.concat(' --remoteIslandModel 0');
        }
            
        const timestamp_seed = Math.trunc(Date.now() / 1000);

        if (!isNaN(this.option_obj.proc_tab[rank].seed_value)) {
            this.options = this.options.concat(' --seed ' + this.option_obj.proc_tab[rank].seed_value);
        } else {
            if (isNaN(this.option_obj.seed)) {
                this.options = this.options.concat(' --seed ' + (timestamp_seed + seed_cpt));
            } else {
                this.options = this.options.concat(' --seed ' + (this.option_obj.seed + seed_cpt));
            }
        }

        // island model options
        if (this.activate_island_model.isChecked()) {
            if (this.island_obj.ip_file){
                this.options = this.options.concat(' --ipFile ' + this.island_obj.ip_file);
            } else if (!this.island_obj.local){
                new Win_alert('IP file require for remote island model\n', 'Run options');
                this.enable_buttons(true);
                return ;
            }

            if (!isNaN(this.island_obj.migration_proba) && this.island_obj.migration_proba <= 1 && this.island_obj.migration_proba >= 0)
                this.options = this.options.concat(' --migrationProbability ' + this.island_obj.migration_proba.toString());

            if (this.island_obj.reevaluate)
                this.options = this.options.concat(' --reevaluateImmigrants 1');
        }

        // offspring options
        if (!isNaN(this.off_obj.surviving_off)) {
            let off_opt = this.off_obj.surviving_off.toString();
            if (this.off_obj.surviving_off_type === '%')
                off_opt = off_opt.concat('%');

            this.options = this.options.concat(' --survivingOffspring ' + off_opt);
        }

        if (!isNaN(this.off_obj.size_off))
            this.options = this.options.concat(' --nbOffspring ' + this.off_obj.size_off.toString());

        if (this.off_obj.reduce_op !== 'Tournament')
            this.options = this.options.concat(' --reduceOffspringOperator=' + this.off_obj.reduce_op.toString());

        if (!isNaN(this.off_obj.reduce_pressure) && this.off_obj.reduce_pressure !== 2)
            this.options = this.options.concat(' --reduceOffspringPressure=' + this.off_obj.reduce_pressure.toString());

        // parents options
        if (!isNaN(this.parent_obj.surviving)) {
            let parent_opt = this.parent_obj.surviving.toString();

            if (this.parent_obj.surviving_parent_type === '%')
                parent_opt = parent_opt.concat('%');

            this.options = this.options.concat(' --survivingParents ' + parent_opt);
        }

        if (this.parent_obj.reduce_op !== 'Tournament')
            this.options = this.options.concat(' --reduceParentsOperator=' + this.parent_obj.reduce_op.toString());

        if (!isNaN(this.parent_obj.reduce_pressure) && this.parent_obj.reduce_pressure !== 2)
            this.options = this.options.concat(' --reduceParentsPressure=' + this.parent_obj.reduce_pressure.toString());
    }
}