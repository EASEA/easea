/**
 * @author Clément Ligneul <clement.ligneul@etu.unistra.fr>
 */

const { QGridLayout } = require("@nodegui/nodegui");
import { QWidget, QLabel, QPushButton, QTextBrowser, TextInteractionFlag, QMovie, QBoxLayout, Direction } from '@nodegui/nodegui';
import { spawn } from 'child_process';
import { spawnSync } from 'child_process';
import { running_plot, running_proc } from './index';
import { Win_alert } from './win_alert';
import * as plot_generation from './plot_generation';
import { plot_obj, run_obj } from './index';

require('child_process').spawn('node', ['--version'], {
    env: {
        PATH: process.env.PATH
    }
});


export class Pseudo_term {
    window: QWidget;
    text: QTextBrowser;
    label_text: QLabel;
    action_label: QLabel;
    action_widget: QWidget;
    action_animation: QMovie;

    constructor(label_txt: string) {
        this.window = new QWidget
        this.window.setFixedSize(1000, 300);

        const layout = new QGridLayout();
        this.window.setLayout(layout);

        this.label_text = new QLabel();
        this.label_text.setText(label_txt);

        this.text = new QTextBrowser();
        this.text.setReadOnly(true);
        this.text.setTextInteractionFlags(TextInteractionFlag.NoTextInteraction);
        this.label_text.setTextInteractionFlags(TextInteractionFlag.NoTextInteraction);

        const reset_btn = new QPushButton();
        reset_btn.setText('Clear');
        reset_btn.setFixedSize(95, 27);
        reset_btn.addEventListener('clicked', () => {
            this.text.clear();
        });

        this.action_widget = new QWidget();
        const action_layout = new QBoxLayout(Direction.LeftToRight);
        this.action_widget.setLayout(action_layout);
        this.action_label = new QLabel();
        const label_movie = new QLabel();
        this.action_animation = new QMovie();
        label_movie.setMovie(this.action_animation);
        action_layout.addWidget(this.action_label);
        action_layout.addWidget(label_movie);
        this.action_label.setFixedSize(150, 30);
        
        layout.addWidget(this.label_text, 0, 0);
        layout.addWidget(this.action_widget, 0, 1);
        layout.addWidget(reset_btn, 0, 2);
        layout.addWidget(this.text, 1, 0, 2, 0);
    }

    compile(command: string, params?: string[], dir?: string) {
        let array_params = [];

        if (params == undefined) {
            this.text.insertPlainText('\n$ ' + command + '\n');
        } else {
            let res = '';
            for (let i = 0; i < params.length; i++) {
                if (params[i] != '') {
                    res = res.concat(params[i], ' ');
                    array_params.push(params[i]);
                }
            }

            if (command != 'nvcc')
                this.text.insertPlainText('\n$ ' + command + ' ' + res + '\n');
        }

        process.env.EZ_PATH = '/usr/local/easena/'
        let child = spawn(command, array_params, {
            cwd: dir,
            env: process.env
        });

        // handle error in spawn
        child.on('error', function (err) {
            new Win_alert('' + err);
            return 1;
        });

        child.stderr.on('data', (data) => {
            this.text.insertPlainText(data.toString());
        });

        return child;

    }


    // run shell command and print stdin & stdout in text window
    // return child process pid or -1 if an error occurs
    // don't show nvcc command result
    run_command(command: string, params?: string[], dir?: string) {
        let array_params = [];

        if (params == undefined) {
            this.text.insertPlainText('\n$ ' + command + '\n');
        } else {
            let res = '';
            for (let i = 0; i < params.length; i++) {
                if (params[i] != '') {
                    res = res.concat(params[i], ' ');
                    array_params.push(params[i]);
                }
            }
            if (command != 'nvcc')
                this.text.insertPlainText('\n$ ' + command + ' ' + res + '\n');
        }

        process.env.EZ_PATH = '/usr/local/easena/';
        let child = spawnSync(command, array_params, {
            cwd: dir,
            env: process.env
        });

        // execute and print result in console
        if (command != 'nvcc') {
            if (child.output) {
                this.text.insertPlainText(child.output.join(""));
                this.text.insertPlainText("\n");
            }
        }

        if (child.error) {
            // console.log('error detected in run_command');
            return -1;
        } else {
            return child;
        }
    }

    run(cmd: string, plot_size: number, print:boolean, params?: string, dir?: string, rank?: number) {
        let array_params: string[] = [];
        let params_tmp: string[] = [];

        if (params == undefined) {
            this.text.insertPlainText('\n$ ' + cmd + '\n');
        } else {
            params_tmp = params.split(' ');
            this.text.insertPlainText('\n$ ' + cmd + ' ' + params + '\n');
        }

        this.text.insertPlainText('\n');

        array_params.push('-oL');   // flush at the end of the line
        array_params.push('-e0');
        array_params.push(cmd);

        for(let i = 0; i < params_tmp.length; i++)
            array_params.push(params_tmp[i]);

        process.env.EZ_PATH = '/usr/local/easena/';

        let child = spawn('stdbuf', array_params, {     // stdbuf allows reducing buffer size to line size (live terminal)
            cwd: dir,
            env: process.env,
            stdio: 'pipe'
        });

        child.on('error', function (err: string) {
            let t = err.toString().split(' ');
            if (t[t.length - 1] === 'ENOENT') {
                let message = '\nExecution impossible try to recompile';
                if (rank !== undefined) {
                    new Win_alert(err + message, 'Process ' + rank);
                } else {
                    new Win_alert(err + message);
                }
            } else {
                new Win_alert(err + '');
            }

            running_proc.length = 0;
            running_plot.length = 0;

            return -1;
        });

        child.stderr.on('data', (data: string) => {
            console.log(data.toString());
            this.text.insertPlainText(data.toString());
        });

        child.stdout.on('data', (data: string) => {
            if (print === true){
                console.log(data.toString());
                this.text.insertPlainText(data.toString());
            }

            run_obj.running_label.show();
            plot_generation.parser(data);

            if (run_obj.island_model) {
                if(rank){
                    let array = String(data).split('\n');

                    for (let i = 0; i < array.length; i++) {
                        if (i > 3) {
                            array[i] = array[i].split('          ').join('   ');
                            array[i] = array[i].split('\t').join('   ');
                        }
                        if (run_obj.run_results[rank - 1]) {
                            run_obj.run_results[rank - 1] = run_obj.run_results[rank - 1].concat('\n' + array[i]);
                        } else {
                            run_obj.run_results[rank - 1] = array[i];
                        }
                    }
                }

            } else if (run_obj.plot_type === '2D') {
                
                if (rank) {
                    let array = String(data).split('\n');
                    for (let i = 0; i < array.length; i++) {
                        if (i > 3) {
                            array[i] = array[i].split('          ').join('   ');
                            array[i] = array[i].split('\t').join('   ');
                        }
                        if (run_obj.run_results[rank - 1]) {
                            run_obj.run_results[rank - 1] = run_obj.run_results[rank - 1].concat('\n' + array[i]);
                        } else {
                            run_obj.run_results[rank - 1] = array[i];
                        }
                    }
                }
            } else if (run_obj.plot_type === '3D') {    // used when easea will change objective file names
                let array = String(data).split('\n');
                for (let i = 0; i < array.length - 2; i++) {
                    if (array[i].startsWith('Total execution time') ||
                        array[i].startsWith('Quality Metrics') ||
                        array[i].startsWith('HyperVolume') ||
                        array[i].startsWith('Generational distance') ||
                        array[i].startsWith('Inverted generational distance')
                    ) {

                        array[i] = array[i].split('EASEA LOG [INFO]: ').join('');
                        if (run_obj.run_results[0]) {
                            run_obj.run_results[0] = run_obj.run_results[0].concat('\n' + array[i]);
                        } else {
                            run_obj.run_results[0] = array[i];
                        }
                    }
                }
            }
        });

        child.on('exit', (code, signal) => {
            const index = running_proc.indexOf(child, 0);
            if (index > -1) {
                if (running_proc.length === 1) {
                    running_proc.pop();
                } else {
                    running_proc.splice(index, 1);
                }
            }

            if(signal?.toString() === 'SIGSEGV'){
                console.log("child process " + rank + " was terminated due to a segfault");
                this.text.insertPlainText("\n!! Batch interrupted due to a segfault !!\n");
                this.text.setEnabled(true);
                return;
            }

            if (code !== 0) {
                this.text.insertPlainText('\n!! Batch interrupted ');
                if(signal){
                    console.log('child process ' + rank + ' interrupted due to signal : ' + signal);
                    this.text.insertPlainText('(' + signal + ' received) ');
                } else {
                    console.log('child process ' + rank + ' interrupted');
                }
                this.text.insertPlainText("!!\n");
                this.text.setEnabled(true);
                return;
            }

            console.log("child process " + rank + " terminated with code " + code);

            if(run_obj.island_model && run_obj.island_obj.local){
                run_obj.remaining_proc++;
                run_obj.progress_bar.setValue(((run_obj.remaining_proc * 100) / run_obj.total_runs));
                run_obj.finished_label.setText('Completed Runs : ' + run_obj.remaining_proc + '/' + run_obj.total_runs);
            } else {
                run_obj.progress_bar.setValue(run_obj.progress_bar.value() + (100 / (2 * run_obj.batch_size)));

                run_obj.running_label.setText('Plotting results ...');
                run_obj.running_label.show();
                // util.fix_csv();  // file correction (not used)
                plot_obj.update_plot('/tmp/plotting/fig.svg', plot_size, run_obj.plot_type, run_obj.dir_path + '/objectives', '', '', '', '');
                run_obj.finished_label.setText('Completed Runs : ' + (run_obj.batch_size - running_proc.length) + '/' + run_obj.batch_size);
            }

            this.text.setEnabled(true);

        })

        running_proc.push(child);

        return child;
    }
}