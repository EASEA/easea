/**
 * @author Clément Ligneul <clement.ligneul@etu.unistra.fr>
 */

import { QGridLayout, QFileDialog, QDialog, QWidget, QLabel, QPushButton, QBoxLayout, QCheckBox, QScrollArea, QComboBox, WindowType, QLineEdit, FileMode } from '@nodegui/nodegui';
import { Processus } from './processus';
import { Advanced_option_widget } from './advanced_option_widget';
import * as util from './utilities';
import { general_css } from './style';
import { run_obj } from '.';

const MAX_BATCH = 500;

export class Run_options {
    window: QDialog;
    batch_size: number;
    thread_number: number;
    seed: number;
    proc_tab: Processus[];
    batch_widget: QWidget;

    // all options
    compression: number = NaN;
    pop_size: number = NaN;
    nb_elite: number = NaN;
    elite_type: number = NaN;
    nb_gen: number = NaN;
    time_limit: number = NaN;
    select_op: string = 'Tournament';
    select_pressure: number = NaN;
    reduce_final_op: string = 'Tournament';
    reduce_final_pressure: number = NaN;
    optimize_it: number = NaN;
    baldwinism: number = NaN;
    output_file: string = '';
    input_file: string = '';
    print_stats: boolean = true;
    plot_stats: boolean = false;
    generate_csv_file: boolean = false;
    generate_plot_script: boolean = false;
    generate_r_script: boolean = false;
    print_init_pop: boolean = false;
    print_final_pop: boolean = false;
    save_pop: boolean = false;
    start_file: boolean = false;
    fstgpu_param: number = NaN;
    lstgpu_param: number = NaN;
    u1: string = '';
    u2: string = '';
    u3: string = '';
    u4: string = '';
    u5: string = '';
    best_fitness_thresh = NaN;
    worst_fitness_thresh = NaN;
    avg_fitness_thresh = NaN;
    std_dev_thresh = NaN;

    constructor() {
        this.window = new QDialog();
        this.window.setWindowTitle("General Run Options");
        this.window.setWindowFlag(WindowType.CustomizeWindowHint, true);
        this.window.setWindowFlag(WindowType.WindowCloseButtonHint, false);
        this.batch_size = 1;
        this.thread_number = 20;
        this.seed = NaN;
        this.proc_tab = [];

        util.disable_keys('Escape', this.window);

        const global_layout = new QGridLayout();
        const global_widget = new QWidget();
        global_widget.setLayout(global_layout);

        // batch size
        this.batch_widget = new QWidget();
        const batch_layout = new QBoxLayout(2);
        const batch_label = new QLabel();
        batch_label.setText("Batch Size : ");
        batch_label.setFixedSize(80, 20);

        const batch_edit = new QLineEdit();
        batch_edit.setText('1');
        batch_edit.setFixedSize(70, 30);

        batch_layout.addWidget(batch_label);
        batch_layout.addWidget(batch_edit);
        this.batch_widget.setLayout(batch_layout);

        // thread number
        const thread_widget = new QWidget();
        const thread_layout = new QBoxLayout(2);
        const thread_label = new QLabel();
        thread_label.setText("Number Of Threads : ");
        thread_label.setFixedSize(150, 20);

        const thread_edit = new QLineEdit();
        thread_edit.setPlaceholderText('20');
        thread_edit.setFixedSize(70, 30);
        thread_edit.addEventListener('textChanged', () => {
            const val = Number(thread_edit.text());

            // if (!isNaN(val))
            this.thread_number = val;

            if (thread_edit.text() === '')
                this.thread_number = 20;
        });

        thread_layout.addWidget(thread_label);
        thread_layout.addWidget(thread_edit);

        thread_widget.setLayout(thread_layout);

        // first seed
        const seed_layout = new QBoxLayout(2);
        const seed_widget = new QWidget();
        const seed_label = new QLabel();
        const seed_text = new QLineEdit();

        seed_text.setPlaceholderText('Default if empty');
        seed_text.setFixedSize(130, 30);

        seed_text.addEventListener('textChanged', () => {
            const seed = Number(seed_text.text());

            if (!isNaN(seed))
                this.seed = seed;

            if (seed_text.text() === '')
                this.seed = NaN;
        });

        seed_label.setText('First Seed : ');
        seed_label.setFixedSize(90, 30);

        seed_layout.addWidget(seed_label);
        seed_layout.addWidget(seed_text);

        seed_widget.setLayout(seed_layout);

        // advanced options
        const check_plot = new QCheckBox();
        check_plot.setText('Plot Stats');
        check_plot.addEventListener('stateChanged', () => {
            this.plot_stats = check_plot.isChecked();
        });

        const compression_arg = new Advanced_option_widget('Compression :', 2);
        compression_arg.text_edit.addEventListener('textChanged', () => {
            let text = compression_arg.text_edit.text();
            let val = Number(text);
            if (text === '') {
                this.compression = NaN;
                return;
            }

            if (!isNaN(val))
                this.compression = val;
        });

        const popu_size = new Advanced_option_widget('Population Size :', 2);
        popu_size.text_edit.addEventListener('textChanged', () => {
            let text = popu_size.text_edit.text();
            let val = Number(text);
            if (text === '') {
                this.pop_size = NaN;
                return;
            }

            if (!isNaN(val))
                this.pop_size = val;
        });

        const elite_layout = new QBoxLayout(2);
        const elite_type_widget = new QWidget();
        const elite_type_label = new QLabel();
        elite_type_label.setText('Elite Type :');
        elite_type_label.setFixedSize(150, 30);
        elite_type_widget.setLayout(elite_layout);
        const combo_elite_type = new QComboBox();
        combo_elite_type.setFixedSize(100, 30);
        combo_elite_type.addItem(undefined, 'Default');
        combo_elite_type.addItem(undefined, 'Strong');
        combo_elite_type.addItem(undefined, 'Weak');
        combo_elite_type.addEventListener('currentTextChanged', (val) => {
            if (val === 'Default') {
                this.elite_type = NaN;
            } else if (val === 'Strong') {
                this.elite_type = 1;
            } else {
                this.elite_type = 0;
            }
        });

        const nb_elite = new Advanced_option_widget('Elite Size :', 2, '0');
        nb_elite.text_edit.addEventListener('textChanged', () => {
            let text = nb_elite.text_edit.text();
            let val = Number(text);

            if (text === '') {
                this.nb_elite = NaN;
                return;
            }

            if (!isNaN(val))
                this.nb_elite = val;
        });

        elite_layout.addWidget(elite_type_label);
        elite_layout.addWidget(combo_elite_type);

        const nb_gen = new Advanced_option_widget('Nb Generations :', 2);
        nb_gen.text_edit.addEventListener('textChanged', () => {
            let text = nb_gen.text_edit.text();
            if (text === '') {
                this.nb_gen = NaN;
                return;
            }

            let val = Number(text);
            if (!isNaN(val))
                this.nb_gen = val;
        });

        const time_limit = new Advanced_option_widget('Time Limit (in secondes) :', 2);
        time_limit.text_edit.addEventListener('textChanged', () => {
            let text = time_limit.text_edit.text();
            let val = Number(text);
            if (text === '') {
                this.time_limit = NaN;
                return;
            }

            if (!isNaN(val))
                this.time_limit = val;
        });

        // selection operator
        const select_op_widget = new QWidget();
        const select_op_layout = new QBoxLayout(2);
        const select_op_label = new QLabel();
        select_op_label.setText('Selection Operator :');
        select_op_label.setFixedSize(155, 30);
        select_op_widget.setLayout(select_op_layout);

        const combo_select_op = new QComboBox();
        select_op_layout.addWidget(select_op_label);
        select_op_layout.addWidget(combo_select_op);
        combo_select_op.setFixedSize(120, 30);
        combo_select_op.addItem(undefined, 'Tournament');
        combo_select_op.addItem(undefined, 'Deterministic');
        combo_select_op.addItem(undefined, 'Roulette');
        combo_select_op.addItem(undefined, 'Random');
        combo_select_op.addEventListener('currentTextChanged', (val) => {
            if (val === 'Tournament' || val === '') {
                select_pressure.widget.setEnabled(true);
            } else if (val === 'Roulette') {
                // evaluator goal = maximize
                select_pressure.widget.setEnabled(false);
            } else {
                select_pressure.widget.setEnabled(false);
            }

            this.select_op = val;
        });
        select_op_layout.addWidget(select_op_label);
        select_op_layout.addWidget(combo_select_op);

        const select_pressure = new Advanced_option_widget('Selection Pressure :', 2, '2.0');
        select_pressure.text_edit.addEventListener('textChanged', () => {
            var text = select_pressure.text_edit.text();
            var val = Number(text);
            if (text === '') {
                this.select_pressure = NaN;
                return;
            }

            if (!isNaN(val))
                this.select_pressure = val;
        });

        // reduce final operator
        const reduce_final_op_widget = new QWidget();
        const reduce_final_op_layout = new QBoxLayout(2);
        const reduce_final_op_label = new QLabel();
        reduce_final_op_label.setText('Reduce Final Operator :');
        reduce_final_op_label.setFixedSize(155, 30);
        reduce_final_op_widget.setLayout(reduce_final_op_layout);

        const combo_reduce_final_op = new QComboBox();
        reduce_final_op_layout.addWidget(reduce_final_op_label);
        reduce_final_op_layout.addWidget(combo_reduce_final_op);
        combo_reduce_final_op.setFixedSize(120, 30);
        combo_reduce_final_op.addItem(undefined, 'Tournament');
        combo_reduce_final_op.addItem(undefined, 'Deterministic');
        combo_reduce_final_op.addItem(undefined, 'Roulette');
        combo_reduce_final_op.addItem(undefined, 'Random');
        combo_reduce_final_op.addEventListener('currentTextChanged', (val) => {
            if (val === 'Tournament' || val === '') {
                reduce_final_pressure.widget.setEnabled(true);
            } else if (val === 'Roulette') {
                // evaluator goal = maximize
                reduce_final_pressure.widget.setEnabled(false);
            } else {
                reduce_final_pressure.widget.setEnabled(false);
            }

            this.reduce_final_op = val;
        });
        reduce_final_op_layout.addWidget(reduce_final_op_label);
        reduce_final_op_layout.addWidget(combo_reduce_final_op);

        const reduce_final_pressure = new Advanced_option_widget('Final Reduce Pressure :', 2, '2.0');
        reduce_final_pressure.text_edit.addEventListener('textChanged', () => {
            var text = reduce_final_pressure.text_edit.text();
            var val = Number(text);
            if (text === '') {
                this.reduce_final_pressure = NaN;
                return;
            }
            if (!isNaN(val))
                this.reduce_final_pressure = val;
        });

        const opti_it = new Advanced_option_widget('Optimize Iterations :', 2, '100');
        opti_it.text_edit.addEventListener('textChanged', () => {
            var text = opti_it.text_edit.text();
            var val = Number(text);
            if (text === '') {
                this.optimize_it = NaN;
                return;
            }
            if (!isNaN(val))
                this.optimize_it = val;
        });

        const baldwinism = new Advanced_option_widget('Baldwinism :', 2, '0');
        baldwinism.text_edit.addEventListener('textChanged', () => {
            var text = opti_it.text_edit.text();
            var val = Number(text);
            if (text === '') {
                this.optimize_it = NaN;
                return;
            }
            if (!isNaN(val))
                this.optimize_it = val;
        });

        const output_filename = new Advanced_option_widget('Ouput File :', 2, 'None');
        output_filename.text_edit.addEventListener('textChanged', () => {
            this.output_file = output_filename.text_edit.text();
        });

        const input_file_widget = new QWidget();
        const input_file_layout = new QBoxLayout(2);
        input_file_widget.setLayout(input_file_layout);
        const input_file_btn = new QPushButton();
        const file_loaded = new QLabel();
        file_loaded.setText('Initial Population :');
        input_file_btn.setText("Load a file");
        input_file_btn.setFixedSize(100, 25);
        input_file_btn.addEventListener('clicked', () => {
            const fileDialog = new QFileDialog();
            fileDialog.setFileMode(FileMode.ExistingFile);

            if (fileDialog.exec())
                this.input_file = fileDialog.selectedFiles().toString();

            file_loaded.setText('Initial Population :\n' + util.get_file_name(this.input_file));
        });

        input_file_layout.addWidget(file_loaded);
        input_file_layout.addWidget(input_file_btn);

        const print_stats = new QCheckBox();
        print_stats.setText('Print Stats');
        print_stats.setChecked(true);
        print_stats.addEventListener('stateChanged', () => {
            this.print_stats = print_stats.isChecked();
        });

        const csv_file = new QCheckBox();
        csv_file.setText('Generate CSV File');
        csv_file.addEventListener('stateChanged', () => {
            this.generate_csv_file = csv_file.isChecked();
        });

        const plot_script = new QCheckBox();
        plot_script.setText('Generate Plot Script');
        plot_script.addEventListener('stateChanged', () => {
            this.generate_plot_script = plot_script.isChecked();
        });

        const r_script = new QCheckBox();
        r_script.setText('Generate R Script');
        r_script.addEventListener('stateChanged', () => {
            this.generate_r_script = r_script.isChecked();
        });

        const print_init_pop = new QCheckBox();
        print_init_pop.setText('Print Initial Population');
        print_init_pop.addEventListener('stateChanged', () => {
            this.print_init_pop = print_init_pop.isChecked();
        });

        const print_final_pop = new QCheckBox();
        print_final_pop.setText('Print Final Population');
        print_final_pop.addEventListener('stateChanged', () => {
            this.print_final_pop = print_final_pop.isChecked();
        });

        const save_pop = new QCheckBox();
        save_pop.setText('Save Population');
        save_pop.addEventListener('stateChanged', () => {
            this.save_pop = save_pop.isChecked();
        });

        const start_from_file = new QCheckBox();
        start_from_file.setText('Start From File (.pop)');
        start_from_file.addEventListener('stateChanged', () => {
            this.start_file = start_from_file.isChecked();
        });

        const fstgpu = new Advanced_option_widget('Number of the first \nGPU used for computation :', 2);
        fstgpu.text_edit.addEventListener('textChanged', () => {
            var text = fstgpu.text_edit.text();
            var val = Number(text);
            if (text === '') {
                this.fstgpu_param = NaN;
                return;
            }
            if (!isNaN(val))
                this.fstgpu_param = val;
        });

        const lstgpu = new Advanced_option_widget('Number of the first \nGPU NOT used for computation :', 2);
        lstgpu.text_edit.addEventListener('textChanged', () => {
            var text = lstgpu.text_edit.text();
            var val = Number(text);
            if (text === '') {
                this.lstgpu_param = NaN;
                return;
            }
            if (!isNaN(val))
                this.lstgpu_param = val;
        });

        // user params
        const global_user_param_layout = new QBoxLayout(2);

        const user_param_label = new QLabel();
        user_param_label.setText('User parameters :');

        const user_param_scroll = new QScrollArea();
        const user_param_layout = new QBoxLayout(2);
        const user_param_widget = new QWidget();
        user_param_widget.setLayout(user_param_layout);

        var u1 = new Advanced_option_widget('User parameter 1', 2);
        user_param_layout.addWidget(u1.widget);
        u1.text_edit.addEventListener('textChanged', () => {
            this.u1 = u1.text_edit.text();
        });

        var u2 = new Advanced_option_widget('User parameter 2', 2);
        user_param_layout.addWidget(u2.widget);
        u2.text_edit.addEventListener('textChanged', () => {
            this.u2 = u2.text_edit.text();
        });

        var u3 = new Advanced_option_widget('User parameter 3', 2);
        user_param_layout.addWidget(u3.widget);
        u3.text_edit.addEventListener('textChanged', () => {
            this.u3 = u3.text_edit.text();
        });

        var u4 = new Advanced_option_widget('User parameter 4', 2);
        user_param_layout.addWidget(u4.widget);
        u4.text_edit.addEventListener('textChanged', () => {
            this.u4 = u4.text_edit.text();
        });

        var u5 = new Advanced_option_widget('User parameter 5', 2);
        user_param_layout.addWidget(u5.widget);
        u5.text_edit.addEventListener('textChanged', () => {
            this.u5 = u5.text_edit.text();
        });

        user_param_scroll.setWidget(user_param_widget);
        user_param_scroll.setFixedSize(180, 140);

        global_user_param_layout.addWidget(user_param_label);
        global_user_param_layout.addWidget(user_param_scroll);

        const user_param_global_widget = new QWidget();
        user_param_global_widget.setLayout(global_user_param_layout);

        // processes/seeds table
        const global_seed_widget = new QWidget();
        const global_seed_layout = new QBoxLayout(2);
        global_seed_widget.setLayout(global_seed_layout);

        const seed_text_label = new QLabel();
        seed_text_label.setText('Seeds by run :');
        global_seed_layout.addWidget(seed_text_label);

        const proc_seed_area = new QScrollArea();
        const proc_seed_widget = new QWidget();
        const proc_seed_layout = new QBoxLayout(2);

        proc_seed_widget.setLayout(proc_seed_layout);
        proc_seed_widget.setMinimumSize(180, 140);

        batch_edit.addEventListener('textChanged', () => {
            var batch_size = Number(batch_edit.text());

            if (isNaN(batch_size) || batch_size > MAX_BATCH || batch_size <= 0)
                return;

            for (let i = 0; i < MAX_BATCH; i++)
                (i < batch_size) ? this.proc_tab[i].enable() : this.proc_tab[i].disable();
        });

        this.proc_tab.push(new Processus(1));
        proc_seed_layout.addWidget(this.proc_tab[0].proc_widget);

        for (let i = 1; i < 500; i++) {
            this.proc_tab.push(new Processus(i + 1));
            proc_seed_layout.addWidget(this.proc_tab[i].proc_widget);
            this.proc_tab[i].disable();
        }

        proc_seed_area.setWidget(proc_seed_widget);
        proc_seed_area.setFixedSize(270, 140);

        global_seed_layout.addWidget(proc_seed_area);

        // close and save buttons
        const save_btn = new QPushButton();
        save_btn.setText('Save');
        save_btn.setFixedSize(100, 25);
        save_btn.addEventListener('clicked', () => {
            var errors = [];
            var ok = 1;
            const input = Number(batch_edit.text());
            const seed = Number(seed_text.text());

            if (!Number.isInteger(input) || input <= 0 || input > MAX_BATCH || isNaN(input)) {
                ok = 0;
                errors.push('Batch Size');
            } else {
                this.batch_size = input;

                for (let i = 0; i < this.batch_size; i++) {
                    let val = Number(this.proc_tab[i].seed_text.text());
                    if (isNaN(val)) {
                        ok = 0;
                        errors.push('Seed for run ' + (i + 1));
                    }
                }
            }

            if (!Number.isInteger(this.thread_number) || this.thread_number <= 0 || this.thread_number > 30 || isNaN(this.thread_number)) {
                ok = 0;
                errors.push('Thread Number');
            }

            if (!Number.isInteger(seed) || isNaN(seed)) {
                ok = 0;
                errors.push('First Seed');
            }

            if (isNaN(Number(compression_arg.text_edit.text()))) {
                ok = 0;
                errors.push('Compression');
            }

            if (isNaN(Number(nb_elite.text_edit.text()))) {
                ok = 0;
                errors.push('Elite Size');
            }

            if (isNaN(Number(popu_size.text_edit.text()))) {
                ok = 0;
                errors.push('Population Size');
            }

            var gen = Number(nb_gen.text_edit.text());
            if (isNaN(gen) || (gen <= 0 && nb_gen.text_edit.text() !== '')) {
                ok = 0;
                errors.push('Nb Generations');
            }

            if (isNaN(Number(time_limit.text_edit.text()))) {
                ok = 0;
                errors.push('Time Limit');
            }

            var pressure = Number(select_pressure.text_edit.text());
            if (isNaN(pressure)) {
                ok = 0;
                errors.push('Selection Pressure');
            } else if (pressure < 0.5 && pressure != 0) {
                ok = 0;
                errors.push('Selection Pressure : should be in [ 0.5 ; 1 [ U [ 2 ; +\u221e [');
            } else if (pressure >= 1 && pressure < 2) {
                ok = 0;
                errors.push('Selection Pressure : should be in [ 0.5 ; 1 [ U [ 2 ; +\u221e [');
            } else if (pressure >= 2 && pressure % 1 !== 0) {
                ok = 0;
                errors.push('Selection Pressure : should be an integer if \u2265 2');
            }

            var final_pressure = Number(reduce_final_pressure.text_edit.text());
            if (isNaN(final_pressure)) {
                ok = 0;
                errors.push('Reduce Final Pressure');
            } else if (final_pressure < 0.5 && final_pressure != 0) {
                ok = 0;
                errors.push('Reduce Final Pressure : should be in [ 0.5 ; 1 [ U [ 2 ; +\u221e [');
            } else if (final_pressure >= 1 && final_pressure < 2) {
                ok = 0;
                errors.push('Reduce Final Pressure : should be in [ 0.5 ; 1 [ U [ 2 ; +\u221e [');
            } else if (final_pressure >= 2 && final_pressure % 1 !== 0) {
                ok = 0;
                errors.push('Reduce Final Pressure : should be an integer if \u2265 2');
            }

            if (isNaN(Number(opti_it.text_edit.text()))) {
                ok = 0;
                errors.push('Optimize Iteration');
            }

            if (isNaN(Number(baldwinism.text_edit.text()))) {
                ok = 0;
                errors.push('Baldwinism');
            }

            if (isNaN(Number(fstgpu.text_edit.text()))) {
                ok = 0;
                errors.push('First GPU used for computation');
            }

            if (isNaN(Number(lstgpu.text_edit.text()))) {
                ok = 0;
                errors.push('First GPU NOT used for computation');
            }

            if (run_obj.plot_type === '3D' && this.batch_size != 1) {
                ok = 0;
                errors.push('Batch size : batch mode is not yet available for multi-objective programs');
            }

            if (ok) {
                this.window.close();
            } else {
                util.print_errors(errors);
            }
        });

        const reset_btn = new QPushButton();
        reset_btn.setText('Reset');
        reset_btn.setFixedSize(100, 25);
        reset_btn.addEventListener('clicked', () => {
            batch_edit.setText('1');
            this.batch_size = 1;
            thread_edit.clear();

            seed_text.clear();
            this.seed = NaN;

            for (let i = 0; i < this.proc_tab.length; i++) {
                this.proc_tab[i].seed_value = NaN;
                this.proc_tab[i].seed_text.clear();
            }

            check_plot.setChecked(false);

            compression_arg.text_edit.clear();
            popu_size.text_edit.clear();
            combo_elite_type.setCurrentIndex(0);
            nb_elite.text_edit.clear();
            nb_gen.text_edit.clear();
            time_limit.text_edit.clear();
            combo_select_op.setCurrentIndex(0);
            select_pressure.text_edit.clear();
            combo_reduce_final_op.setCurrentIndex(0);
            reduce_final_pressure.text_edit.clear();
            opti_it.text_edit.clear();
            baldwinism.text_edit.clear();
            output_filename.text_edit.clear();
            file_loaded.setText('Initial Population :');
            print_stats.setChecked(true);
            csv_file.setChecked(false);
            plot_script.setChecked(false);
            r_script.setChecked(false);
            print_init_pop.setChecked(false);
            print_final_pop.setChecked(false);
            save_pop.setChecked(false);
            start_from_file.setChecked(false);
            fstgpu.text_edit.clear();
            lstgpu.text_edit.clear();
            u1.text_edit.clear();
            u2.text_edit.clear();
            u3.text_edit.clear();
            u4.text_edit.clear();
            u5.text_edit.clear();
        });

        // regroup buttons
        const widget_btn = new QWidget();
        const layout_btn = new QBoxLayout(0);

        widget_btn.setLayout(layout_btn);
        layout_btn.addWidget(save_btn);
        layout_btn.addWidget(reset_btn);

        // display
        global_layout.addWidget(check_plot, 0, 0);
        global_layout.addWidget(csv_file, 0, 1);
        // global_layout.addWidget(print_stats,0,2);
        global_layout.addWidget(print_init_pop, 0, 2);
        global_layout.addWidget(print_final_pop, 0, 3);
        global_layout.addWidget(start_from_file, 0, 4);

        global_layout.addWidget(r_script, 1, 0);
        global_layout.addWidget(plot_script, 1, 1);
        global_layout.addWidget(save_pop, 1, 2);

        global_layout.addWidget(popu_size.widget, 2, 0);
        global_layout.addWidget(nb_gen.widget, 2, 1);
        global_layout.addWidget(time_limit.widget, 2, 2);
        global_layout.addWidget(elite_type_widget, 2, 3);
        global_layout.addWidget(nb_elite.widget, 2, 4);

        global_layout.addWidget(select_op_widget, 3, 0);
        global_layout.addWidget(select_pressure.widget, 3, 1);
        global_layout.addWidget(reduce_final_op_widget, 3, 2);
        global_layout.addWidget(reduce_final_pressure.widget, 3, 3);
        global_layout.addWidget(baldwinism.widget, 3, 4);

        global_layout.addWidget(widget_btn, 7, 2);

        global_layout.addWidget(fstgpu.widget, 4, 0);
        global_layout.addWidget(lstgpu.widget, 4, 1);
        global_layout.addWidget(input_file_widget, 4, 2);
        global_layout.addWidget(output_filename.widget, 4, 3);
        global_layout.addWidget(opti_it.widget, 4, 4);

        global_layout.addWidget(compression_arg.widget, 5, 0);
        global_layout.addWidget(this.batch_widget, 5, 1);
        global_layout.addWidget(seed_widget, 5, 2);
        global_layout.addWidget(thread_widget, 5, 3);
        
        global_layout.addWidget(global_seed_widget, 6, 1);
        global_layout.addWidget(user_param_global_widget, 6, 3);

        this.window.setLayout(global_layout);
        this.window.adjustSize();
        this.window.setFixedSize(this.window.size().width(), this.window.size().height());
        this.window.setStyleSheet(general_css);
    }

    execution() {
        this.window.exec();
        return this;
    }
}