/**
 * @author Clément Ligneul <clement.ligneul@etu.unistra.fr>
 */

import { AlignmentFlag, Orientation, WindowType, QDialog, QGridLayout, QLabel, QPlainTextEdit, QPushButton, QSlider, QWidget, TickPosition, QBoxLayout, QTabWidget, QIcon, QScrollArea, QTableWidget, QTableWidgetItem, ScrollBarPolicy, ItemFlag, SizeConstraint, SortOrder } from "@nodegui/nodegui";
import { general_css } from "./style";
import { plot_obj, run_obj } from ".";
import * as util from "./utilities"
import * as fs from 'fs';
import { Advanced_option_widget } from "./advanced_option_widget";
import { exit } from "process";


export class Results_win {
    window: QDialog;
    window_layout: QBoxLayout;

    // results
    results_widget: QWidget;
    results_layout: QBoxLayout;
    slider: QSlider;
    console: QPlainTextEdit;
    run_label: QLabel;

    // thresholds
    threshold_widget : QWidget;
    threshold_layout : QBoxLayout;
    best_fit_thresh  : number = NaN;
    worst_fit_thresh : number = NaN;
    avg_fit_thresh   : number = NaN;
    std_dev_thresh   : number = NaN;
    // table 
    best_fit_left    : QTableWidget = new QTableWidget(0,0);
    best_fit_right   : QTableWidget = new QTableWidget(0,0);
    worst_fit_left   : QTableWidget = new QTableWidget(0,0);
    worst_fit_right  : QTableWidget = new QTableWidget(0,0);
    avg_fit_left     : QTableWidget = new QTableWidget(0,0);
    avg_fit_right    : QTableWidget = new QTableWidget(0,0);
    std_dev_left     : QTableWidget = new QTableWidget(0,0);
    std_dev_right    : QTableWidget = new QTableWidget(0,0);

    constructor() {
        this.window = new QDialog();
        this.window.setWindowTitle('Batch Results')
        this.window.setWindowFlag(WindowType.CustomizeWindowHint, true);
        this.window.setWindowFlag(WindowType.WindowCloseButtonHint, false);
        this.window.setStyleSheet(general_css);

        this.window_layout = new QBoxLayout(2);

        // results tab
        this.results_layout = new QBoxLayout(2);
        this.results_widget = new QWidget();
        this.results_widget.setLayout(this.results_layout);
        this.results_layout.setSizeConstraint(SizeConstraint.SetFixedSize);

        this.slider = new QSlider();

        this.console = new QPlainTextEdit();
        this.console.setFixedSize(1000, 450);
        this.console.setReadOnly(true);

        // thresholds tab
        this.threshold_widget = new QWidget();
        // this.threshold_layout = new QGridLayout();
        this.threshold_layout = new QBoxLayout(2);
        this.threshold_widget.setLayout(this.threshold_layout);

        this.run_label = new QLabel();
    }

    generate() {
        /************************ Results ************************/
        let batch_size = run_obj.runned_proc;

        // batch average
        let average_bfit_label = new QLabel();
        let average_bfit_value: number = 0;
        for (let i = 0; i < run_obj.runned_proc; i++) {
            average_bfit_value += util.get_best_fitness(run_obj.run_results[i]);
        }

        average_bfit_value = Math.floor(average_bfit_value * 100000) / 100000;
        average_bfit_value = average_bfit_value / run_obj.runned_proc;
        average_bfit_value = Number(average_bfit_value.toFixed(5));
        average_bfit_label.setText('Best fitness average : ' + average_bfit_value);
        average_bfit_label.setAlignment(AlignmentFlag.AlignLeft);

        // batch size
        let bsize_label = new QLabel();
        bsize_label.setText('Batch Size : ' + run_obj.batch_size);

        // run number label
        const run_label_sep = this.generate_separator(this.run_label.text());

        if(run_obj.island_model){
            this.run_label.setText('Results for island ' + (this.slider.value() + 1) + ' (run n°1)');
        } else {
            this.run_label.setText('Results for run ' + (this.slider.value() + 1));
        }

        run_label_sep.setText(this.run_label.text());

        this.run_label.setAlignment(AlignmentFlag.AlignCenter);
        this.run_label.setStyleSheet('font-size: 16pt;');

        // buttons
        const btn_widget = new QWidget();
        const btn_layout = new QGridLayout()

        btn_widget.setLayout(btn_layout);
        const btn_close = new QPushButton();
        btn_close.setFixedSize(150, 30);
        btn_close.setText('Close');
        btn_close.addEventListener('clicked', () => {
            this.window.close();
        });

        btn_layout.addWidget(btn_close, 0, 1);

        // slider
        const slider_widget = new QWidget();
        const slider_layout = new QGridLayout();
        slider_widget.setLayout(slider_layout);

        this.slider.setRange(1, batch_size);
        this.slider.setOrientation(Orientation.Horizontal);
        this.slider.setPageStep(1);
        this.slider.setTickPosition(TickPosition.TicksBelow);
        this.slider.setTickInterval(1);
        this.slider.addEventListener('valueChanged', () => {
            if(run_obj.island_model){
                this.run_label.setText('Results for island ' + (this.slider.value()) + ' (run n°?)');
            } else {
                this.run_label.setText('Results for run ' + (this.slider.value()));
            }

            run_label_sep.setText(this.run_label.text());

            // write results
            this.console.setPlainText('');
            this.console.insertPlainText(run_obj.run_results[this.slider.value() - 1]);

        });

        slider_layout.addWidget(this.slider, 0, 0, 1, batch_size);

        for (let i = 0; i < batch_size; i++) {
            let run_rank = new QLabel();
            run_rank.setText((i + 1).toString());
            run_rank.setFixedSize(16, 30);

            slider_layout.addWidget(run_rank, 1, i);
        }

        // display results
        this.results_layout.addWidget(run_label_sep, undefined, AlignmentFlag.AlignCenter | AlignmentFlag.AlignTop);
        this.results_layout.addWidget(average_bfit_label);
        this.results_layout.addWidget(bsize_label);

        if (batch_size > 1)
            this.results_layout.addWidget(slider_widget);

        this.results_layout.addWidget(this.console);

        /************************ Thresholds ************************/
        const thresh_sep = this.generate_separator("Thresholds");

        // threshold choice
        const thresh_choice_widget = new QWidget;
        const thresh_choice_layout = new QGridLayout;
        thresh_choice_widget.setLayout(thresh_choice_layout);
        thresh_choice_layout.setColumnStretch(1,20);
        thresh_choice_layout.setColumnStretch(2,20);

        // best fitness
        const best_fitness_thresh = new Advanced_option_widget('Best Fitness', 2, 'None', true, true);
        best_fitness_thresh.text_edit_double_spin.addEventListener('valueChanged', text =>{
            this.best_fit_thresh = text;
        });

        // worst fitness
        const worst_fitness_thresh = new Advanced_option_widget('Worst Fitness', 2, 'None', true, true);
        worst_fitness_thresh.text_edit_double_spin.addEventListener('valueChanged', text =>{
            this.worst_fit_thresh = text;
        });

        // average fitness
        const avg_fitness_thresh = new Advanced_option_widget('Average Fitness', 2, 'None', true, true);
        avg_fitness_thresh.text_edit_double_spin.addEventListener('valueChanged', text =>{
            this.avg_fit_thresh = text;
        });

        // std dev fitness
        const std_dev_thresh = new Advanced_option_widget('Standard Deviation', 2, 'None', true, true);
        std_dev_thresh.text_edit_double_spin.addEventListener('valueChanged', text =>{
            this.std_dev_thresh = text;
        });

        // go button
        const go_btn = new QPushButton;
        go_btn.setText('Start');
        go_btn.setFixedSize(115,30);
        go_btn.addEventListener('clicked', () =>{
            this.find_threshold();
        });

        // reset button
        const reset_btn = new QPushButton;
        reset_btn.setText('Reset');
        reset_btn.setFixedSize(115,30);
        reset_btn.addEventListener('clicked', () =>{
            best_fitness_thresh.text_edit_double_spin.setValue(0);
            worst_fitness_thresh.text_edit_double_spin.setValue(0);
            avg_fitness_thresh.text_edit_double_spin.setValue(0);
            std_dev_thresh.text_edit_double_spin.setValue(0);
        });

        // threshold choice display
        thresh_choice_layout.addWidget(best_fitness_thresh.widget,0,0);
        thresh_choice_layout.addWidget(worst_fitness_thresh.widget,0,1);
        thresh_choice_layout.addWidget(avg_fitness_thresh.widget,0,2);
        thresh_choice_layout.addWidget(std_dev_thresh.widget,0,3);
        thresh_choice_layout.addWidget(go_btn,1,2);
        thresh_choice_layout.addWidget(reset_btn,1,1);

        // === tables ===
        const table_area = new QScrollArea();
        const table_widget = new QWidget();
        const table_layout = new QGridLayout();
        table_area.setAlignment(AlignmentFlag.AlignCenter | AlignmentFlag.AlignTop);
        table_area.setFixedSize(1000,450);
        table_area.setHorizontalScrollBarPolicy(ScrollBarPolicy.ScrollBarAlwaysOff);

        let tmp: [QWidget, QTableWidget];
        const nb_col = run_obj.batch_size < 10 ? 10 : run_obj.batch_size;

        // best fitness table
        const bfit_widget = new QWidget;
        const bfit_layout = new QBoxLayout(2);
        bfit_widget.setLayout(bfit_layout);

        const best_sep = this.generate_separator('Best Fitness');
        best_sep.setStyleSheet('font-size: 15pt; border-bottom: 0.5px solid; background-color:#efeeee;');

        tmp = this.generate_table_widget('Runs below best fitness threshold', nb_col, 2);
        const best_fit_left_table = tmp[0];
        this.best_fit_left = tmp[1];

        tmp = this.generate_table_widget('Runs above best fitness threshold', nb_col, 3);
        const best_fit_right_table = tmp[0];
        this.best_fit_right = tmp[1]; 
        
        const bfit_table_widget = new QWidget;
        const bfit_table_layout = new QBoxLayout(0);
        bfit_table_widget.setLayout(bfit_table_layout);
        bfit_table_layout.addWidget(best_fit_left_table);
        bfit_table_layout.addWidget(best_fit_right_table);

        bfit_layout.addWidget(best_sep, undefined, AlignmentFlag.AlignCenter | AlignmentFlag.AlignTop);
        bfit_layout.addWidget(bfit_table_widget, undefined, AlignmentFlag.AlignCenter | AlignmentFlag.AlignTop);

        // worst fitness table
        const wfit_widget = new QWidget;
        const wfit_layout = new QBoxLayout(2);
        wfit_widget.setLayout(wfit_layout);
        
        const worst_sep = this.generate_separator('Worst Fitness');
        worst_sep.setStyleSheet('font-size: 15pt; border-bottom: 0.5px solid; background-color:#efeeee;');
        
        tmp = this.generate_table_widget('Runs below worst fitness threshold', nb_col, 2);
        const worst_fit_left_table =  tmp[0];
        this.worst_fit_left = tmp[1];

        tmp = this.generate_table_widget('Runs above worst fitness threshold', nb_col, 3);
        const worst_fit_right_table = tmp[0];
        this.worst_fit_right = tmp[1];
        
        const wfit_table_widget = new QWidget;
        const wfit_table_layout = new QBoxLayout(0);
        wfit_table_widget.setLayout(wfit_table_layout);
        wfit_table_layout.addWidget(worst_fit_left_table);
        wfit_table_layout.addWidget(worst_fit_right_table);

        wfit_layout.addWidget(worst_sep, undefined, AlignmentFlag.AlignCenter | AlignmentFlag.AlignTop);
        wfit_layout.addWidget(wfit_table_widget, undefined, AlignmentFlag.AlignCenter | AlignmentFlag.AlignTop);

        // avg fitness table
        const avgfit_widget = new QWidget;
        const avgfit_layout = new QBoxLayout(2);
        avgfit_widget.setLayout(avgfit_layout);
        
        const avg_sep = this.generate_separator('Average Fitness');
        avg_sep.setStyleSheet('font-size: 15pt; border-bottom: 0.5px solid; background-color:#efeeee;');
        
        tmp = this.generate_table_widget('Runs below average fitness threshold', nb_col, 2);
        const avg_fit_left_table = tmp[0]; 
        this.avg_fit_left = tmp[1];

        tmp = this.generate_table_widget('Runs above average fitness threshold', nb_col, 3);
        const avg_fit_right_table = tmp[0];
        this.avg_fit_right = tmp[1];
        
        const avgfit_table_widget = new QWidget;
        const avgfit_table_layout = new QBoxLayout(0);
        avgfit_table_widget.setLayout(avgfit_table_layout);
        avgfit_table_layout.addWidget(avg_fit_left_table);
        avgfit_table_layout.addWidget(avg_fit_right_table);

        avgfit_layout.addWidget(avg_sep, undefined, AlignmentFlag.AlignCenter | AlignmentFlag.AlignTop);
        avgfit_layout.addWidget(avgfit_table_widget, undefined, AlignmentFlag.AlignCenter | AlignmentFlag.AlignTop);

        // std deviation table
        const std_dev_widget = new QWidget;
        const std_dev_layout = new QBoxLayout(2);
        std_dev_widget.setLayout(std_dev_layout);
        
        const std_dev_sep = this.generate_separator('Standard deviation');
        std_dev_sep.setStyleSheet('font-size: 15pt; border-bottom: 0.5px solid; background-color:#efeeee;');
        
        tmp = this.generate_table_widget('Runs below standard deviation threshold', nb_col, 2);
        const std_dev_left_table = tmp[0]; 
        this.std_dev_left = tmp[1];

        tmp = this.generate_table_widget('Runs above standard deviation threshold', nb_col, 3);
        const std_dev_right_table = tmp[0]; 
        this.std_dev_right = tmp[1];
        
        const std_dev_table_widget = new QWidget;
        const std_dev_table_layout = new QBoxLayout(0);
        std_dev_table_widget.setLayout(std_dev_table_layout);
        std_dev_table_layout.addWidget(std_dev_left_table);
        std_dev_table_layout.addWidget(std_dev_right_table);

        std_dev_layout.addWidget(std_dev_sep, undefined, AlignmentFlag.AlignCenter | AlignmentFlag.AlignTop);
        std_dev_layout.addWidget(std_dev_table_widget, undefined, AlignmentFlag.AlignCenter | AlignmentFlag.AlignTop);
        

        // tables area display
        table_layout.addWidget(bfit_widget,0,0);
        table_layout.addWidget(wfit_widget,1,0);
        table_layout.addWidget(avgfit_widget,2,0);
        table_layout.addWidget(std_dev_widget,3,0);

        table_widget.setLayout(table_layout);
        table_area.setWidget(table_widget);

        // threshold display
        this.threshold_layout.addWidget(thresh_sep, undefined, AlignmentFlag.AlignTop | AlignmentFlag.AlignCenter);
        this.threshold_layout.addWidget(thresh_choice_widget, undefined, AlignmentFlag.AlignCenter | AlignmentFlag.AlignTop);
        this.threshold_layout.addWidget(table_area);
        
        // tabs
        const tabs = new QTabWidget();
        tabs.addTab(this.results_widget, new QIcon(), 'Results');
        // tabs.addTab(this.threshold_widget, new QIcon(), 'Thresholds');
        
        this.window_layout.addWidget(tabs);
        this.window_layout.addWidget(btn_widget);
        this.window.setLayout(this.window_layout);

        this.window.adjustSize();
        this.window.setFixedSize(this.window.size().width(), this.window.size().height());
    }

    execute() {
        // write results
        this.console.insertPlainText(run_obj.run_results[0]);
        this.window.exec();
    }

    generate_separator(text:string) {
        const sep = new QLabel();
        sep.setText(text);
        sep.setAlignment(AlignmentFlag.AlignCenter);
        sep.setFixedSize(500,50);
        sep.setStyleSheet(`
            font-size: 15pt;
            border-bottom: 0.5px solid;
            background-color:#e2e2e2;
        `);

        return sep;
    }

    find_threshold():void {
        let current_row_bfit_left     = 0;
        let current_row_bfit_right    = 0;
        let current_row_wfit_left     = 0;
        let current_row_wfit_right    = 0;
        let current_row_avgfit_left   = 0;
        let current_row_avgfit_right  = 0;
        let current_row_std_dev_left  = 0;
        let current_row_std_dev_right = 0;

        this.reset_tables();
        
        let run_num = 0;
        let file = fs.readdirSync(run_obj.dir_save);
        let nb_gen = isNaN(run_obj.option_obj.nb_gen) ? plot_obj.get_generations(`${run_obj.ez_file_address}`) : run_obj.option_obj.nb_gen;

        for(let filename of file){
            let ok_bfit:boolean = false;
            let ok_wfit:boolean = false;
            let ok_avg_fit:boolean = false;
            let ok_std_dev:boolean = false;

            let text = fs.readFileSync(run_obj.dir_save + filename, 'utf-8');
            let line = text.split(/\r?\n/).map(function(item){ return item.trim(); });

            if(line[0] === `Batch id : ${run_obj.batch_id}`){
                let second = line[2].split(':').map(function(item){ return item.trim(); });
                if(second[0] === 'Run_num'){
                    run_num = Number(second[1]);
                }

                for(let val of line){
                    let col = val.split('\t');

                    if(col.length === 8){
                        const item_gen = new QTableWidgetItem(`${col[0]}`);
                        item_gen.setTextAlignment(AlignmentFlag.AlignCenter);
                        item_gen.setFlags(ItemFlag.ItemIsEditable);

                        const item_bfit = new QTableWidgetItem(`${col[4]}`);
                        item_bfit.setTextAlignment(AlignmentFlag.AlignCenter);
                        item_bfit.setFlags(ItemFlag.ItemIsEditable);

                        const item_wfit = new QTableWidgetItem(`${col[6]}`);
                        item_wfit.setTextAlignment(AlignmentFlag.AlignCenter);
                        item_wfit.setFlags(ItemFlag.ItemIsEditable);

                        const item_avg_fit = new QTableWidgetItem(`${col[5]}`);
                        item_avg_fit.setTextAlignment(AlignmentFlag.AlignCenter);
                        item_avg_fit.setFlags(ItemFlag.ItemIsEditable);

                        const item_std_dev = new QTableWidgetItem(`${col[7]}`);
                        item_std_dev.setTextAlignment(AlignmentFlag.AlignCenter);
                        item_std_dev.setFlags(ItemFlag.ItemIsEditable);
                    
                        // best fitness threshold
                        if(ok_bfit === false){                            
                            const item_run = new QTableWidgetItem(`${run_num}`);
                            item_run.setTextAlignment(AlignmentFlag.AlignCenter);
                            item_run.setFlags(ItemFlag.ItemIsEditable);

                            if(!isNaN(this.best_fit_thresh) && 
                                (Number(col[4]) <= this.best_fit_thresh)){
                                    // console.log(`best_fitness => ${run_num} : (${col[4]} -> ${Number(col[0])})`);
                                    this.best_fit_right.takeItem(current_row_bfit_right, 0);
                                    this.best_fit_right.takeItem(current_row_bfit_right, 1);
                                    this.best_fit_right.takeItem(current_row_bfit_right, 2);

                                    this.best_fit_right.setItem(current_row_bfit_right, 0, item_run);
                                    this.best_fit_right.setItem(current_row_bfit_right, 1, item_bfit);
                                    this.best_fit_right.setItem(current_row_bfit_right, 2, item_gen);
                                    current_row_bfit_right++;
                                    ok_bfit = true;

                                } else if(Number(col[0]) === (nb_gen-1)) {
                                    this.best_fit_left.takeItem(current_row_bfit_left, 0);
                                    this.best_fit_left.takeItem(current_row_bfit_left, 1);
                                    

                                    this.best_fit_left.setItem(current_row_bfit_left, 0, item_run);
                                    this.best_fit_left.setItem(current_row_bfit_left, 1, item_bfit);
                                    current_row_bfit_left++;
                                }
                        }

                        // avg fitness threshold
                        if(ok_avg_fit === false){
                            const item_run = new QTableWidgetItem(`${run_num}`);
                            item_run.setTextAlignment(AlignmentFlag.AlignCenter);
                            item_run.setFlags(ItemFlag.ItemIsEditable);

                            if(!isNaN(this.avg_fit_thresh) && 
                                (Number(col[5]) <= this.avg_fit_thresh)){
                                    // console.log(`avg_fitness => ${run_num} : (${col[5]} -> ${Number(col[0])})`);
                                    this.avg_fit_right.takeItem(current_row_avgfit_right, 0);
                                    this.avg_fit_right.takeItem(current_row_avgfit_right, 1);
                                    this.avg_fit_right.takeItem(current_row_avgfit_right, 2);

                                    this.avg_fit_right.setItem(current_row_avgfit_right, 0, item_run);
                                    this.avg_fit_right.setItem(current_row_avgfit_right, 1, item_avg_fit);
                                    this.avg_fit_right.setItem(current_row_avgfit_right, 2, item_gen);
                                    current_row_avgfit_right++;
                                    ok_avg_fit = true;
                                    
                                } else if(Number(col[0]) === (nb_gen-1)) {
                                    this.avg_fit_left.takeItem(current_row_avgfit_right, 0);
                                    this.avg_fit_left.takeItem(current_row_avgfit_right, 1);

                                    this.avg_fit_left.setItem(current_row_avgfit_left, 0, item_run);
                                    this.avg_fit_left.setItem(current_row_avgfit_left, 1, item_avg_fit);
                                    current_row_avgfit_left++;
                                }
                        }

                        // worst fitness threshold
                        if(ok_wfit === false){
                            const item_run = new QTableWidgetItem(`${run_num}`);
                            item_run.setTextAlignment(AlignmentFlag.AlignCenter);
                            item_run.setFlags(ItemFlag.ItemIsEditable);

                            if(!isNaN(this.worst_fit_thresh) && 
                                (Number(col[6]) <= this.worst_fit_thresh)){
                                    // console.log(`worst_fitness => ${run_num} : (${col[6]} -> ${Number(col[0])})`);
                                    this.worst_fit_right.takeItem(current_row_wfit_right, 0);
                                    this.worst_fit_right.takeItem(current_row_wfit_right, 1);
                                    this.worst_fit_right.takeItem(current_row_wfit_right, 2);

                                    this.worst_fit_right.setItem(current_row_wfit_right, 0, item_run);
                                    this.worst_fit_right.setItem(current_row_wfit_right, 1, item_wfit);
                                    this.worst_fit_right.setItem(current_row_wfit_right, 2, item_gen);
                                    current_row_wfit_right++;
                                    ok_wfit = true;

                                } else if(Number(col[0]) === (nb_gen-1)) {
                                    this.worst_fit_left.takeItem(current_row_wfit_left, 0);
                                    this.worst_fit_left.takeItem(current_row_wfit_left, 1);
                                    
                                    this.worst_fit_left.setItem(current_row_wfit_left, 0, item_run);
                                    this.worst_fit_left.setItem(current_row_wfit_left, 1, item_wfit);
                                    current_row_wfit_left++;
                                }
                        }

                        // std dev threshold
                        if(ok_std_dev === false){
                            const item_run = new QTableWidgetItem(`${run_num}`);
                            item_run.setTextAlignment(AlignmentFlag.AlignCenter);
                            item_run.setFlags(ItemFlag.ItemIsEditable);

                            if(!isNaN(this.std_dev_thresh) && 
                                (Number(col[7]) <= this.std_dev_thresh)){
                                    // console.log(`std_dev => ${run_num} : (${col[7]} -> ${Number(col[0])})`);
                                    this.std_dev_right.takeItem(current_row_std_dev_right, 0);
                                    this.std_dev_right.takeItem(current_row_std_dev_right, 1);
                                    this.std_dev_right.takeItem(current_row_std_dev_right, 2);

                                    this.std_dev_right.setItem(current_row_std_dev_right, 0, item_run);
                                    this.std_dev_right.setItem(current_row_std_dev_right, 1, item_std_dev);
                                    this.std_dev_right.setItem(current_row_std_dev_right, 2, item_gen);
                                    current_row_std_dev_right++;
                                    ok_std_dev = true;

                                } else if(Number(col[0]) === (nb_gen-1)) {
                                    this.std_dev_left.takeItem(current_row_std_dev_left, 0);
                                    this.std_dev_left.takeItem(current_row_std_dev_left, 1);

                                    this.std_dev_left.setItem(current_row_std_dev_left, 0, item_run);
                                    this.std_dev_left.setItem(current_row_std_dev_left, 1, item_std_dev);
                                    current_row_std_dev_left++;
                                }
                        }

                        if(ok_bfit && ok_wfit && ok_avg_fit && ok_std_dev)
                            break;
                    }
                }
            }
        }
    }

    generate_table_widget(title: string, nb_row: number, nb_col: number): [QWidget, QTableWidget] {
        const widget = new QWidget;
        const layout = new QBoxLayout(2);
        widget.setLayout(layout);

        const label = new QLabel;
        label.setText(title);
        label.setAlignment(AlignmentFlag.AlignCenter | AlignmentFlag.AlignTop);
        label.setInlineStyle('font-size: 11pt;');

        const table = new QTableWidget(nb_row, nb_col);
        table.setHorizontalScrollBarPolicy(ScrollBarPolicy.ScrollBarAlwaysOff);

        // sorts
        table.addEventListener('cellClicked',(row, col) => {
            table.sortByColumn(col, SortOrder.AscendingOrder);
        });

        table.addEventListener('cellDoubleClicked',(row, col) => {
            table.sortByColumn(col, SortOrder.DescendingOrder);
        });
        
        if(nb_col === 2) {
            table.setHorizontalHeaderLabels(['Run number', 'Last generation value']);
        } else if(nb_col === 3) {
            table.setHorizontalHeaderLabels(['Run number', 'Value','Since generation']);
        }
        
        if(nb_col === 2){
            table.setFixedSize(400, 210);
            table.setColumnWidth(0, 179);
            table.setColumnWidth(1, 179);
        } else if(nb_col === 3) {
            table.setFixedSize(500, 210);
            table.setColumnWidth(0, 120);
            table.setColumnWidth(1, 170);
            table.setColumnWidth(2, 170);
        }

        // table.item(1,1).setFlags(ItemFlag.NoItemFlags)
        layout.addWidget(label, undefined, AlignmentFlag.AlignCenter | AlignmentFlag.AlignTop);
        layout.addWidget(table, undefined, AlignmentFlag.AlignCenter | AlignmentFlag.AlignTop);

        return [widget, table];
    }

    reset_tables():void {
        // clear best fitness
        this.best_fit_left.clear();
        this.best_fit_left.setHorizontalHeaderLabels(['Run number', 'Last generation value']);
        
        this.best_fit_right.clear();
        this.best_fit_right.setHorizontalHeaderLabels(['Run number', 'Value','Since generation']);

        // clear worst fitness
        this.worst_fit_left.clear();
        this.worst_fit_left.setHorizontalHeaderLabels(['Run number', 'Last generation value']);
        
        this.worst_fit_right.clear();
        this.worst_fit_right.setHorizontalHeaderLabels(['Run number', 'Value','Since generation']);

        // clear avg fitness
        this.avg_fit_left.clear();
        this.avg_fit_left.setHorizontalHeaderLabels(['Run number', 'Last generation value']);
        
        this.avg_fit_right.clear();
        this.avg_fit_right.setHorizontalHeaderLabels(['Run number', 'Value','Since generation']);

        // clear std deviation
        this.std_dev_left.clear();
        this.std_dev_left.setHorizontalHeaderLabels(['Run number', 'Last generation value']);
        
        this.std_dev_right.clear();
        this.std_dev_right.setHorizontalHeaderLabels(['Run number', 'Value','Since generation']);
    }
}