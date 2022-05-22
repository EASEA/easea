/**
 * @author Clément Ligneul <clement.ligneul@etu.unistra.fr>
 */

import { Direction, QBoxLayout, QDialog, QCheckBox, QPushButton, QWidget, QLabel, QFileDialog, WindowType, FileMode, QComboBox, AlignmentFlag } from "@nodegui/nodegui";
import { exit } from "process";
import { Advanced_option_widget } from "./advanced_option_widget";
import { general_css } from "./style";
import * as util from './utilities';
import fs from 'fs';
import ip from 'ip';

export class Island_options_win {
    window: QDialog;
    reevaluate: boolean;
    ip_file: string;
    ip_table: number[];         // array of ports or ip addresses
    migration_proba: number;
    server_port: number;
    local: boolean;
    nb_islands: number;
    nb_isl_per_run: number;

    constructor() {
        this.window = new QDialog();
        this.window.setWindowTitle('Island Model Options');
        this.window.setWindowFlag(WindowType.CustomizeWindowHint, true);
        this.window.setWindowFlag(WindowType.WindowCloseButtonHint, false);
        this.reevaluate = false;
        this.server_port = NaN;
        this.migration_proba = 0.33;
        this.ip_file = '';
        this.ip_table = [];
        this.local = true;
        this.nb_islands = 2;
        this.nb_isl_per_run = 2;

        util.disable_keys('Escape', this.window);

        const main_layout = new QBoxLayout(Direction.TopToBottom);

        // separator island management 
        const separator_island_management = this.generate_separator('Island Management');

        // Immigrants evaluation
        const reevaluate_im = new QCheckBox();
        reevaluate_im.setText('Evaluate Immigrants');
        reevaluate_im.setStyleSheet("margin-left: 10px;");
        reevaluate_im.addEventListener('stateChanged', () => {
            this.reevaluate = reevaluate_im.isChecked();
        });

        // Number of Islands
        const nb_islands_widget = new Advanced_option_widget('Ports/machines available : ' , 0, '2');
        nb_islands_widget.text_edit.addEventListener('textChanged', () => {
            let text = nb_islands_widget.text_edit.text();
            let val = Number(text);

            if (text === '') {
                this.nb_islands = 2;

                combo_batch.clear();

                combo_batch.insertItem(0, undefined, '2');
                combo_batch.setCurrentIndex(0);

                return;
            }

            this.nb_islands = val;
            let batch_array: number[] = [];                
            let new_items = 0;
                
            // batch propositions                
            let tmp = val;
            let n = 2;

            while(n != 30){
                if(tmp % n == 0 && tmp != 0){
                    batch_array.push(n);
                    new_items++;
                }
                n++;
            }

            combo_batch.clear();

            if(combo_batch.itemText(1) != undefined)
                combo_batch.removeItem(1);

            for(let i = 0; i < new_items; i++){
                combo_batch.insertItem(i+1, undefined, batch_array[i].toString());
            }

            combo_batch_size = new_items + 1;
            
            combo_batch.setCurrentIndex(0);
        });

        // Batch propositions
        const batch_prop_layout = new QBoxLayout(0);
        const batch_prop_widget = new QWidget();
        const batch_prop_label = new QLabel();
        batch_prop_label.setText('Number of Islands per run :');
        batch_prop_label.setFixedSize(180, 30);
        batch_prop_widget.setLayout(batch_prop_layout);

        const combo_batch = new QComboBox();
        let combo_batch_size = 1;
        combo_batch.setFixedSize(100, 30);
        combo_batch.addItem(undefined, '2');
        combo_batch.addEventListener('currentTextChanged', (val) => {
            this.nb_isl_per_run = Number(val);
        });

        batch_prop_layout.addWidget(batch_prop_label);
        batch_prop_layout.addWidget(combo_batch);

        //number of island per run
        // const nb_isl_per_run_widget = new Advanced_option_widget('Number of Island per run : ', 0, '2');
        // nb_isl_per_run_widget.text_edit.addEventListener('textChanged', () =>{
        //     let text = nb_isl_per_run_widget.text_edit.text();
        //     let val = Number(text);

        //     if(text === ''){
        //         this.nb_isl_per_run = 2;
        //         return;
        //     }

        //     this.nb_isl_per_run = val;
        // });

        // Island parameters separator
        const separator_options = this.generate_separator('Island Parameters');

        // Remote or local islands
        const local_layout = new QBoxLayout(0);
        const local_widget = new QWidget();
        const local_label = new QLabel();
        local_label.setText('Batch type :');
        local_label.setFixedSize(180, 30);
        local_widget.setLayout(local_layout);
        const combo_local = new QComboBox();
        combo_local.setFixedSize(100, 30);
        combo_local.addItem(undefined, 'Local');
        combo_local.addItem(undefined, 'Remote');
        combo_local.addEventListener('currentTextChanged', (val) => {
            if(val === 'Local'){
                this.local = true;
                ip_widget.hide();
            } else if(val === 'Remote'){
                this.local = false;
                ip_widget.show();
            } else {
                console.log('Island options window : local type error\n');
                exit(1);
            }
        });

        local_layout.addWidget(local_label);
        local_layout.addWidget(combo_local);

        // IP file
        const ip_widget = new QWidget();
        const ip_layout = new QBoxLayout(Direction.LeftToRight);
        ip_widget.setLayout(ip_layout);
        const ip_file_btn = new QPushButton();
        const file_loaded = new QLabel();
        file_loaded.setText('IP file :');
        ip_file_btn.setText("Load IP file");
        ip_file_btn.setFixedSize(120, 30);
        ip_file_btn.setAutoDefault(false);
        ip_file_btn.setDefault(false);
        ip_file_btn.addEventListener('clicked', () => {
            const fileDialog = new QFileDialog();
            fileDialog.setFileMode(FileMode.ExistingFile);

            if (fileDialog.exec()) {
                let file = fileDialog.selectedFiles().toString();
                if (file) {
                    this.ip_file = file
                    file_loaded.setText('IP file : ' + util.get_file_name(this.ip_file));
                }
            }

        });

        ip_layout.addWidget(file_loaded);
        ip_layout.addWidget(ip_file_btn);
        ip_widget.hide();

        // Migration probability
        const proba_migration = new Advanced_option_widget('Migration Probability :', 0, '0.33');
        proba_migration.text_edit.addEventListener('textChanged', () => {
            let text = proba_migration.text_edit.text();
            let val = Number(text);

            if (text === '') {
                this.migration_proba = 0.33;
                return;
            }

            if (!isNaN(val) && val >= 0 && val <=1){
                this.migration_proba = val;
            } else {
                this.migration_proba = NaN;
            }
        });

        // close and save buttons
        const btn_layout = new QBoxLayout(Direction.LeftToRight);
        const btn_widget = new QWidget();
        btn_widget.setLayout(btn_layout);

        const save_btn = new QPushButton();
        save_btn.setText('Save');
        save_btn.setAutoDefault(false);
        save_btn.setDefault(true);
        save_btn.addEventListener('clicked', () => {
            let errors = [];
            let ok = 1;

            // let proba_migr = Number(proba_migration.text_edit.text());
            if(isNaN(this.migration_proba)) {
                ok = 0;
                errors.push('Migration Probability');
            }

            if(isNaN(this.nb_islands) || this.nb_islands < 2){
                ok = 0;
                errors.push('Ports/machines available : must be ≥ 2');
            }

            if(isNaN(this.nb_isl_per_run) || this.nb_isl_per_run < 2 || this.nb_isl_per_run > this.nb_islands){
                ok = 0;
                errors.push('Number of Islands per run : must be in [2;nb islands]');
            }

            if(this.local && this.ip_file != ''){
                this.ip_table = [];
                const text_file = fs.readFileSync(this.ip_file, 'utf-8');
                const lines = text_file.split('\n');
                for (let i = 0; i < lines.length - 1; i++) {
                    // remove spaces
                    lines[i] = lines[i].replace(/\s/g, '');

                    let num = lines[i].split(':');
                    if(num.length != 2){
                        errors.push('IP file line ' + (i+1) + ' : no port found');
                        ok = 0;
                    }

                    if(num[0] != '127.0.0.1' && num[0] != ip.address() || num[1] === '' || isNaN(+num[1])){
                        errors.push('The ip file must only contain\nlocal addresses with port number')
                        ok = 0;
                        break;
                    }
                    this.ip_table.push(+num[1]);
                }
            }

            ok ? this.window.close() : util.print_errors(errors);
        });

        save_btn.setFixedSize(100, 25);

        const reset_btn = new QPushButton();
        reset_btn.setText('Reset');
        reset_btn.setAutoDefault(false);
        reset_btn.setDefault(false);
        reset_btn.addEventListener('clicked', () => {
            file_loaded.setText('IP file :');
            this.ip_file = '';
            proba_migration.text_edit.clear();
            reevaluate_im.setChecked(false);
            nb_islands_widget.text_edit.clear();
        });

        reset_btn.setFixedSize(100, 25);

        btn_layout.addWidget(save_btn);
        btn_layout.addWidget(reset_btn);

        main_layout.addWidget(separator_island_management, undefined, AlignmentFlag.AlignCenter);
        main_layout.addWidget(local_widget, undefined, AlignmentFlag.AlignLeft);
        main_layout.addWidget(nb_islands_widget.widget, undefined, AlignmentFlag.AlignLeft);
        main_layout.addWidget(batch_prop_widget, undefined, AlignmentFlag.AlignLeft);
        main_layout.addWidget(ip_widget, undefined, AlignmentFlag.AlignLeft);
        main_layout.addWidget(separator_options, undefined, AlignmentFlag.AlignCenter);
        main_layout.addWidget(proba_migration.widget, undefined, AlignmentFlag.AlignCenter);
        main_layout.addWidget(reevaluate_im, undefined, AlignmentFlag.AlignCenter);
        main_layout.addWidget(btn_widget, undefined, AlignmentFlag.AlignCenter);

        this.window.setLayout(main_layout);
        this.window.adjustSize();
        this.window.setFixedSize(this.window.size().width(), this.window.size().height());
        this.window.setStyleSheet(general_css);
    }

    execution() {
        this.window.exec();
        return this;
    }

    generate_separator(text:string) {
        const sep = new QLabel();
        sep.setText(text);
        sep.setAlignment(AlignmentFlag.AlignCenter);
        sep.setFixedSize(300,30);
        sep.setStyleSheet(`
            font-size: 15pt; 
            background-color: #ececec;
            border-bottom: 0.5px solid;
            padding-bottom: 5px;
        `);

        return sep;
    }   
}