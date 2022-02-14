/**
 * @author Clément Ligneul <clement.ligneul@etu.unistra.fr>
 */

import { Direction, QBoxLayout, QComboBox, QDialog, QPushButton, QWidget, WindowType } from "@nodegui/nodegui";
import { Advanced_option_widget } from "./advanced_option_widget";
import { general_css } from "./style";
import { print_errors } from "./utilities";
import * as util from './utilities';

export class Offspring_options_win {
    window: QDialog
    size_off: number;
    surviving_off: number;
    surviving_off_type: string;
    reduce_op: string;
    reduce_pressure: number;

    constructor() {
        this.window = new QDialog();
        this.window.setWindowTitle('Offspring Options');
        this.window.setWindowFlag(WindowType.CustomizeWindowHint, true);
        this.window.setWindowFlag(WindowType.WindowCloseButtonHint, false);
        this.surviving_off = NaN;
        this.reduce_op = 'Tournament';
        this.reduce_pressure = 2.0;
        this.size_off = NaN;
        this.surviving_off_type = '#';

        util.disable_keys('Escape', this.window);

        const main_layout = new QBoxLayout(Direction.TopToBottom);

        const nb_off = new Advanced_option_widget('Offspring Size :', 0);
        nb_off.text_edit.addEventListener('textChanged', () => {
            var text = nb_off.text_edit.text();
            var val = Number(text);

            if (text === '') {
                this.size_off = NaN;
                return;
            }

            if (!isNaN(val))
                this.size_off = val;
        });

        const surviving_offspring = new Advanced_option_widget('Surviving Offspring :', 0);
        surviving_offspring.text_edit.addEventListener('textChanged', () => {
            var text = surviving_offspring.text_edit.text();
            var val = Number(text);

            if (text === '') {
                this.surviving_off = NaN;
                return;
            }

            if (!isNaN(val))
                this.surviving_off = val;
        });

        const surviving_off_type = new QComboBox();
        surviving_off_type.addItem(undefined, '#');
        surviving_off_type.addItem(undefined, '%');
        surviving_off_type.addEventListener('currentTextChanged', () => {
            this.surviving_off_type = surviving_off_type.currentText();
        });

        surviving_offspring.layout.addWidget(surviving_off_type);

        const reduce_off_op = new Advanced_option_widget('Reduce Operator :', 0, 'Tournament');
        reduce_off_op.text_edit.addEventListener('textChanged', () => {
            var text = reduce_off_op.text_edit.text().toLowerCase();

            if (text === 'tournament' || text === '') {
                reduce_off_pressure.widget.setEnabled(true);
            } else {
                reduce_off_pressure.widget.setEnabled(false);
                reduce_off_pressure.text_edit.clear();
            }

            text = reduce_off_op.text_edit.text();

            if (text === '') {
                this.reduce_op = 'Tournament';
            } else {
                this.reduce_op = text;
            }
        });

        const reduce_off_pressure = new Advanced_option_widget('Reduce Pressure :', 0, '2.0');
        reduce_off_pressure.text_edit.addEventListener('textChanged', () => {
            var text = reduce_off_pressure.text_edit.text();
            var val = Number(text);

            if (text === '') {
                this.reduce_pressure = NaN;
                return;
            }

            if (!isNaN(val))
                this.reduce_pressure = val;
        });

        // close and save buttons
        const btn_layout = new QBoxLayout(Direction.LeftToRight);
        const btn_widget = new QWidget();
        btn_widget.setLayout(btn_layout);

        const save_btn = new QPushButton();
        save_btn.setText('Save');
        save_btn.addEventListener('clicked', () => {
            var errors = [];
            var ok = 1;

            var pressure = Number(reduce_off_pressure.text_edit.text());
            if (isNaN(pressure)) {
                ok = 0;
                errors.push('Reduce Pressure');
            } else if (pressure < 0.5 && pressure != 0) {
                ok = 0;
                errors.push('Reduce Pressure : should be in [ 0.5 ; 1 [ U [ 2 ; +\u221e [');
            } else if (pressure >= 1 && pressure < 2) {
                ok = 0;
                errors.push('Reduce Pressure : should be in [ 0.5 ; 1 [ U [ 2 ; +\u221e [');
            } else if (pressure >= 2 && pressure % 1 !== 0) {
                ok = 0;
                errors.push('Reduce Pressure : should be an integer if \u2265 2');
            }

            if (isNaN(Number(surviving_offspring.text_edit.text()))) {
                ok = 0;
                errors.push('Surviving Offspring');
            }

            if (isNaN(Number(nb_off.text_edit.text()))) {
                ok = 0;
                errors.push('Offspring Size');
            }

            if (ok) {
                this.window.close();
            } else {
                print_errors(errors);
            }
        });
        save_btn.setFixedSize(100, 25);

        const reset_btn = new QPushButton();
        reset_btn.setText('Reset');
        reset_btn.addEventListener('clicked', () => {
            reduce_off_op.text_edit.clear();

            reduce_off_pressure.text_edit.clear();

            nb_off.text_edit.clear();

            surviving_offspring.text_edit.clear();
        });
        reset_btn.setFixedSize(100, 25);

        btn_layout.addWidget(save_btn);
        btn_layout.addWidget(reset_btn);


        main_layout.addWidget(nb_off.widget);
        main_layout.addWidget(surviving_offspring.widget);
        main_layout.addWidget(reduce_off_op.widget);
        main_layout.addWidget(reduce_off_pressure.widget);
        main_layout.addWidget(btn_widget);

        this.window.setLayout(main_layout);
        this.window.setStyleSheet(general_css);
        this.window.adjustSize();
        this.window.setFixedSize(this.window.size().width(), this.window.size().height());
    }

    execution() {
        this.window.exec();
        return this;
    }


}