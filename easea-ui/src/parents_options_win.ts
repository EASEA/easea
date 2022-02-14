/**
 * @author Clément Ligneul <clement.ligneul@etu.unistra.fr>
 */

import { Direction, QBoxLayout, QComboBox, QDialog, QPushButton, QWidget, WindowType } from "@nodegui/nodegui";
import { Advanced_option_widget } from "./advanced_option_widget";
import { general_css } from "./style";
import * as util from './utilities';


export class Parent_options_win {
    surviving: number;
    reduce_op: string;
    reduce_pressure: number;
    window: QDialog;
    surviving_parent_type: string;

    constructor() {
        this.window = new QDialog();
        this.window.setWindowTitle('Parents Options');
        this.window.setWindowFlag(WindowType.CustomizeWindowHint, true);
        this.window.setWindowFlag(WindowType.WindowCloseButtonHint, false);
        this.surviving = NaN;
        this.reduce_op = 'Tournament';
        this.reduce_pressure = 2.0;
        this.surviving_parent_type = '#';

        util.disable_keys('Escape', this.window);

        const main_layout = new QBoxLayout(Direction.TopToBottom);

        const surviving_parent = new Advanced_option_widget('Surviving Parents :', 0)
        const surviving_parent_type = new QComboBox();
        surviving_parent_type.addItem(undefined, '#');
        surviving_parent_type.addItem(undefined, '%');
        surviving_parent_type.addEventListener('currentTextChanged', () => {
            this.surviving_parent_type = surviving_parent_type.currentText();
        });

        surviving_parent.layout.addWidget(surviving_parent_type);

        surviving_parent.text_edit.addEventListener('textChanged', () => {
            var text = surviving_parent.text_edit.text();
            var val = Number(text);

            if (text === '') {
                this.surviving = NaN;
                return;
            }

            if (!isNaN(val))
                this.surviving = val;
        });

        const reduce_parent_op = new Advanced_option_widget('Reduction Operator :', 0, 'Tournament');
        reduce_parent_op.text_edit.addEventListener('textChanged', () => {
            var text = reduce_parent_op.text_edit.text().toLowerCase();

            if (text === 'tournament' || text === '') {
                reduce_parent_pressure.widget.setEnabled(true);
            } else {
                reduce_parent_pressure.widget.setEnabled(false);
                reduce_parent_pressure.text_edit.clear();
            }

            text = reduce_parent_op.text_edit.text();

            text === '' ? this.reduce_op = 'Tournament' : this.reduce_op = text;
        });

        const reduce_parent_pressure = new Advanced_option_widget('Reduce pressure :', 0, '2.0');
        reduce_parent_pressure.text_edit.addEventListener('textChanged', () => {
            var text = reduce_parent_pressure.text_edit.text();
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

            if (isNaN(Number(surviving_parent.text_edit.text()))) {
                ok = 0;
                errors.push('Surviving Parents');
            }

            var pressure = Number(reduce_parent_pressure.text_edit.text());
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

            ok ? this.window.close() : util.print_errors(errors);
        });

        save_btn.setFixedSize(100, 25);

        const reset_btn = new QPushButton();
        reset_btn.setText('Reset');
        reset_btn.addEventListener('clicked', () => {
            surviving_parent.text_edit.clear();
            reduce_parent_pressure.text_edit.clear();
            reduce_parent_op.text_edit.clear();
        });

        reset_btn.setFixedSize(100, 25);

        btn_layout.addWidget(save_btn);
        btn_layout.addWidget(reset_btn);

        main_layout.addWidget(surviving_parent.widget);
        main_layout.addWidget(reduce_parent_op.widget);
        main_layout.addWidget(reduce_parent_pressure.widget);
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