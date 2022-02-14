/**
 * @author Clément Ligneul <clement.ligneul@etu.unistra.fr>
 */

import { AlignmentFlag, Orientation, WindowType, QDialog, QGridLayout, QLabel, QPlainTextEdit, QPushButton, QSlider, QWidget, TickPosition, QBoxLayout } from "@nodegui/nodegui";
import { general_css } from "./style";
import { run_obj } from ".";
import * as util from "./utilities"

export class Results_win {
    window: QDialog;
    layout: QBoxLayout;
    slider: QSlider;
    console: QPlainTextEdit;
    run_label: QLabel;

    constructor() {
        this.window = new QDialog();
        this.window.setWindowTitle('Batch Results')
        this.window.setWindowFlag(WindowType.CustomizeWindowHint, true);
        this.window.setWindowFlag(WindowType.WindowCloseButtonHint, false);
        this.window.setStyleSheet(general_css);

        this.layout = new QBoxLayout(2);
        this.window.setLayout(this.layout);

        this.slider = new QSlider();

        this.console = new QPlainTextEdit();
        this.console.setFixedSize(1000, 450);
        this.console.setReadOnly(true);

        this.run_label = new QLabel();
    }

    generate() {
        let batch_size = run_obj.runned_proc;

        // batch size
        let bsize_label = new QLabel();
        bsize_label.setText('Batch Size : ' + run_obj.batch_size);

        // batch average
        let average_label = new QLabel();
        let average_value: number = 0;
        for (let i = 0; i < run_obj.runned_proc; i++) {
            average_value += util.get_best_fitness(run_obj.run_results[i]);
        }

        average_value = Math.floor(average_value * 100000) / 100000;
        average_value = average_value / run_obj.runned_proc;
        average_value = Number(average_value.toFixed(5));
        average_label.setText('Batch Average : ' + average_value);

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

        // display
        this.layout.addWidget(run_label_sep, undefined, AlignmentFlag.AlignCenter);
        this.layout.addWidget(bsize_label);
        this.layout.addWidget(average_label);

        if (batch_size > 1)
            this.layout.addWidget(slider_widget);

        this.layout.addWidget(this.console);
        this.layout.addWidget(btn_widget);


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
            background-color:#ececec;
        `);

        return sep;
    }
}