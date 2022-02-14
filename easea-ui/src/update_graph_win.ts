/**
 * @author Clément Ligneul <clement.ligneul@etu.unistra.fr>
 */

import { QColorDialog, QDialog, QGridLayout, QPushButton, QWidget, WindowType } from "@nodegui/nodegui";
import { plot_obj, run_obj } from ".";
import { Advanced_option_widget } from "./advanced_option_widget";
import * as util from './utilities';
import { general_css } from "./style";

export class Update_graph_win {
    window: QDialog;
    title: string;
    x_axis: string;
    y_axis: string;
    z_axis: string;
    nb_plots: number;
    changed: boolean = false;
    color: string;
    color_palet: QPushButton;
    axe_z_box: Advanced_option_widget;
    plots: Advanced_option_widget;

    constructor() {
        this.window = new QDialog();
        this.title = 'Results';
        this.x_axis = 'f1';
        this.y_axis = 'f2';
        this.z_axis = 'f3';
        this.nb_plots = 1;
        this.color = '#FF0000';

        this.window.setWindowTitle('Plot Parameters');
        this.window.setWindowFlag(WindowType.CustomizeWindowHint, true);
        this.window.setWindowFlag(WindowType.WindowCloseButtonHint, false);

        util.disable_keys('Escape', this.window);

        const layout = new QGridLayout();
        this.window.setLayout(layout);

        const title_box = new Advanced_option_widget('Plot Title : ', 0, 'Results');
        title_box.text_edit.addEventListener('textChanged', () => {
            var text = title_box.text_edit.text();

            if (!text) {
                this.title = 'Results';
            } else {
                this.title = text;
            }

        });

        const axe_x_box = new Advanced_option_widget('x axis name : ', 0, 'f1');
        axe_x_box.text_edit.addEventListener('textChanged', () => {
            var text = axe_x_box.text_edit.text();

            if (!text) {
                this.x_axis = 'f1';
            } else {
                this.x_axis = text;
            }
        });

        const axe_y_box = new Advanced_option_widget('y axis name : ', 0, 'f2');
        axe_y_box.text_edit.addEventListener('textChanged', () => {
            var text = axe_y_box.text_edit.text();

            if (!text) {
                this.y_axis = 'f2';
            } else {
                this.y_axis = text;
            }
        });

        this.axe_z_box = new Advanced_option_widget('z axis name : ', 0, 'f3');
        this.axe_z_box.text_edit.addEventListener('textChanged', () => {
            var text = this.axe_z_box.text_edit.text();

            if (!text) {
                this.z_axis = 'f3';
            } else {
                this.z_axis = text;
            }
        });

        this.plots = new Advanced_option_widget('Nb Plots : ', 0, '1');
        this.plots.text_edit.addEventListener('textChanged', () => {
            var val = Number(this.plots.text_edit.text());

            if (isNaN(val) || val < 1 || val > run_obj.total_generations) {
                this.nb_plots = 1;
            } else {
                this.nb_plots = val;
            }
        });

        // graph color
        this.color_palet = new QPushButton();
        this.color_palet.setFixedSize(this.plots.widget.size().width() - 10, 25);
        this.color_palet.setText('Change Color');
        this.color_palet.addEventListener('clicked', ()=>{
            const col = new QColorDialog();
            col.addEventListener('accepted', ()=>{
                this.color = '#' + col.selectedColor().red().toString(16)
                + col.selectedColor().green().toString(16) 
                + col.selectedColor().blue().toString(16);
            });
            col.exec();
        });

        // buttons
        const btn_widget = new QWidget();
        const btn_layout = new QGridLayout();

        const save_btn = new QPushButton();
        save_btn.setText('Update');
        save_btn.setFixedSize(100, 25);
        save_btn.addEventListener('clicked', () => {
            var errors = [];
            var ok = 1;

            var nplots = Number(this.plots.text_edit.text())
            if (isNaN(nplots) || nplots > run_obj.total_generations || nplots < 1 && this.plots.text_edit.text()) {
                errors.push('Number of plots (must be \u2264 nb generations)');
                ok = 0;
            }

            if (run_obj.plot_type === '3D' && this.plots.text_edit.text()) {
                errors.push('Number of plots available only for 2D graphs');
                ok = 0;
            }

            if (run_obj.plot_type === '2D') {
                if (this.axe_z_box.text_edit.text()) {
                    errors.push('z Axis name only available for 3D graphs');
                    ok = 0;
                }
            }

            if (ok) {
                this.changed = true;
                this.window.close();
            } else {
                util.print_errors(errors);
            }
        });

        const close_btn = new QPushButton();
        close_btn.setText('Close');
        close_btn.setFixedSize(100, 25);
        close_btn.addEventListener('clicked', () => {
            this.changed = false;
            this.window.close();
        });


        const reset_btn = new QPushButton();
        reset_btn.setText('Reset');
        reset_btn.setFixedSize(100, 25);
        reset_btn.addEventListener('clicked', () => {
            title_box.text_edit.clear();
            axe_x_box.text_edit.clear();
            axe_y_box.text_edit.clear();
            this.axe_z_box.text_edit.clear();
            this.plots.text_edit.clear();
        });

        btn_layout.addWidget(save_btn, 0, 0);
        btn_layout.addWidget(reset_btn, 0, 3);
        btn_layout.addWidget(close_btn, 0, 1);

        btn_widget.setLayout(btn_layout);

        //display
        layout.addWidget(title_box.widget, 0, 0);
        layout.addWidget(axe_x_box.widget, 1, 0);
        layout.addWidget(axe_y_box.widget, 2, 0);
        layout.addWidget(this.axe_z_box.widget, 3, 0);
        layout.addWidget(this.plots.widget, 4, 0);
        layout.addWidget(btn_widget, 6, 0);
        layout.addWidget(this.color_palet, 5, 0);

        this.window.adjustSize();
        this.window.setFixedSize(this.window.size().width(), this.window.size().height());
        this.window.setStyleSheet(general_css);
    }

    execution() {
        this.window.exec();

        if (this.changed) {
            plot_obj.update_label.show();
            plot_obj.update_plot(
                    '/tmp/plotting/fig.svg', 
                    this.nb_plots, 
                    run_obj.plot_type, 
                    run_obj.dir_path + '/objectives', 
                    this.title, 
                    this.x_axis, 
                    this.y_axis, 
                    this.z_axis,
                    this.color
                );
        }
        return;
    }
}