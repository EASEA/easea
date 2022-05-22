/**
 * @author Clément Ligneul <clement.ligneul@etu.unistra.fr>
 */

import { QBoxLayout, QLabel, QLineEdit, QWidget } from "@nodegui/nodegui";

export class Processus {
    proc_widget: QWidget;
    proc_label: QLabel;
    proc_layout: QBoxLayout;
    seed_text: QLineEdit;
    rang: number;
    run_options: string;
    seed_value: number;

    constructor(num: number) {
        this.run_options = '';
        this.rang = num;
        this.proc_widget = new QWidget();
        this.proc_layout = new QBoxLayout(0);
        this.seed_text = new QLineEdit();
        this.proc_label = new QLabel();
        this.seed_value = NaN;

        this.proc_label.setText('Run ' + num + ' : ');
        this.proc_label.setFixedSize(80, 30);

        this.seed_text.setPlaceholderText('Default if empty');
        this.seed_text.setFixedSize(130, 30);

        this.proc_layout.addWidget(this.proc_label);
        this.proc_layout.addWidget(this.seed_text);

        this.proc_widget.setLayout(this.proc_layout);

        this.seed_text.addEventListener('textChanged', () => {
            var seed = Number(this.seed_text.text());

            if (!isNaN(seed) && seed > 0 && !this.seed_text.isReadOnly())
                this.seed_value = seed;

            if (this.seed_text.text() === '')
                this.seed_value = NaN;
        });
    }

    enable() {
        this.seed_text.setReadOnly(false);
        this.seed_text.setPlaceholderText('Default if empty');

        if (!isNaN(this.seed_value))
            this.seed_text.setText(this.seed_value.toString());
    }

    disable() {
        this.seed_text.setReadOnly(true);
        this.seed_text.setPlaceholderText('Disabled');
        this.seed_text.setText('');
    }
}