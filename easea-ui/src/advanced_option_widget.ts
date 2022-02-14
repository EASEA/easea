/**
 * @author Clément Ligneul <clement.ligneul@etu.unistra.fr>
 */

import { QBoxLayout, QLabel, QLineEdit, QWidget } from "@nodegui/nodegui";
import { exit } from "process";


export class Advanced_option_widget {
    label_text: string;
    label: QLabel;
    place_holder: string;
    widget: QWidget;
    layout: QBoxLayout;
    text_edit: QLineEdit;


    constructor(txt: string, alignement: number, place_hold?: string) {
        if (alignement < 0 || alignement > 3) {
            console.log('bad parameters for advanced widget');
            exit(1);
        }

        this.text_edit = new QLineEdit();
        this.text_edit.setFixedSize(95, 30);

        this.label_text = txt;

        this.label = new QLabel();
        this.label.setText(this.label_text);

        this.widget = new QWidget();
        this.layout = new QBoxLayout(alignement);
        this.widget.setLayout(this.layout);

        place_hold ? this.place_holder = place_hold : this.place_holder = 'Default';

        this.text_edit.setPlaceholderText(this.place_holder);
        this.layout.addWidget(this.label);
        this.layout.addWidget(this.text_edit);
        this.widget.setMaximumSize(330, 100);
    }

}