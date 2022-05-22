/**
 * @author Clément Ligneul <clement.ligneul@etu.unistra.fr>
 */

import { AlignmentFlag, ButtonSymbols, QBoxLayout, QDoubleSpinBox, QLabel, QLineEdit, QSpinBox, QWidget } from "@nodegui/nodegui";
import { exit } from "process";


export class Advanced_option_widget {
    label_text: string;
    label: QLabel;
    place_holder: string;
    widget: QWidget;
    layout: QBoxLayout;
    text_edit: QLineEdit = new QLineEdit;
    text_edit_spin: QSpinBox = new QSpinBox;
    text_edit_double_spin: QDoubleSpinBox = new QDoubleSpinBox;


    constructor(txt: string, alignement: number, place_hold?: string, isNum: boolean = false, isFloat = false) {
        if (alignement < 0 || alignement > 3) {
            console.log('bad parameters for advanced widget');
            exit(1);
        }

        // text edit
        this.text_edit.setFixedSize(95, 30);

        // spinbox
        this.text_edit_spin.setFixedSize(95, 30);
        this.text_edit_spin.setRange(0,1000000);
        this.text_edit_spin.setButtonSymbols(ButtonSymbols.NoButtons);
        // this.text_edit_spin.setValue(0);

        // doublespinbox
        this.text_edit_double_spin.setFixedSize(95, 30);
        this.text_edit_double_spin.setRange(0,1000000);
        this.text_edit_double_spin.setButtonSymbols(ButtonSymbols.NoButtons);
        this.text_edit_double_spin.setAlignment(AlignmentFlag.AlignCenter);
        // this.text_edit_double_spin.setValue(0);

        this.label_text = txt;

        this.label = new QLabel();
        this.label.setText(this.label_text);

        this.widget = new QWidget();
        this.layout = new QBoxLayout(alignement);
        this.widget.setLayout(this.layout);

        place_hold ? this.place_holder = place_hold : this.place_holder = 'Default';

        this.text_edit.setPlaceholderText(this.place_holder);
        this.layout.addWidget(this.label);
        if(isNum){
            this.label.setAlignment(AlignmentFlag.AlignCenter);
            if(isFloat){
                this.layout.addWidget(this.text_edit_double_spin, undefined, AlignmentFlag.AlignCenter);
            } else {
                this.layout.addWidget(this.text_edit_spin, undefined, AlignmentFlag.AlignCenter);
            }
        } else {
            this.layout.addWidget(this.text_edit);
        }
        this.widget.setMaximumSize(330, 100);
    }

}