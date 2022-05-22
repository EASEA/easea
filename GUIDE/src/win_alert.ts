/**
 * @author Clément Ligneul <clement.ligneul@etu.unistra.fr>
 */

import { QDialog, QLabel, QPushButton, AlignmentFlag, QGridLayout, WindowType } from '@nodegui/nodegui';

export class Win_alert {
    message: string;
    window: QDialog;

    constructor(message: string, window_name?: string) {
        this.message = message;
        this.window = new QDialog();

        this.window.setObjectName("window");
        this.window.setWindowFlag(WindowType.CustomizeWindowHint, true);
        this.window.setWindowFlag(WindowType.WindowCloseButtonHint, false);

        if (window_name) {
            this.window.setWindowTitle(window_name);
        } else {
            this.window.setWindowTitle('Error');
        }

        var layout = new QGridLayout();

        const label = new QLabel();
        label.setObjectName("label");
        label.setText(this.message);
        label.setObjectName('label');
        label.setAlignment(AlignmentFlag.AlignCenter);

        const quit_button = new QPushButton();
        quit_button.setText("Close");
        quit_button.setFixedSize(90, 25);
        quit_button.addEventListener("clicked", () => this.window.close());
        quit_button.setObjectName('button');

        layout.addWidget(label, 0, 0, 1, 3);
        layout.addWidget(quit_button, 1, 1);
        this.window.setLayout(layout);

        this.window.setMinimumSize(200, 100);
        this.window.adjustSize();
        this.window.setFixedSize(this.window.size().width() + 15, this.window.size().height());
        this.window.exec();

    }
}
