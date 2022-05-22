/**
 * @author Clément Ligneul <clement.ligneul@etu.unistra.fr>
 */

export const general_css =
    `
    QWidget {
        font-size: 13px;
        font-family: 'Proxima Nova';
    }

    QMainWindow {
        background-color:#ececec;
    }

    QLineEdit, QSpinBox, QDoubleSpinBox {
        border-width: 1px; border-radius: 4px;
        border-style: solid;
        border-color: qlineargradient(spread:pad, x1:0.5, y1:1, x2:0.5, y2:0, stop:0 rgba(0, 113, 255, 255), stop:1 rgba(91, 171, 252, 255));
    }

    QMenuBar {
        background-color: qlineargradient(spread:pad, x1:0.5, y1:1, x2:0.5, y2:0, stop:0 rgba(207, 209, 207, 255), stop:1 rgba(230, 229, 230, 255));
    }

    QMenuBar::item {
        color: #000000;
        spacing: 3px;
        padding: 1px 4px;
        background-color: qlineargradient(spread:pad, x1:0.5, y1:1, x2:0.5, y2:0, stop:0 rgba(207, 209, 207, 255), stop:1 rgba(230, 229, 230, 255));
    }

    QMenuBar::item:selected {
        background-color: qlineargradient(spread:pad, x1:0.5, y1:1, x2:0.5, y2:0, stop:0 rgba(0, 113, 255, 255), stop:1 rgba(91, 171, 252, 255));
        color: #FFFFFF;
    }

    QMenu::item:selected {
        border-style: solid;
        border-top-color: transparent;
        border-right-color: transparent;
        border-left-color: qlineargradient(spread:pad, x1:0.5, y1:1, x2:0.5, y2:0, stop:0 rgba(0, 113, 255, 255), stop:1 rgba(91, 171, 252, 255));
        border-bottom-color: transparent;
        border-left-width: 2px;
        color: #000000;
        padding-left:15px;
        padding-top:4px;
        padding-bottom:4px;
        padding-right:7px;
    }

    QMenu::item {
        border-style: solid;
        border-top-color: transparent;
        border-right-color: transparent;
        border-left-color: transparent;
        border-bottom-color: transparent;
        border-bottom-width: 1px;
        color: #000000;
        padding-left:17px;
        padding-top:4px;
        padding-bottom:4px;
        padding-right:7px;
    }

    QTabWidget {
        color:rgb(0,0,0);
        background-color:#000000;
    }

    QTabWidget::pane {
        border-color: rgb(223,223,223);
        background-color:rgb(226,226,226);
        border-style: solid;
        border-width: 2px;
        border-radius: 6px;
    }

    QTabBar::tab:first {
        border-style: solid;
        border-left-width:1px;
        border-right-width:0px;
        border-top-width:1px;
        border-bottom-width:1px;
        border-top-color: rgb(209,209,209);
        border-left-color: qlineargradient(spread:pad, x1:0.5, y1:1, x2:0.5, y2:0, stop:0 rgba(209, 209, 209, 209), stop:1 rgba(229, 229, 229, 229));
        border-bottom-color: rgb(229,229,229);
        border-top-left-radius: 4px;
        border-bottom-left-radius: 4px;
        color: #000000;
        padding: 3px;
        margin-left:0px;
        background-color: qlineargradient(spread:pad, x1:0.5, y1:1, x2:0.5, y2:0, stop:0 rgba(247, 247, 247, 255), stop:1 rgba(255, 255, 255, 255));
    }

    QTabBar::tab:last {
        border-style: solid;
        border-width:1px;
        border-top-color: rgb(209,209,209);
        border-left-color: qlineargradient(spread:pad, x1:0.5, y1:1, x2:0.5, y2:0, stop:0 rgba(209, 209, 209, 209), stop:1 rgba(229, 229, 229, 229));
        border-right-color: qlineargradient(spread:pad, x1:0.5, y1:1, x2:0.5, y2:0, stop:0 rgba(209, 209, 209, 209), stop:1 rgba(229, 229, 229, 229));
        border-bottom-color: rgb(229,229,229);
        border-top-right-radius: 4px;
        border-bottom-right-radius: 4px;
        color: #000000;
        padding: 3px;
        margin-left:0px;
        background-color: qlineargradient(spread:pad, x1:0.5, y1:1, x2:0.5, y2:0, stop:0 rgba(247, 247, 247, 255), stop:1 rgba(255, 255, 255, 255));
    }

    QTabBar::tab {
        border-style: solid;
        border-top-width:1px;
        border-bottom-width:1px;
        border-left-width:1px;
        border-top-color: rgb(209,209,209);
        border-left-color: qlineargradient(spread:pad, x1:0.5, y1:1, x2:0.5, y2:0, stop:0 rgba(209, 209, 209, 209), stop:1 rgba(229, 229, 229, 229));
        border-bottom-color: rgb(229,229,229);
        color: #000000;
        padding: 3px;
        margin-left:0px;
        background-color: qlineargradient(spread:pad, x1:0.5, y1:1, x2:0.5, y2:0, stop:0 rgba(247, 247, 247, 255), stop:1 rgba(255, 255, 255, 255));
    }

    QTabBar::tab:selected, QTabBar::tab:last:selected, QTabBar::tab:hover {
        border-style: solid;
        border-left-width:1px;
        border-right-color: transparent;
        border-top-color: rgb(209,209,209);
        border-left-color: qlineargradient(spread:pad, x1:0.5, y1:1, x2:0.5, y2:0, stop:0 rgba(209, 209, 209, 209), stop:1 rgba(229, 229, 229, 229));
        border-bottom-color: rgb(229,229,229);
        color: #FFFFFF;
        padding: 3px;
        margin-left:0px;
        background-color: qlineargradient(spread:pad, x1:0.5, y1:1, x2:0.5, y2:0, stop:0 rgba(0, 113, 255, 255), stop:1 rgba(91, 171, 252, 255));
    }

    QTabBar::tab:selected, QTabBar::tab:first:selected, QTabBar::tab:hover {
        border-style: solid;
        border-left-width:1px;
        border-bottom-width:1px;
        border-top-width:1px;
        border-right-color: transparent;
        border-top-color: rgb(209,209,209);
        border-left-color: qlineargradient(spread:pad, x1:0.5, y1:1, x2:0.5, y2:0, stop:0 rgba(209, 209, 209, 209), stop:1 rgba(229, 229, 229, 229));
        border-bottom-color: rgb(229,229,229);
        color: #FFFFFF;
        padding: 3px;
        margin-left:0px;
        background-color: qlineargradient(spread:pad, x1:0.5, y1:1, x2:0.5, y2:0, stop:0 rgba(0, 113, 255, 255), stop:1 rgba(91, 171, 252, 255));
    }

    QCheckBox {
        color: #000000;
        padding: 2px;
    }

    QCheckBox:disabled {
        color: #808086;
        padding: 2px;
    }

    QCheckBox:hover {
        border-radius:4px;
        border-style:solid;
        padding-left: 1px;
        padding-right: 1px;
        padding-bottom: 1px;
        padding-top: 1px;
        border-width:1px;
        border-color: transparent;
    }

    QCheckBox::indicator:checked {
        height: 10px;
        width: 10px;
        border-style:solid;
        border-width: 1px;
        border-color: qlineargradient(spread:pad, x1:0.5, y1:1, x2:0.5, y2:0, stop:0 rgba(0, 113, 255, 255), stop:1 rgba(91, 171, 252, 255));
        color: #000000;
        background-color: qlineargradient(spread:pad, x1:0.5, y1:1, x2:0.5, y2:0, stop:0 rgba(0, 113, 255, 255), stop:1 rgba(91, 171, 252, 255));
    }
    
    QCheckBox::indicator:unchecked {
        height: 10px;
        width: 10px;
        border-style:solid;
        border-width: 1px;
        border-color: qlineargradient(spread:pad, x1:0.5, y1:1, x2:0.5, y2:0, stop:0 rgba(0, 113, 255, 255), stop:1 rgba(91, 171, 252, 255));
        color: #000000;
    }

    QTextBrowser, QPlainTextEdit {
        color: white;
        background-color: black;
        font-family: "ubuntu mono";
        font-size: 15px;
    }

    QScrollArea {
        color: #FFFFFF;
    }

    QTableWidget {
        color:#DCDCDC;
        background-color:#444444;
        border:1px solid #242424;
        alternate-background-color:#525252;
        gridline-color:#242424;
    }

    QTableWidget::item:selected {
        color:#DCDCDC;
        background:qlineargradient(spread:pad,x1:0,y1:0,x2:0,y2:0,stop:0 #484848, stop:1 #383838);
    }

    QTableWidget::item:hover {
        background:#5B5B5B;
    }

    QHeaderView::section {
        text-align:center;
        background:#5E5E5E;
        padding:3px;
        margin:0px;
        color:#DCDCDC;
        border:1px solid #242424;
        border-left-width:0;
    }
  `