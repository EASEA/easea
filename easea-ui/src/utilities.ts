/**
 * @author Clément Ligneul <clement.ligneul@etu.unistra.fr>
 */

import { ChildProcess } from "child_process";
import { Win_alert } from "./win_alert";
import fs from 'fs';
import { QKeySequence, QShortcut, QWidget } from "@nodegui/nodegui";

// sleep function equivalent (async call) 
export function sleep(ms: number) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// return file name from his path
export function get_file_name(path: string): string {
    if (path === '')
        return '';

    var result = '';

    for (var i = 0; i < path.length; i++) {
        var last = path.charAt(path.length - 1);

        if (last === '/') {
            return result;
        } else {
            path = path.substring(0, path.length - 1);
            result = last.concat(result);
        }
    }

    return 'error';
}

// return directory path
export function get_path(str: string): string {
    var res = str;

    for (var i = 0; i < str.length; i++) {
        var last = res.charAt(res.length - 1);

        if (last === '/') {
            return res;
        } else {
            res = res.substring(0, res.length - 1);
        }
    }

    return res;
}

// kill all child processes
export function kill_all(child_array: ChildProcess[]) {
    for (var i = 0; i < child_array.length; i++) {
        if (child_array[i].kill())
            console.log('send kill to process ' + i);

        if (child_array[i].killed) {
            console.log(i + ' is killed');
        } else {
            console.log('error : child not killed : ' + i);
        }
    }
}


export function print_errors(errors: string[]) {
    var message = 'The following parameters are not valid :\n';

    for (var i = 0; i < errors.length; i++)
        message = message.concat('\n- ' + errors[i]);
    
    message.concat('\n');

    new Win_alert(message, 'Parameters Error');
}

// return the best fitness from results
export function get_best_fitness(text: string): number {
    var res: number = 0;
    var row = text.split('\n');

    for (var i = 0; i < row.length; i++) {
        if (row[i].startsWith('EASEA LOG [INFO]: Best fitness:')) {
            res = Number(row[i].split(':')[2]);
        }
    }

    return res
}

// errors correction in csv file
export function fix_csv() {
    var updated = 0;
    var data = fs.readFileSync('/tmp/plotting/data.csv', 'utf8');

    var rows = data.split('\n');
    var index = rows[0].length + 1; // position in the text

    for (var i = 1; i < rows.length; i++) {
        var sub_str = rows[i].split(',');

        if (sub_str.length > 8) {
            updated = 1;
            var p = 0;
            var id = 0;

            while (id != rows[i].length) {
                if (rows[i][id] === ',' && p === 7) {
                    rows[i] = rows[i].substr(0, id) + '\n' + rows[i].substr(id + 1);
                    console.log('new row = ' + rows[i])
                    // break;
                } else if (rows[i][id] === ',') {
                    p++;
                }
                id++;
            }
        }
    }
    if (updated) {
        rows[0] += '\n';
        var t = '';
        for (var j = 0; j < rows.length; j++) {
            // rows[j] = rows[j].substr(1);
            rows[j] += '\n';
            t += rows[j];
        }

        fs.unlinkSync('/tmp/plotting/data.csv');
        fs.writeFileSync('/tmp/plotting/data.csv', t);
    }
}

// used to disable escape key (can be used for a shortcut)
export function disable_keys(keys: string, window: QWidget) {
    const shortcut = new QShortcut(window);
    shortcut.setKey(new QKeySequence(keys));
    shortcut.setEnabled(true);
    shortcut.addEventListener("activated", () => { });
}